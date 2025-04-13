from collections import defaultdict
import os
import copy
import datetime
from concurrent import futures
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
import torchvision.transforms as T
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

import EditSpecialists.reward_modeling.score_functions
from EditSpecialists.diffusion_pipeline.ddim_with_logprob import ddim_step_with_logprob
from EditSpecialists.diffusion_pipeline.eulerancestral_withlogprob import eulerancestral_step_with_logprob
from EditSpecialists.diffusion_pipeline.ip2p_pipeline import instruct_pix2pix_pipeline_with_logprob
from EditSpecialists.diffusion_pipeline.architecture import CA_PreconvModule, single_noise_pred
from EditSpecialists.data.dataset import SamplingDataset

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.load_checkpoint:
        config.load_checkpoint = os.path.normpath(os.path.expanduser(os.path.join(config.checkpoint_path, config.load_checkpoint)))
        if "checkpoint-" not in os.path.basename(config.load_checkpoint):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint-" in x, os.listdir(config.load_checkpoint)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.load_checkpoint}")
            config.load_checkpoint = os.path.join(
                config.load_checkpoint,
                sorted(checkpoints, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="EditSpecialists", config=config.to_dict(), 
            init_kwargs={"wandb": {"name": config.run_name}}
        )
    logger.info(f"\n{config}")
    # set seed (device_specific is very important to get different prompts on different devices)
    np.random.seed(config.seed)
    available_devices = accelerator.num_processes
    random_seeds = np.random.randint(0,100000,size=available_devices)
    device_seed = random_seeds[accelerator.process_index]
    set_seed(device_seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision) 
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    if not config.use_lora and config.train.activation_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to the correct scheduler
    if config.pretrained.scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif config.pretrained.scheduler == "euler_ancestral":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError("Scheduler should be one of `ddim` or `euler_ancestral` ")

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # Add Cross-Attention before the U-Net conv_in layer
    pipeline.ca_preconv = CA_PreconvModule(inside_dim=320, nb_heads=8)
    with torch.no_grad():
        # convert to_k, to_v with the weights of attn2 down_blocks[0] 
        pipeline.ca_preconv.CA_Preconv_to_k.weight.copy_(pipeline.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k.weight)
        pipeline.ca_preconv.CA_Preconv_to_v.weight.copy_(pipeline.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_v.weight)
        pipeline.ca_preconv.CA_Preconv_to_q.weight.copy_(torch.stack(pipeline.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_v.weight.chunk(8, -1)).mean(-1).T)
    # Adjust the U-Net conv_in layer
    pretrained_weights = pipeline.unet.conv_in.weight.data
    pretrained_bias = pipeline.unet.conv_in.bias.data
    new_conv = nn.Conv2d(12, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    with torch.no_grad():
        # Copy the existing weights to the new layer
        new_conv.weight[:, :8, :, :] = pretrained_weights
        # Initialize the new weights (the additional 4 input channels) to zero
        new_conv.weight[:, 8:, :, :] = torch.zeros((320, 4, 3, 3))
        # Copy the bias as it remains the same (shape=320)
        new_conv.bias = nn.Parameter(pretrained_bias)
    # replace old with new
    pipeline.unet.conv_in = new_conv

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype if config.use_lora else torch.float32)
    ref = copy.deepcopy(pipeline.unet)
    for param in ref.parameters():
        param.requires_grad = False

    if config.use_lora:
        UNET_TARGET_MODULES = ["to_q", "to_v", "to_k", "to_out.0"]
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=8,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=0.0,
            bias='none',
        )
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)
        for param in pipeline.unet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    # new (non-LoRA) parameters CA_preconv and conv_in are trained in float32
    pipeline.unet.conv_in.bias.requires_grad = True
    pipeline.unet.conv_in.weight.requires_grad = True
    for param in pipeline.unet.conv_in.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    pipeline.ca_preconv.to(accelerator.device, dtype=torch.float32)
    for param in pipeline.ca_preconv.parameters():
        param.requires_grad = True
        param.data = param.data.to(torch.float32)
    
    trainable_layers = pipeline.unet
    trainable_layers_CApreconv = pipeline.ca_preconv
    ref_CApreconv = copy.deepcopy(pipeline.ca_preconv)
    for param in ref_CApreconv.parameters():
        param.requires_grad = False

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Define which parameters must be trained or kept frozen
    trainable_parameters = list(trainable_layers.parameters()) + list(trainable_layers_CApreconv.parameters())
    optimizer = optimizer_cls(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare score functions
    structural_fn = getattr(EditSpecialists.reward_modeling.score_functions, "structural_score")(device=accelerator.device)
    semantic_fn = getattr(EditSpecialists.reward_modeling.score_functions, "semantic_score")(device=accelerator.device)
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = accelerator.autocast
    
    # Dataloader
    sampling_transform = T.Compose([T.Resize(config.sample.data_resize),
                                    T.CenterCrop((config.sample.data_resize, config.sample.data_resize)),
                                    T.ToTensor()])
    dataset = SamplingDataset(input_dataset_path=config.inputimg_dir, style_dataset_path=config.styleimg_dir, input_name=config.input_name, style_name=config.style_name, transform=sampling_transform, is_inference=False)
    sampling_dataloader = DataLoader(dataset, batch_size=config.sample.batch_size, shuffle=True)

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer, sampling_dataloader = accelerator.prepare(trainable_layers, optimizer, sampling_dataloader)
    trainable_layers_CApreconv = accelerator.prepare(trainable_layers_CApreconv)
    accelerator.gradient_state.plugin_kwargs["sync_with_dataloader"] = False # Trick to avoid accelerator to do weight update when every data in dataloader is seen because not aligned with nb_timesteps etc
    # executor to perform callbacks asynchronously.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.load_checkpoint:
        state_dict = torch.load(config.load_checkpoint, map_location="cpu")
        pipeline.unet.load_state_dict(state_dict["unet"], strict=True)
        pipeline.ca_preconv.load_state_dict(state_dict["ca_preconv"], strict=True)
        first_epoch = int(config.load_checkpoint.split("-")[-1].split(".")[0]) + 1
        logger.info(f"Resuming from: {config.load_checkpoint}")
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0
            ):

            # Sample images and edit prompts
            in_images, sty_images, prompts = next(iter(sampling_dataloader))

            # Encode prompts with CLIP
            prompt_ids_multi = [pipeline.text_encoder(pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device))[0]
                for i in range(config.sample.nb_gen)]

            # Encode input and style images through VAE
            with torch.autocast(device_type="cuda"):
                input_image_latents = 2 * in_images.float() - 1
                input_image_latents = pipeline.vae.encode(input_image_latents)['latent_dist'].mode().to(inference_dtype)
                style_image_latents = 2 * sty_images.float() - 1
                style_image_latents = pipeline.vae.encode(style_image_latents)['latent_dist'].mode().to(inference_dtype)
            noise = torch.randn_like(input_image_latents)

            # Sample
            with autocast(), torch.no_grad():
                gen_images = []
                latents = []
                log_probs = []
                for it in range(config.sample.nb_gen):
                    images_it, _, latents_it, log_probs_it = instruct_pix2pix_pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_ids_multi[it],
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        image_guidance_scale=config.sample.image_guidance_scale,
                        output_type="pt",
                        latents=noise,
                        image=input_image_latents,
                        style_image=style_image_latents,
                        style_image_guidance_scale=config.sample.style_image_guidance_scale,
                    )
                    gen_images.append(images_it)
                    latents.append(torch.stack(latents_it, dim=1))
                    log_probs.append(torch.stack(log_probs_it, dim=1))

            # Save samples for evaluation and model training
            latents = torch.stack(latents, dim=1)  # (batch_size, sample.nb_gen, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            prompt_embeds = torch.stack(prompt_ids_multi, dim=1)
            gen_images = torch.stack(gen_images, dim=1)
            current_latents = latents[:, :, :-1]
            next_latents = latents[:, :, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # Compute scores asynchronously
            # Structural Score
            L_struct_gen = [executor.submit(structural_fn, gen_images[:, i]).result() for i in range(gen_images.shape[1])]
            L_struct_input = executor.submit(structural_fn, in_images).result()
            L_struct = [torch.nn.L1Loss(reduction='none')(L_struct_input, L_i).mean(dim=(1, 2)).cpu().detach() for L_i in L_struct_gen]
            L_struct = torch.stack(L_struct, axis=1) # (batch_size, sample.nb_gen)
            # Semantic Score
            L_sem = [executor.submit(semantic_fn, gen_images[:, i], sty_images, in_images, prompts, rec_lambda=config.train.rec_lambda, whole_frame=config.train.whole_frame).result() for i in range(gen_images.shape[1])]
            L_sem = torch.stack(L_sem, axis=1)

            # Convert Score into Advantage
            L_struct = L_struct.to(accelerator.device)
            avd_struct = accelerator.gather(L_struct)
            avd_struct = (avd_struct - avd_struct.mean()) / (avd_struct.std() + 1e-8)
            L_sem = L_sem.to(accelerator.device)
            adv_sem = accelerator.gather(L_sem)
            adv_sem = (adv_sem - adv_sem.mean()) / (adv_sem.std() + 1e-8)
            
            # Preference Ranking
            preference_gathered = torch.argsort(avd_struct + config.train.score_alpha * adv_sem , axis=1)
            preference = torch.as_tensor(preference_gathered).reshape(accelerator.num_processes, -1, config.sample.nb_gen)[accelerator.process_index].to(accelerator.device)

            # Store information
            prompts = list(prompts)
            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "prompts": prompts,
                    "timesteps": timesteps,
                    "input_image_latents": input_image_latents,
                    "style_image_latents": style_image_latents,
                    "latents": current_latents,  # each entry is the latent before timestep t
                    "next_latents": next_latents,  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "images": gen_images,
                    "L_struct": torch.as_tensor(L_struct, device=accelerator.device),
                    "L_sem": torch.as_tensor(L_sem, device=accelerator.device),
                    "preference": torch.as_tensor(preference, device=accelerator.device),
                }
            )
        
        prompts = samples[-1]["prompts"]
        del samples[0]["prompts"]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}        
        images = samples["images"]
        L_semantic = accelerator.gather(samples["L_sem"]).cpu().numpy()
        L_structural = accelerator.gather(samples["L_struct"]).cpu().numpy()
        # Log scores
        accelerator.log(
            {"num_samples": epoch*available_devices*config.sample.batch_size,
             "semantic_score": wandb.Histogram(L_semantic), "semantic_mean": L_semantic.mean(), "semantic_std": L_semantic.std(),
             "structural_score": wandb.Histogram(L_structural), "structural_mean": L_structural.mean(), "structural_std": L_structural.std(),
            },
            step=global_step,
        )
        # Log last batch of images
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (image_gen, image_in, image_sty) in enumerate(zip(images[-in_images.shape[0]:], in_images, sty_images)):
                pil = Image.fromarray((image_gen[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}_gen.jpg"))
                pil = Image.fromarray((image_in.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}_in.jpg"))
                pil = Image.fromarray((image_sty.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)) 
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}_sty.jpg"))

            accelerator.log(
                {
                    "images_gen": [
                        wandb.Image(os.path.join(tmpdir, f"{i}_gen.jpg"), caption=f"{prompt:.25} | {reward:.2f} | {depth:.2f}")
                        for i, (prompt, reward, depth) in enumerate(zip(prompts, samples["L_sem"][-in_images.shape[0]:,0], samples["L_struct"][-in_images.shape[0]:,0]))
                    ],
                    "images_in": [
                        wandb.Image(os.path.join(tmpdir, f"{i}_in.jpg"), caption=f"{prompt:.25} | {reward:.2f} | {depth:.2f}")
                        for i, (prompt, reward, depth) in enumerate(zip(prompts, samples["L_sem"][-in_images.shape[0]:,0], samples["L_struct"][-in_images.shape[0]:,0]))
                    ],
                    "images_sty": [
                        wandb.Image(os.path.join(tmpdir, f"{i}_sty.jpg"), caption=f"{prompt:.25} | {reward:.2f} | {depth:.2f}")
                        for i, (prompt, reward, depth) in enumerate(zip(prompts, samples["L_sem"][-in_images.shape[0]:,0], samples["L_struct"][-in_images.shape[0]:,0]))
                    ],
                },
                step=global_step,
            )

        del samples["images"]
        torch.cuda.empty_cache()
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps
        init_samples = copy.deepcopy(samples)

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in init_samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["latents", "next_latents"]:
                tmp = samples[key].permute(0,2,3,4,5,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
                samples[key] = tmp.permute(0,5,1,2,3,4)
            samples["timesteps"] = samples["timesteps"][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms].unsqueeze(1).repeat(1, samples["latents"].shape[1], 1)
            tmp = samples["log_probs"].permute(0,2,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
            samples["log_probs"] = tmp.permute(0,2,1)
            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i in tqdm(range(0,total_batch_size,config.train.batch_size),
                        desc="Update",
                        position=2,
                        leave=False, 
                          ):
                
                # Assemble samples
                pref_perm = samples["preference"][i:i+config.train.batch_size]
                multi_samples = []
                embeds = []
                for sample_it in range(samples['prompt_embeds'].shape[1]):
                    multi_samples.append({})
                    for key, value in samples.items():
                        if key not in ["input_image_latents", "style_image_latents", "preference"]: # shared among all samples so saved once 
                            multi_samples[sample_it][key] = value[i:i+config.train.batch_size, sample_it]

                    # Prepare prompt embeddings
                    prompt_cat = torch.cat([multi_samples[sample_it]["prompt_embeds"], train_neg_prompt_embeds, train_neg_prompt_embeds])
                    prompt_cat = torch.cat([prompt_cat, train_neg_prompt_embeds])
                    embeds.append(prompt_cat) 

                # Prepare image latents
                input_image_latent = samples["input_image_latents"][i:i+config.train.batch_size] 
                input_image_latents_all = torch.cat([input_image_latent, input_image_latent, torch.zeros_like(input_image_latent), input_image_latent], dim=0)
                style_image_latent = samples["style_image_latents"][i:i+config.train.batch_size]

                loss_all_ts = []
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=3,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):  
                    models_to_accumulate = trainable_layers
                    with accelerator.accumulate(models_to_accumulate), autocast():
                        # Reset error detection variable
                        is_err = torch.tensor(False).to(accelerator.device) 

                        # Noise prediction forward passes
                        noise_preds = [single_noise_pred(sample=sample_i, 
                                                            embeds=embeds_i, 
                                                            scheduler=pipeline.scheduler, 
                                                            unet=trainable_layers,
                                                            timestep=j, 
                                                            image=input_image_latents_all, 
                                                            guidance_scale=config.sample.guidance_scale, 
                                                            image_guidance_scale=config.sample.image_guidance_scale,
                                                            style_image=style_image_latent,
                                                            style_image_guidance_scale=config.sample.style_image_guidance_scale,
                                                            ca_preconv=trainable_layers_CApreconv,
                                                            )
                                        for sample_i, embeds_i in zip(multi_samples, embeds)]
                        noise_ref_preds = [single_noise_pred(sample=sample_i, 
                                                                embeds=embeds_i, 
                                                                scheduler=pipeline.scheduler, 
                                                                unet=ref,
                                                                timestep=j, 
                                                                image=input_image_latents_all, 
                                                                guidance_scale=config.sample.guidance_scale, 
                                                                image_guidance_scale=config.sample.image_guidance_scale,
                                                                style_image=style_image_latent,
                                                                style_image_guidance_scale=config.sample.style_image_guidance_scale,
                                                                ca_preconv=ref_CApreconv,
                                                                )
                                            for sample_i, embeds_i in zip(multi_samples, embeds)]

                        # compute the log prob of next_latents given latents under the current model
                        if isinstance(pipeline.scheduler, DDIMScheduler):
                            total_probs = torch.stack([ddim_step_with_logprob(pipeline.scheduler, 
                                                                            noise_pred, 
                                                                            multi_samples[it]["timesteps"][:, j], 
                                                                            multi_samples[it]["latents"][:, j], 
                                                                            eta=config.sample.eta,
                                                                            prev_sample=multi_samples[it]["next_latents"][:, j],
                                                                            )[1]
                                                        for it, noise_pred in enumerate(noise_preds)])
                            total_ref_probs = torch.stack([ddim_step_with_logprob(pipeline.scheduler, 
                                                                            noise_ref_pred, 
                                                                            multi_samples[it]["timesteps"][:, j], 
                                                                            multi_samples[it]["latents"][:, j], 
                                                                            eta=config.sample.eta,
                                                                            prev_sample=multi_samples[it]["next_latents"][:, j],
                                                                            )[1]
                                                        for it, noise_ref_pred in enumerate(noise_ref_preds)])
                        elif isinstance(pipeline.scheduler, EulerAncestralDiscreteScheduler):
                            # EulerAncestral has variance==0 for timestep 0, causing numerical issues with gradient
                            curr_ts = multi_samples[0]["timesteps"][:, j]
                            if torch.tensor([0.]).cuda() in curr_ts:
                                is_err = torch.tensor(True).to(accelerator.device)
                            
                            total_probs = torch.stack([eulerancestral_step_with_logprob(pipeline.scheduler, 
                                                                            noise_pred, 
                                                                            multi_samples[it]["timesteps"][:, j], 
                                                                            multi_samples[it]["latents"][:, j], 
                                                                            prev_sample=multi_samples[it]["next_latents"][:, j],
                                                                            )[1]
                                                        for it, noise_pred in enumerate(noise_preds)])
                            total_ref_probs = torch.stack([eulerancestral_step_with_logprob(pipeline.scheduler, 
                                                                            noise_ref_pred, 
                                                                            multi_samples[it]["timesteps"][:, j], 
                                                                            multi_samples[it]["latents"][:, j], 
                                                                            prev_sample=multi_samples[it]["next_latents"][:, j],
                                                                            )[1]
                                                            for it, noise_ref_pred in enumerate(noise_ref_preds)])
                    
                        # D3PO Preference Loss
                        ratios = torch.clamp(torch.exp(total_probs - total_ref_probs), 1 - config.train.eps, 1 + config.train.eps)
                        ratios = torch.exp(config.train.beta * torch.log(ratios))

                        # Check for NaN or inf
                        if torch.isnan(ratios).any() or torch.isinf(ratios).any() or (
                                torch.isnan(torch.exp(total_probs - total_ref_probs)).any() or torch.isinf(torch.exp(total_probs - total_ref_probs)).any()
                            ):
                            is_err = torch.tensor(True).to(accelerator.device)
                        
                        # Compute preference loss
                        permuted_ratios = torch.gather(ratios.T, 1, pref_perm) # Permute samples in order of their preferences
                        loss = 0.
                        for it in range(permuted_ratios.shape[1]):
                            loss_it = torch.log(permuted_ratios[:, it]) - torch.log(torch.sum(permuted_ratios[:, it:], dim=1))
                            loss -= torch.mean(loss_it)

                        global_err = torch.any(accelerator.gather(is_err)).item()
                        if not global_err:
                            # Logging
                            loss_all_ts.append(loss.detach().cpu().numpy())
                            if (j == num_train_timesteps - 1):
                                accelerator.log(
                                    {"loss": np.mean(loss_all_ts)},
                                    step=global_step)

                            # Update weights
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                unclipped_grad = accelerator.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                        else:
                            if accelerator.sync_gradients:
                                unclipped_grad = accelerator.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                            # Synchronize model wights across GPUs because they fall out of sync when err happens on iteration where weights should be updated
                            if accelerator.sync_gradients and accelerator.num_processes > 1:
                                accelerator.wait_for_everyone()
                                # Synchronize to one of the gpus with no error, else main_process
                                base_idx = torch.where(~accelerator.gather(is_err))[0][0] if torch.tensor(False).to(accelerator.device) in accelerator.gather(is_err) else 0
                                for param in pipeline.unet.parameters():
                                    torch.distributed.broadcast(param.data, src=base_idx)
                        
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + config.train.batch_size
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

                        # make sure we did an optimization step at the end of the inner epoch
                        assert accelerator.sync_gradients
        
        # Save weights
        if epoch > 0 and (epoch+1) % config.save_freq == 0 and accelerator.is_main_process:
            os.makedirs(os.path.join(config.checkpoint_path, config.run_name), exist_ok=True)
            save_path = os.path.join(config.checkpoint_path, config.run_name, f"checkpoint-{epoch}.pth")
            
            # delete earliest checkpoints if nb > max files
            files = sorted(os.listdir(os.path.join(config.checkpoint_path, config.run_name)), key=lambda x: int(x.split("-")[1].split(".")[0]))
            if (config.num_checkpoint_limit > 0) and (len(files) > config.num_checkpoint_limit):
                os.remove(os.path.join(config.checkpoint_path, config.run_name, files[0]))

            # Save new checkpoint
            torch.save({"unet": accelerator.unwrap_model(pipeline.unet).state_dict(),
                        "ca_preconv": accelerator.unwrap_model(pipeline.ca_preconv).state_dict()},
                        save_path)
            logger.info(f"Saved state to {save_path}")

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    app.run(main)
