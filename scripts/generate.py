import os
import datetime
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
import torchvision.transforms as T
import numpy as np
from EditSpecialists.diffusion_pipeline.ip2p_pipeline import instruct_pix2pix_pipeline_with_logprob
from EditSpecialists.diffusion_pipeline.architecture import CA_PreconvModule
from EditSpecialists.data.dataset import CustomImageDataset
import torch
from functools import partial
import tqdm
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import DataLoader

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")



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
    else:
        print(f"No checkpoint defined in config, using base InstructPix2Pix weights")
    
    print(f"\n{config}")
    # set seed (device_specific is very important to get different prompts on different devices)
    device_seed = np.random.randint(0,100000,size=1)[0]
    random.seed(device_seed)
    np.random.seed(device_seed)
    torch.manual_seed(device_seed)
    torch.cuda.manual_seed_all(device_seed)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision) 
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=False,
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
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline.vae.to(device, dtype=inference_dtype)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.unet.to(device, dtype=inference_dtype)
    pipeline.ca_preconv.to(device, dtype=inference_dtype)

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
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1).to(device)
    
    # Dataloader
    sampling_transform = T.Compose([T.Resize(config.sample.data_resize),
                                    T.CenterCrop((config.sample.data_resize, config.sample.data_resize)),
                                    T.ToTensor()])
    dataset = CustomImageDataset(input_dataset_path=config.inputimg_dir, style_dataset_path=config.styleimg_dir, input_name=config.input_name, style_name=config.style_name, transform=sampling_transform, is_inference=True)
    sampling_dataloader = DataLoader(dataset, batch_size=config.sample.batch_size, shuffle=False)

    # Load checkpoint
    if config.load_checkpoint:
        state_dict = torch.load(config.load_checkpoint, map_location="cpu")
        pipeline.unet.load_state_dict(state_dict["unet"], strict=True)
        pipeline.ca_preconv.load_state_dict(state_dict["ca_preconv"], strict=True)
        print(f"Loading checkpoint for sampling: {config.load_checkpoint}")

    to_image = T.ToPILImage()
    pipeline.unet.eval()
    # Inference Loop
    for itt, (input_images, sty_images, prompts, input_imgnames) in tqdm(enumerate(sampling_dataloader)):
        out_path = os.path.join(config.sample_savedir, f"{config.input_name}_{config.style_name}")
        os.makedirs(out_path, exist_ok=True)

        # Encode prompts with CLIP
        prompt_ids_multi = pipeline.text_encoder(pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(device))[0]

        # Encode input and style images through VAE
        with torch.autocast(device_type=device), torch.no_grad():
            input_image_latents = (2 * input_images.float() - 1).to(device)
            input_image_latents = pipeline.vae.encode(input_image_latents)['latent_dist'].mode().to(inference_dtype)
            style_image_latents = (2 * sty_images.float() - 1).to(device)
            style_image_latents = pipeline.vae.encode(style_image_latents)['latent_dist'].mode().to(inference_dtype)
            noise = torch.randn_like(input_image_latents)

            # Sample
            images_gen, _, _, _ = instruct_pix2pix_pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_ids_multi,
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

        # Save images
        for img, name in zip(images_gen, input_imgnames):
            to_image(img).save(os.path.join(out_path, "edit_" + name))

    print(f"Generation complete, generated {len(os.listdir(out_path))} files in {out_path}")


if __name__ == "__main__":
    app.run(main)
