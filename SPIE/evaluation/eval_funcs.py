import torch
from transformers import AutoModel, CLIPProcessor, CLIPModel
from torchvision import transforms
from dreamsim import dreamsim
import os
import huggingface_hub

def eval_main(device="cuda"):
    """
    Prepares the main evaluation function.
    The returned function computes multiple evaluation metrics including:
    - L2 distance between input and generated images in non-masked regions
    - CLIP-based text-image alignment scores
    - CLIP directional similarity
    - CLIP patch-based similarity between generated images and both input/style images
    - DINOv2 patch-based similarity between generated images and both input/style images
    - DreamSim patch-based similarity between generated images and both input/style images
    
    Args
    ----
    device : str, default="cuda"
        Device to load and run the models on ('cuda' or 'cpu')
    
    Returns
    -------
    callable
        An evaluation function that takes the following parameters:
        - in_images: Input images tensor
        - gen_images: Generated (edited) images tensor
        - sty_images: Style reference images tensor
        - mask_sty: Binary mask indicating style regions
        - mask_input: Binary mask indicating input regions to be edited
        - prompts: Text prompts describing the edits
        And returns a dictionary containing all computed evaluation metrics.
    """
    
    # DINO
    dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
    normalize_dino = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # DreamSim
    toPIL = transforms.ToPILImage()
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, dreamsim_type="dinov2_vitb14", device=device, use_patch_model=True, cache_dir="dreamsim_weights")

    def _fn(in_images, gen_images, sty_images, mask_sty, mask_input, prompts):

        with torch.no_grad():
            mask_sty = mask_sty.to(device)
            mask_input = mask_input.to(device)
            prompts_in = [i.split()[0] for i in prompts] # prompts for input images
            
            # L2
            l2_dist = torch.nn.MSELoss(reduction='none')(in_images, gen_images).mean(dim=1)
            # flip mask and avoid div by zero
            mask_input_out = ~mask_input
            zero_rows = (mask_input_out == 0).all(dim=(1, 2))
            mask_input_out[zero_rows] = 1
            l2_out = ((l2_dist * mask_input_out).sum((1,2)) / mask_input_out.sum((1, 2))).mean()

            # CLIP features
            inputs = clip_processor(text=prompts_in+prompts, images=torch.cat([in_images, gen_images, sty_images]), return_tensors="pt", padding=True, do_rescale=False, do_resize=True, do_center_crop=False).to(device)
            img_fts = clip_model.get_image_features(inputs['pixel_values'])
            in_embeds = img_fts[:gen_images.shape[0]]
            gen_embeds = img_fts[gen_images.shape[0]:2*gen_images.shape[0]]
            patch_fts = clip_model.vision_model.post_layernorm(clip_model.vision_model(pixel_values=inputs['pixel_values']).last_hidden_state[:, 1:, :])
            patch_fts = torch.stack([clip_model.visual_projection(patch_fts[:, i, :]) for i in range(patch_fts.shape[1])]).transpose(0, 1)
            gen_embeds_patch = patch_fts[:gen_images.shape[0]]
            sty_embeds_patch = patch_fts[gen_images.shape[0]:2*gen_images.shape[0]]
            in_embeds_patch = patch_fts[2*gen_images.shape[0]:]

            # CLIP DIR
            text_embeddings = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            text_in = text_embeddings[:gen_images.shape[0]]
            text_gen = text_embeddings[gen_images.shape[0]:]
            diff_text = text_gen - text_in + 1e-8
            diff_text /= diff_text.norm(-1, keepdim=True)
            diff_img = gen_embeds - in_embeds + 1e-8
            diff_img /= diff_img.norm(-1, keepdim=True)
            clip_dir_persample = torch.nn.functional.cosine_similarity(diff_text, diff_img)
            clip_dir = clip_dir_persample.mean()

            # CLIP OUTPUT
            inputs = clip_processor(text=prompts, images=gen_images, return_tensors="pt", padding=True, do_rescale=False, do_resize=True, do_center_crop=False).to(device)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity score
            img_prompt_score_persample = torch.diagonal(logits_per_image) / 100
            img_prompt_score = img_prompt_score_persample.mean()
            
            # CLIP Patch Sim
            # downsize masks
            resize_size = 16
            mask_input_ds = torch.nn.functional.interpolate(mask_input.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            mask_sty_ds = torch.nn.functional.interpolate(mask_sty.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            mask_input_out_ds = torch.nn.functional.interpolate(mask_input_out.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            zero_rows = (mask_input_out_ds == 0).all(dim=1) # avoid div by zero
            mask_input_out_ds[zero_rows] = 1
            # compute similarities
            sim_patches_in = torch.nn.functional.cosine_similarity(gen_embeds_patch.unsqueeze(2), in_embeds_patch.unsqueeze(1), dim=-1)
            temp = (sim_patches_in * mask_input_out_ds.unsqueeze(1)).sum(-1) / mask_input_out_ds.unsqueeze(1).sum(-1) 
            clipSim_input = torch.nanmean((temp * mask_input_out_ds / mask_input_out_ds.sum(-1, keepdim=True)).sum(-1))
            sim_patches_sty = torch.nn.functional.cosine_similarity(gen_embeds_patch.unsqueeze(2), sty_embeds_patch.unsqueeze(1), dim=-1)
            temp = (sim_patches_sty * mask_sty_ds.unsqueeze(1)).sum(-1) / mask_sty_ds.unsqueeze(1).sum(-1) 
            clipSim_style = torch.nanmean((temp * mask_input_ds / mask_input_ds.sum(-1, keepdim=True)).sum(-1))

            # DINOv2 Patch Sim
            # downsize masks
            resize_size = 18
            mask_input_ds = torch.nn.functional.interpolate(mask_input.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            mask_sty_ds = torch.nn.functional.interpolate(mask_sty.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            mask_input_out_ds = torch.nn.functional.interpolate(mask_input_out.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            zero_rows = (mask_input_out_ds == 0).all(dim=1)
            mask_input_out_ds[zero_rows] = 1
            # compute patch features
            features_gen = dino_model(normalize_dino(in_images).to(device)).last_hidden_state[:, 1:, :]
            features_in = dino_model(normalize_dino(gen_images).to(device)).last_hidden_state[:, 1:, :]
            features_sty = dino_model(normalize_dino(sty_images).to(device)).last_hidden_state[:, 1:, :]
            # compute similarities
            sim_patches_in = torch.stack([torch.nn.functional.cosine_similarity(features_gen, features_in[:, i].unsqueeze(1), dim=-1) for i in range(features_in.shape[1])], 2) # Avoid OOM
            temp = (sim_patches_in * mask_input_out_ds.unsqueeze(1)).sum(-1) / mask_input_out_ds.unsqueeze(1).sum(-1) 
            dinoSim_input = torch.nanmean((temp * mask_input_out_ds / mask_input_out_ds.sum(-1, keepdim=True)).sum(-1))
            sim_patches_sty = torch.stack([torch.nn.functional.cosine_similarity(features_gen, features_sty[:, i].unsqueeze(1), dim=-1) for i in range(features_sty.shape[1])], 2) # Avoid OOM
            temp = (sim_patches_sty * mask_sty_ds.unsqueeze(1)).sum(-1) / mask_sty_ds.unsqueeze(1).sum(-1) 
            dinoSim_style = torch.nanmean((temp * mask_input_ds / mask_input_ds.sum(-1, keepdim=True)).sum(-1))

            # DreamSim Patch Sim
            # downsize masks
            resize_size = 16
            mask_input_ds = torch.nn.functional.interpolate(mask_input.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            mask_sty_ds = torch.nn.functional.interpolate(mask_sty.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            mask_input_out_ds = torch.nn.functional.interpolate(mask_input_out.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0].reshape(len(in_images), -1)
            zero_rows = (mask_input_out_ds == 0).all(dim=1)
            mask_input_out_ds[zero_rows] = 1
            # compute patch features
            sty_images_dreamsim = torch.cat([dreamsim_preprocess(toPIL(i)) for i in sty_images]).to(device)
            gen_images_dreamsim = torch.cat([dreamsim_preprocess(toPIL(i)) for i in gen_images]).to(device)
            in_images_dreamsim = torch.cat([dreamsim_preprocess(toPIL(i)) for i in in_images]).to(device)
            features_gen = dreamsim_model.embed(gen_images_dreamsim)[:, 1:, :]
            features_sty = dreamsim_model.embed(sty_images_dreamsim)[:, 1:, :]
            features_in = dreamsim_model.embed(in_images_dreamsim)[:, 1:, :]
            # compute similarities
            sim_patches_in = torch.stack([torch.nn.functional.cosine_similarity(features_gen, features_in[:, i].unsqueeze(1), dim=-1) for i in range(features_in.shape[1])], 2) # Avoid OOM
            temp = (sim_patches_in * mask_input_out_ds.unsqueeze(1)).sum(-1) / mask_input_out_ds.unsqueeze(1).sum(-1) 
            dreamsimSim_input = torch.nanmean((temp * mask_input_out_ds / mask_input_out_ds.sum(-1, keepdim=True)).sum(-1))
            sim_patches_sty = torch.stack([torch.nn.functional.cosine_similarity(features_gen, features_sty[:, i].unsqueeze(1), dim=-1) for i in range(features_sty.shape[1])], 2) # Avoid OOM
            temp = (sim_patches_sty * mask_sty_ds.unsqueeze(1)).sum(-1) / mask_sty_ds.unsqueeze(1).sum(-1) 
            dreamsimSim_style = torch.nanmean((temp * mask_input_ds / mask_input_ds.sum(-1, keepdim=True)).sum(-1))

        result_dict = {
            "l2_out": l2_out.cpu().item(),
            "clip_txt": img_prompt_score.cpu().item(),
            "clip_dir": clip_dir.cpu().item(),
            "clipSim_gen_sty": clipSim_style.cpu().item(),
            "clipSim_gen_input": clipSim_input.cpu().item(),
            "dinoSim_gen_sty": dinoSim_style.cpu().item(),
            "dinoSim_gen_input": dinoSim_input.cpu().item(),
            "dreamsimSim_gen_sty": dreamsimSim_style.cpu().item(),
            "dreamsimSim_gen_input": dreamsimSim_input.cpu().item()
        }

        return result_dict
    
    return _fn



def pickscore(device):
    """
    Prepares the function for computing the PickScore score between images and text prompts.
    
    Args
    ----
    device : torch.device
        The device (CPU or GPU) on which to perform the computation.
    
    Returns
    -------
    callable
        A function that takes:
            - images (list): A list of torch.tensor images to evaluate
            - prompts (list): A list of text prompts corresponding to the images
        And returns:
            - torch.Tensor: PickScore scores between the images and associated prompts
    """
    from transformers import AutoProcessor, AutoModel
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    
    def _fn(images, prompts):
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        return scores.cpu()
    
    return _fn


def imagereward(device):
    """
    Prepares the function for computing the ImageReward score between images and text prompts.
    
    Args
    ----
    device : torch.device
        The device (CPU or GPU) on which to perform the computation.
    
    Returns
    -------
    callable
        A function that takes:
            - images (list): A list of torch.tensor images to evaluate
            - prompts (list): A list of text prompts corresponding to the images
        And returns:
            - torch.Tensor: ImageReward scores between the images and associated prompts
    """
    import ImageReward as reward
    model = reward.load("ImageReward-v1.0", device=device)

    toPIL = transforms.ToPILImage()

    def _fn(images, prompts):
        
        images = [toPIL(images[i]) for i in range(images.shape[0])]
        with torch.no_grad():
            _, rewards = model.inference_rank(prompts, images)
            
            sqrt_dim = int(len(rewards) ** 0.5)
            reshaped_rewards = torch.diagonal(torch.as_tensor(rewards).reshape(sqrt_dim, sqrt_dim))
        
        return reshaped_rewards.cpu()

    return _fn        


def hpsv2(device, hps_version="v2.1"):
    """
    Prepares the function for computing Human Preference Score (HPSv2) between images and text prompts.

    Args
    ----
    device : torch.device
        The device (CPU or GPU) on which to perform the computation.
    hps_version: str, optional
        Version of the HPS model to use. Options are "v2.0" or "v2.1". 
        Defaults to "v2.1".
    
    Returns
    -------
    callable
        A function that takes:
            - images (list): A list of torch.tensor images to evaluate
            - prompts (list): A list of text prompts corresponding to the images
        And returns:
            - torch.Tensor: HPS scores between the images and associated prompts
    """
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path, hps_version_map
    toPIL = transforms.ToPILImage()

    # Initialize the model
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    def hpsv2_score(image, prompt):
        with torch.no_grad():
            # Process the image
            image = preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return [hps_score[0]]

    def _fn(images, prompts):
        with torch.no_grad():
            result = [hpsv2_score(toPIL(images[i]), prompts[i]) for i in range(images.shape[0])]

        return torch.tensor(result).squeeze().cpu()

    return _fn
