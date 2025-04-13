from .grounded_sam2 import masks_from_gSAM2
from .depth_anythingv2 import load_depth_model, infer_image
from torchvision import transforms

import numpy as np
import torch

###### Simple score ######
def light_reward():
    def _fn(images):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach())
    return _fn

###### Semantic score ######
def semantic_score(device):
    """
    Prepares the function to compute the semantic score.
    Args:
        device (torch.device): Device to run the models on (CPU or GPU)
    Returns:
        callable: A function that accepts:
            - gen_images (torch.Tensor): Generated images
            - sty_images (torch.Tensor): Style reference images
            - in_images (torch.Tensor): Input images
            - prompts (list): Text prompts describing objects to edit and style to apply
            - for_eval (bool, optional): Whether to return additional mask outputs for evaluation
            - whole_frame (bool, optional): Whether the editing prompt targets the entire image
            - rec_lambda (float, optional): Weight for reconstruction loss term (default: 0.5)
        The returned function outputs the semantic distance score (lower is better),
        and optionally returns segmentation masks when for_eval=True.
    """

    # Load Grounded-SAM2
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    detector_id = "IDEA-Research/grounding-dino-tiny"
    object_detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)
    detector_processor = AutoProcessor.from_pretrained(detector_id)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    segmentator = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    use_multimask = False
    resize_size = 16
    SAM_batchsize = 4 # adjust based on memory limitations
    
    # Load DreamSIM
    from dreamsim import dreamsim
    toPIL = transforms.ToPILImage()   
    if torch.cuda.device_count() > 1:
        dreamsim_model, dreamsim_preprocess = synchronized_dreamsim_loading(
            dreamsim,
            pretrained=True, 
            dreamsim_type="dinov2_vitb14", 
            device=device, 
            use_patch_model=True, 
            cache_dir="dreamsim_weights"
        )
    else:
        dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, dreamsim_type="dinov2_vitb14", device=device, use_patch_model=True, cache_dir="dreamsim_weights")

    def _fn(gen_images, sty_images, in_images, prompts, for_eval=False, whole_frame=False, rec_lambda=0.5):
        _, _, h, w = in_images.shape
        
        # Split prompts into generation and style prompts
        prompts_sty = []
        prompts_gen = []
        for prompt in prompts:
            words = prompt.split()
            prompts_gen.extend([words[2] if not for_eval else words[0]])
            prompts_sty.extend([words[-1]])
        split_prompts = prompts_gen + prompts_sty

        with torch.no_grad():
            # Segment objects specified in prompts
            nb_chunks = int(np.ceil(len(split_prompts) // SAM_batchsize))
            masks_patch = []
            masks_pixel = [] # full res masks for precise rec pixel masking
            for image, text in zip(torch.cat([in_images, sty_images]).chunk(nb_chunks), chunk_list(split_prompts, nb_chunks)):
                # For whole-image operations (ex. style transfer), select the entire frame
                if whole_frame:
                    masks = torch.ones((SAM_batchsize, h, w))
                
                # Extract masks from Grounded-SAM2
                else:
                    masks = masks_from_gSAM2(image, text, h, w, detector_processor, object_detector, segmentator, device, use_multimask=use_multimask)

                masks_pixel.append(masks)
                masks_downsize = torch.nn.functional.interpolate(masks.unsqueeze(0).to(torch.float32), size=(resize_size, resize_size), mode='nearest')[0]
                masks_patch.append(masks_downsize)

            masks_pixel_sty = torch.cat(masks_pixel)[gen_images.shape[0]:].to(torch.bool).to(device)
            masks_pixel_gen = torch.cat(masks_pixel)[:gen_images.shape[0]].to(torch.bool).to(device)
            masks_patch = torch.cat(masks_patch).reshape(len(split_prompts), -1).to(device)
            
            # If nothing was detected on an image, select everything to avoid div by zero later
            zero_rows = (masks_patch == 0).all(dim=1)
            masks_patch[zero_rows] = 1

            # Compute dreamsim patch embeddings
            sty_images_dreamsim = torch.cat([dreamsim_preprocess(toPIL(i)) for i in sty_images]).to(device)
            gen_images_dreamsim = torch.cat([dreamsim_preprocess(toPIL(i)) for i in gen_images]).to(device)
            features_gen = dreamsim_model.embed(gen_images_dreamsim)[:, 1:, :]
            features_sty = dreamsim_model.embed(sty_images_dreamsim)[:, 1:, :]

        # Split patch-level segmentation masks
        masks_patch_gen = masks_patch[:gen_images.shape[0]]
        masks_patch_sty = masks_patch[gen_images.shape[0]:]

        # Compute semantic score
        # cosine_sim(sty, gen)
        sim_sty_gen = torch.nn.functional.cosine_similarity(features_gen.unsqueeze(2), features_sty.unsqueeze(1), dim=-1)
        # Mean cosine distance (dist = 1 - sim)
        mean_cos_dist = ((1 - sim_sty_gen) * masks_patch_sty.unsqueeze(1)).sum(-1) / masks_patch_sty.unsqueeze(1).sum(-1)
        mean_cos_dist = mean_cos_dist * masks_patch_gen / masks_patch_gen.sum(-1, keepdim=True) 
        mean_cos_dist = torch.sum(mean_cos_dist, dim=-1)

        # Integrate the reconstruction term
        MSE_loss = torch.nn.MSELoss(reduction='none')(gen_images, in_images).mean(dim=1)
        masks_MSE = ~masks_pixel_gen
        # avoid div by zero
        zero_rows = (masks_pixel_gen == 0).all(dim=(1,2))
        masks_pixel_gen[zero_rows] = 1
        zero_rows = (masks_MSE == 0).all(dim=(1,2))
        masks_MSE[zero_rows] = 1
        mean_MSE_loss = (MSE_loss * masks_MSE).sum((1,2)) / masks_MSE.sum((1, 2))

        # Compute final semantic score
        mean_cos_dist += rec_lambda * mean_MSE_loss

        if for_eval:
            return mean_cos_dist.cpu(), masks_pixel_gen.cpu(), masks_pixel_sty.cpu()
        else:
            return mean_cos_dist.cpu()

    return _fn

def chunk_list(input_list, num_chunks):
    avg_chunk_size = len(input_list) // num_chunks
    remainder = len(input_list) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + avg_chunk_size + (1 if i < remainder else 0)
        chunks.append(input_list[start:end])
        start = end
    return chunks


###### Structural score ######
def structural_score(device):
    """
    Prepares the function to infer depth maps from images with a pre-trained depth estimation model.
    Args:
    device : str or torch.device
        The device to load the depth model on (e.g., 'cuda', 'cpu').
    Returns:
    callable
        A function that takes a batch of images and returns their corresponding depth maps.
    """

    depth_model = load_depth_model(encoder='vitl', device=device)

    def _fn(images):
        depth_maps = infer_image(depth_model, images)
        return depth_maps

    return _fn


def synchronized_dreamsim_loading(load_function, *args, **kwargs):
    """
    Ensures only one GPU downloads the model while others wait to avoid Error.
    
    Args:
        load_function: The function that loads (and possibly downloads) the model
        *args, **kwargs: Arguments to pass to the load_function
    
    Returns:
        The loaded model
    """
    from accelerate.utils import broadcast_object_list
    from accelerate.state import AcceleratorState
    
    # Get accelerator state
    state = AcceleratorState()
    
    # Create a flag to indicate if model is ready
    model_ready = False
    
    # Process 0 (first GPU) downloads the model
    if state.process_index == 0:
        try:
            model = load_function(*args, **kwargs)
            model_ready = True
        except Exception as e:
            print(f"Process {state.process_index}: Error loading model: {e}. Try running the script in a single-GPU mode first.")
    
    # Broadcast the ready status from process 0 to all processes
    model_ready = broadcast_object_list([model_ready])[0]
    
    # If process 0 failed to load the model, raise exception on all processes
    if not model_ready:
        raise RuntimeError("Model loading failed on the primary process. Try running the script in a single-GPU mode first.")
    
    # Synchronize all processes
    state.wait_for_everyone()
    
    # For processes other than 0, load the model now that files exist
    if state.process_index != 0:
        model = load_function(*args, **kwargs)
    
    return model