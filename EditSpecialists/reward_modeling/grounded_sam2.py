import numpy as np
import cv2
import torch

def masks_from_gSAM2(image, text, h, w, detector_processor, object_detector, segmentator, device, use_multimask=False):
    """
    Generate binary segmentation masks using Grounded SAM2 model.
    This function processes input images and text prompts to detect objects and generate 
    segmentation masks. It first uses an object detector to identify regions based on 
    the text prompt, then applies SAM2 to create detailed masks for these regions.
    
    Args
    ----
    image : torch.Tensor
        Input image tensor with shape (B, C, H, W) or (C, H, W)
    text : str or list
        Text prompt(s) describing the object(s) to detect
    h : int
        Height of the original image
    w : int
        Width of the original image
    detector_processor : object
        Processor for the object detector model
    object_detector : object
        Pre-trained object detection model
    segmentator : object
        SAM2 segmentation model
    device : torch.device
        Device to run the models on
    use_multimask : bool, default=False
        Whether to generate multiple mask proposals for each detection
    
    Returns
    -------
    torch.Tensor
        Binary segmentation masks with shape (B, H, W)
    """
    
    if len(image.shape) == 3: image = image.unsqueeze(0)
    text = text+"." if isinstance(text, str) else [i+"." for i in text]
    # Detect
    detect_inputs = detector_processor(images=image, text=text, padding=True, return_tensors="pt", do_rescale=False, do_resize=True).to(device)
    detections = object_detector(**detect_inputs)
    detections_results = detector_processor.post_process_grounded_object_detection(
        detections,
        detect_inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[(h, w) for it in range(image.shape[0])],
    )
    # Extract bounding boxes and pad if unequal nb of detections across all images in batch
    bounding_boxes = [result["boxes"].tolist() if result["boxes"].shape[0] > 0 else [[0, 0, w, h]] for result in detections_results]
    max_nb_boxes = max([len(i) for i in bounding_boxes])
    bounding_boxes = [i + [i[-1]]*(max_nb_boxes - len(i)) for i in bounding_boxes] 
    
    # Segment with SAM-2
    segmentator.set_image_batch([i for i in image.cpu().numpy().transpose(0, 2, 3, 1)])
    masks, scores, logits = segmentator.predict_batch(None, None, box_batch=bounding_boxes, multimask_output=use_multimask)

    # Overlay all masks (if multimask) and all bboxes into one
    if use_multimask:
        temp = torch.cat([torch.from_numpy(np_array) for np_array in masks]) if len(masks[0].shape) == 4 else torch.stack([torch.from_numpy(np_array) for np_array in masks]) # sam2 output has different shape with multimask or just one
        masks = torch.from_numpy(np.stack(refine_masks(temp, True))).reshape(-1, max_nb_boxes, h, w).any(dim=1)
    else:
        masks = torch.from_numpy(np.concatenate(masks, 1).any(0)) if len(masks[0].shape) == 4 else torch.cat([torch.from_numpy(np_array) for np_array in masks])

    return masks


def refine_masks(masks,polygon_refinement=False):
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)
    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask
    return masks

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask
