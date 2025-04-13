# Functions adapted from https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py & https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/util/transform.py
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import torch.nn.functional as F
import torch
from torchvision.transforms import Normalize
from torchvision.transforms import Compose
import numpy as np

def load_depth_model(encoder='vitl', device='cpu'):
    """
    This function initializes a DepthAnythingV2 model with the specified encoder 
    architecture, loads its pre-trained weights, and places it on the specified device.
    Parameters:
        encoder (str): The encoder architecture to use. Options are:
                       'vitl' (ViT-Large, default)
                       'vits' (ViT-Small)
                       'vitb' (ViT-Base)
                       'vitg' (ViT-Giant)
        device (str): The device to load the model onto ('cpu', 'cuda', etc.). Default is 'cpu'.
    Returns:
        model: A loaded and evaluation-ready DepthAnythingV2 model.
    """
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'/n/home10/ebenarous/code/VLM/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    return model

@torch.no_grad()
def infer_image(depth_model, raw_image, input_size=518):
    """
    Performs depth inference on a given image batch using a depth estimation model.
    Args:
        depth_model: The depth estimation model to use for inference.
        raw_image (torch.tensor): The input image batch for depth estimation.
        input_size (int, optional): The size to which the image will be resized for processing. Defaults to 518.
    Returns:
        torch.Tensor: The predicted depth map, resized to match the original image dimensions.
    """

    # expecting torch.tensor normalized [0, 1], and batch processing [bs, h, w]
    image, (h, w) = image2tensor(raw_image, input_size)
    
    # either autocast to compute on torch.half or bring image up to float32
    if image.dtype == torch.float16:
        with torch.autocast(device_type="cuda"):
            depth = depth_model.forward(image) 
        depth = depth.to(torch.float32)
    else:
        depth = depth_model.forward(image) 

    depth = F.interpolate(depth.unsqueeze(0), (h, w), mode="bilinear", align_corners=True)[0]

    return depth

def image2tensor(raw_image, input_size=518):
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method='bilinear',
        ),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    _, _, h, w = raw_image.shape
    image = transform(raw_image)
    
    return image, (h, w)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method="bilinear",
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
            image_interpolation_method (str, optional):
                Interpolation method for resizing. Defaults to "bilinear".
        """
        self.width = width
        self.height = height

        self.keep_aspect_ratio = keep_aspect_ratio
        self.multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.multiple_of) * self.multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.multiple_of) * self.multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.multiple_of) * self.multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.height / height
        scale_width = self.width / width

        if self.keep_aspect_ratio:
            if self.resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.resize_method == "minimal":
                # scale as least as possible
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.resize_method} not implemented")

        if self.resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.width)
        elif self.resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.width)
        elif self.resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        height, width = self.get_size(sample.shape[-2], sample.shape[-1])
        
        # resize sample
        sample = F.interpolate(sample, size=(height, width), mode=self.image_interpolation_method, align_corners=False)

        return sample