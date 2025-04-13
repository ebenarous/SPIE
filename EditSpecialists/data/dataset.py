from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image
import os

# dataset for sampling
class SamplingDataset(Dataset):
    def __init__(self, input_dataset_path, style_dataset_path, input_name, style_name, transform=None, is_inference=False):
        self.input_dataset_path = input_dataset_path
        self.style_dataset_path = style_dataset_path
        self.input_imgs = np.asarray([i for i in os.listdir(self.input_dataset_path) if i.endswith(('.jpg', '.jpeg', '.png'))])
        self.style_imgs = np.asarray([i for i in os.listdir(self.style_dataset_path) if i.endswith(('.jpg', '.jpeg', '.png'))])
        self.input_name = input_name
        self.style_name = style_name
        
        self.transform = transform
        self.is_inference = is_inference

        self.prompt_formulations = [
            "turn the {} into {}",
            "replace the {} with {}",
            "transform the {} into {}",
            "convert the {} to {}",
            "change the {} to {}",
            "morph the {} into {}",
            "modify the {} into {}",
            "swap the {} for {}",
        ]

        print(f"Total number of images used for {'training' if not is_inference else 'generation'} is: {len(self.input_imgs)}")

    def __len__(self):
        return len(self.input_imgs)
    
    def _generate_prompt(self):
        prompt_template = random.choice(self.prompt_formulations)
        prompt = prompt_template.format(self.input_name, self.style_name)
        return prompt

    def __getitem__(self, idx):
        # Load data (style image and text prompt are randomly selected)
        in_image_path = os.path.join(self.input_dataset_path, self.input_imgs[idx])
        rdm_idx = np.random.randint(0, len(self.style_imgs))
        sty_image_path = os.path.join(self.style_dataset_path, self.style_imgs[rdm_idx])
        prompt = self._generate_prompt()

        # Open and preprocess Images
        sty_image = Image.open(sty_image_path).convert('RGB')
        in_image = Image.open(in_image_path).convert('RGB')
        sty_image = self.transform(sty_image)
        in_image = self.transform(in_image)
        
        if self.is_inference:
            # include the name of the input image during inference
            return in_image, sty_image, prompt, self.input_imgs[idx]
        else:
            return in_image, sty_image, prompt

# dataset for evaluation of already-generated samples
class EvalDataset(Dataset):
    def __init__(self, input_dataset_path, style_dataset_path, generated_imgs_path, input_name, style_name, transform=None):
        self.input_dataset_path = input_dataset_path
        self.style_dataset_path = style_dataset_path
        self.generated_imgs_path = generated_imgs_path
        self.style_imgs = np.asarray([i for i in os.listdir(self.style_dataset_path) if i.endswith(('.jpg', '.jpeg', '.png'))])
        self.generated_imgs = np.asarray([i for i in os.listdir(self.generated_imgs_path) if i.endswith(('.jpg', '.jpeg', '.png'))])
        self.input_name = input_name
        self.style_name = style_name
        
        self.transform = transform

        print("Total number of images used for evaluation is: " + str(len(self.generated_imgs)))

    def __len__(self):
        return len(self.generated_imgs)

    def __getitem__(self, idx):
        # Load data (style image and text prompt are randomly selected)
        gen_image_path = os.path.join(self.generated_imgs_path, self.generated_imgs[idx])
        in_image_path = os.path.join(self.input_dataset_path, self.generated_imgs[idx].split("_")[1])
        rdm_idx = np.random.randint(0, len(self.style_imgs))
        sty_image_path = os.path.join(self.style_dataset_path, self.style_imgs[rdm_idx])
        descriptive_prompt = "{} made out of {}".format(self.input_name, self.style_name)

        # Open and preprocess Images
        sty_image = Image.open(sty_image_path).convert('RGB')
        in_image = Image.open(in_image_path).convert('RGB')
        gen_image = Image.open(gen_image_path).convert('RGB')
        sty_image = self.transform(sty_image)
        in_image = self.transform(in_image)
        gen_image = self.transform(gen_image)
        
        return in_image, sty_image, gen_image, descriptive_prompt
