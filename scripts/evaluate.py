import os
from concurrent import futures
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
import torchvision.transforms as T
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
import json
import EditSpecialists.reward_modeling.score_functions
import EditSpecialists.evaluation.eval_funcs
from torchvision import transforms as T
from EditSpecialists.data.dataset import EvalDataset

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/eval.py", "Evaluation configuration.")

def main(_):
    config = FLAGS.config

    generated_imgs_path = os.path.join(config.sample_savedir, f"{config.input_name}_{config.style_name}")
    print(f'Evaluating on images found in: {generated_imgs_path}')

    # Load eval functions
    executor = futures.ThreadPoolExecutor(max_workers=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    structural_fn = getattr(EditSpecialists.reward_modeling.score_functions, "structural_score")(device=device)
    semantic_fn = getattr(EditSpecialists.reward_modeling.score_functions, "semantic_score")(device=device)
    eval_fn = getattr(EditSpecialists.evaluation.eval_funcs, "eval_main")(device=device)
    hps_fn = getattr(EditSpecialists.evaluation.eval_funcs, "hpsv2")(device=device)
    imagereward_fn = getattr(EditSpecialists.evaluation.eval_funcs, "imagereward")(device=device)
    pickscore_fn = getattr(EditSpecialists.evaluation.eval_funcs, "pickscore")(device=device)

    # Prepare dataloader
    eval_transform = T.Compose([T.Resize(config.data_resize),
                                    T.CenterCrop((config.data_resize, config.data_resize)),
                                    T.ToTensor()])
    dataset = EvalDataset(input_dataset_path=config.inputimg_dir, style_dataset_path=config.styleimg_dir, generated_imgs_path=generated_imgs_path, input_name=config.input_name, style_name=config.style_name, transform=eval_transform)
    eval_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # Iterate over batches of data to evaluate
    results_dict = {
                "semantic_score": [],
                "structural_score": [],
                "l2_out": [],
                "clip_txt": [],
                "clip_dir": [],
                "clipSim_gen_sty": [],
                "clipSim_gen_input": [],
                "dinoSim_gen_sty": [],
                "dinoSim_gen_input": [],
                "dreamsimSim_gen_sty": [],
                "dreamsimSim_gen_input": [],
                "hps_score": [],
                "imgrew_score": [],
                "pick_score": [],
            }
    for itt, (input_images, sty_images, gen_images, descriptive_prompts) in tqdm(enumerate(eval_dataloader)):
        gen_images, sty_images, input_images = gen_images.to(device), sty_images.to(device), input_images.to(device)
        descriptive_prompts = list(descriptive_prompts)
        
        # Semantic Score
        L_sem, masks_pixel_gen, masks_pixel_sty = executor.submit(semantic_fn, gen_images, sty_images, input_images, descriptive_prompts, for_eval=True, whole_frame=config.whole_frame).result()
        L_sem = torch.nanmean(L_sem).item()
        
        # Structural Score
        L_struct_gen = executor.submit(structural_fn, gen_images).result()
        L_struct_input = executor.submit(structural_fn, input_images).result()
        L_struct = torch.nn.L1Loss(reduction='none')(L_struct_input, L_struct_gen).cpu().detach()
        L_struct = torch.nanmean(L_struct).item()

        # CLIP, DINO, DREAMSIM similarites and L2
        eval_results = executor.submit(eval_fn, input_images, gen_images, sty_images, masks_pixel_sty, masks_pixel_gen, descriptive_prompts).result()
        
        # Aesthetics
        hps_score = torch.nanmean(executor.submit(hps_fn, gen_images, descriptive_prompts).result().cpu().detach()).item()
        imgrew_score = torch.nanmean(executor.submit(imagereward_fn, gen_images, descriptive_prompts).result().cpu().detach()).item()
        pick_score = torch.nanmean(executor.submit(pickscore_fn, gen_images, descriptive_prompts).result().cpu().detach()).item()
        
        # Save results
        for key in eval_results.keys():
            results_dict[key].append(eval_results[key])
        results_dict["semantic_score"].append(L_sem)
        results_dict["structural_score"].append(L_struct)
        results_dict["hps_score"].append(hps_score)
        results_dict["imgrew_score"].append(imgrew_score)
        results_dict["pick_score"].append(pick_score)

    # Save json summary of results
    for key in results_dict.keys():
        results_dict[key] = np.mean(results_dict[key])
    eval_savepath = os.path.join(config.eval_savedir, f"{config.input_name}_{config.style_name}.json")
    os.makedirs(config.eval_savedir, exist_ok=True)
    # Write the dictionary to a JSON file
    with open(eval_savepath, 'w') as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    app.run(main)
