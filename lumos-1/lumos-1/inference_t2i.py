from inference_solver_vid import FlexARVidInferenceSolver
from PIL import Image
import cv2
import os
import numpy as np
import json
import glob
import re
from model.sampling import cosine_schedule
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from multiprocessing import Process, set_start_method
import argparse
import ast

##############################################################################
#                         Grid Image Generation                              #
##############################################################################
def images_into_grid(image_paths, save_grid_path):
    def load_images(image_paths):
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Unable to read {path}")
        return images

    def concatenate_images(images, rows, cols):
        # First, ensure that all image shapes are the same
        h, w = images[0].shape[:2]
        for img in images:
            assert img.shape[:2] == (h, w), "All images must have the same dimensions."
        
        # Create an empty array for the final image grid
        grid_image = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Place each image in the grid
        for i in range(rows):
            for j in range(cols):
                if i * cols + j < len(images):
                    grid_image[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[i * cols + j]
        
        return grid_image

    # Load the images
    images = load_images(image_paths)

    # Define grid dimensions (rows and columns)
    rows = len(image_paths)  # For the given image example
    cols = 1  # For the given image example

    # Make sure the number of images matches the grid definition
    if len(images) != rows * cols:
        print(f"Warning: Number of images ({len(images)}) does not match grid size ({rows}x{cols}).")

    # Concatenate images into grid
    grid_image = concatenate_images(images, rows, cols)

    # Save the resulting image
    cv2.imwrite(save_grid_path, grid_image)
    print(f"Image grid saved as {save_grid_path}")

        

def get_part_of_list(input_list, part=[1, 4]):
    """
    Splits the input_list into 'n_parts' and returns the 'part_index'th part.
    
    :param input_list: List to be split.
    :param part: A list with two elements [part_index, n_parts]. Defaults to [1, 4].
                 part_index is 1-indexed.
    :return: The specified part of the list.
    """
    # Unpack the parameters
    part_index, n_parts = part
    
    # Calculate the approximate size of each part
    part_size = len(input_list) // n_parts
    
    # Determine the start and end indices for the specified part
    start_index = (part_index - 1) * part_size
    end_index = part_index * part_size
    
    # Handle cases where it's not evenly divisible
    # Start padding with extra elements if there are any leftover
    remainder = len(input_list) % n_parts
    if part_index <= remainder:
        start_index += part_index - 1
        end_index += part_index
    else:
        start_index += remainder
        end_index += remainder
    
    # Return the specified part of the list
    return input_list[start_index:end_index]



##############################################################################
#                       Lumos-1 Inference pipeline                           #
##############################################################################
def lumos_1_inference(
        caption, inference_solver, model_type, generation_task, partial_videos, 
        save_path, vp_mode, cfg, output_video_path,
        duration_placeholder = 2, fps_placeholder = 16, timesteps = 18
    ):
    frames_placeholder = duration_placeholder * fps_placeholder
    max_gen_len_dict = {1: 1536, 6: 3700, 12: 6720, 16:8192, 24:14000, 32:16384, 36:18500, 48:24576, 64:32768, 96:49152} # 12: 6144 # 24:12288


    q1 = f"Generate a video with a resolution of <video_resolution_placeholder>, consisting of {frames_placeholder} frames at {fps_placeholder} frames per second, according to the following prompt:\n " \
         + caption
    a1 = "<|partial_video|>"

    if model_type=='next_token_prediction':
        if generation_task == "vp":
            # generated: tuple of (generated response, list of generated images)
            generated = inference_solver.generate(
                partial_videos=partial_videos,
                qas=[[q1, a1]],     
                # qas=[[q1]],
                max_gen_len=max_gen_len_dict[frames_placeholder],
                temperature=1.0,
                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=2000),
                output_video_path=output_video_path,
            )
        else:
            assert False, "Still not implemented."
        print(generated)

    elif model_type=='masked_AR':
        if generation_task == "vp":
            ### vp
            generated = inference_solver.generate_maskedAR_mmrope_v8_newcache(
                partial_videos=partial_videos,
                qas=[[q1, a1]],     
                max_gen_len=max_gen_len_dict[frames_placeholder],
                temperature=1.0,
                timesteps=timesteps,  # ideal number of steps is 18 in maskgit paper
                guidance_scale=cfg,
                noise_schedule=cosine_schedule,
                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=2000),
                output_video_path=output_video_path,
                fps_duration=[fps_placeholder, duration_placeholder],
                generation_task='vp',
            )
        elif generation_task == "t2v":
            ### t2v
            generated = inference_solver.generate_maskedAR_mmrope_v8_newcache(
                partial_videos=[],
                qas=[[q1, ""]],    
                max_gen_len=max_gen_len_dict[frames_placeholder],
                temperature=1.0,
                timesteps=timesteps,  # ideal number of steps is 18 in maskgit paper
                guidance_scale=cfg,
                noise_schedule=cosine_schedule,
                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=2000),
                output_video_path=output_video_path,
                fps_duration=[fps_placeholder, duration_placeholder],
                generation_task='t2v',
                # video_resolution=[256, 448],
                video_resolution=[448, 256],
                # video_resolution=[352, 352],
            )
        else:
            assert False, "The generation task is not supported."
    else:
        assert False, f'{model_type} not implemented'



def lumos_1_inference_img(
        model_path, model_type, generation_task, caption_path, save_path, vp_mode, cfg, caption_part = None, 
        visual_tokenizer="Chameleon", duration_placeholder=2, fps_placeholder=16, timesteps=18
    ):
    # Load meta-info from the JSON file
    meta_data = []
    with open(caption_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip any whitespace (like newlines) and then parse JSON
            json_object = json.loads(line.strip())
            meta_data.append(json_object)
    
    os.makedirs(save_path, exist_ok=True)
    
    if isinstance(model_path, FlexARVidInferenceSolver):
        inference_solver = model_path
    elif isinstance(model_path, str):  # Check if model_path is a string
        st_compression_ratio = obtain_compression_ratio(visual_tokenizer)
        inference_solver = FlexARVidInferenceSolver(
            model_path = model_path,
            precision="bf16",   
            target_fps=fps_placeholder,
            duration=duration_placeholder,
            visual_tokenizer=visual_tokenizer,
            target_size=352,
            vae_st_compress=st_compression_ratio,
        )
    

    run_list = list(range(len(meta_data)))
    if caption_part is not None:
        run_list = get_part_of_list(run_list, part=caption_part)
    
    ### We do not generate those that have already generated.
    new_run_list = []
    for run_idx in run_list:
        meta = meta_data[run_idx]
        idx_path = str(run_idx).zfill(5)
        four_imgs_path = os.path.join(os.path.join(save_path, idx_path), 'samples')
        grid_img_path = os.path.join(os.path.join(save_path, idx_path), 'grid.png')
        if os.path.exists(four_imgs_path) and len(os.listdir(four_imgs_path)) == 4 and os.path.exists(grid_img_path):
            continue
        else:
            new_run_list.append(run_idx)
    bf_dedup_num = len(run_list)
    run_list = new_run_list
    print(f"We need to generate {len(run_list)} items (before deduplcation {bf_dedup_num}).")


    num_samples_per_prompt = 4
    for meta_idx, meta in enumerate(meta_data): # {"tag": "single_object", "include": [{"class": "carrot", "count": 1}], "prompt": "a photo of a carrot"}
        if meta_idx not in run_list:
            continue
        
        caption = meta["prompt_rewrite_qwen32b"]

        idx_path = str(meta_idx).zfill(5)
        os.makedirs(os.path.join(save_path, idx_path), exist_ok=True)
        idx_samples_path = os.path.join(save_path, idx_path, "samples")
        os.makedirs(idx_samples_path, exist_ok=True)

        ### Image generation.
        image_path_list = [os.path.join(idx_samples_path, str(i).zfill(5) + ".png") for i in range(num_samples_per_prompt)]
        for image_path in image_path_list:
            lumos_1_inference(
                caption, inference_solver, model_type, generation_task, [], save_path, vp_mode, cfg, 
                output_video_path=image_path,
                duration_placeholder=duration_placeholder, fps_placeholder=fps_placeholder, timesteps=timesteps
            )

        ### Save as a grid.
        idx_grid_path = os.path.join(save_path, idx_path, "grid.png")
        images_into_grid(image_path_list, idx_grid_path)

        ### Save jsonl metainfo.
        idx_meta_path = os.path.join(save_path, idx_path, "metadata.jsonl")
        meta.pop("prompt_rewrite_qwen32b")
        with open(idx_meta_path, "w") as fp:
            json.dump(meta, fp)


def obtain_compression_ratio(visual_tokenizer):
    if visual_tokenizer == "Chameleon":
        spatial_compression, temporal_compression = 16, 1
    elif "Cosmos-Tokenizer" in visual_tokenizer:
        temporal_compression, spatial_compression = [int(i) for i in visual_tokenizer.split('DV')[1].split('x')[:2]]
    else:
        assert False, f"Visual_tokenizer {visual_tokenizer} is not supported."
    
    return [spatial_compression, temporal_compression]



def parse_argument():
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add a 'part' argument that accepts a string
    parser.add_argument('--part', type=str, required=True, help='[part_idx, num_of_parts]')
    parser.add_argument('--prompt_source', type=str, required=False, default='geneval', help='geneval or custom')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the string representation of the list to an actual list
    part = ast.literal_eval(args.part)
    
    return part, args.prompt_source


def one_part_genetion(part): # [4,4]
    ################################ Eval data collection ################################
    eval_data_collection = {
        "geneval": {
            "caption_path": "eval/prompts/geneval_evaluation_metadata.jsonl",
        },
    }

    ################################# GenEval Generation ##################################
    ### 1B stage 1
    # model = "ckpts/1B/stage-1-image"
    save_path = f"ckpts/generated_results/1B_images" 

    ### 3B stage 1
    model = "ckpts/3B/stage-1-image"
    save_path = f"ckpts/generated_results/3B_images" 


    model_type = "masked_AR"
    generation_task = "t2v" 
    eval_data = "geneval"
    timesteps = 50
    cfg=16

    vp_mode = None 
    duration_placeholder=1 # duration = 1 and fps = 1 mean image generation.
    fps_placeholder=1
    visual_tokenizer="Cosmos-Tokenizer-DV4x8x8"
    lumos_1_inference_img(
        model, model_type, generation_task, 
        eval_data_collection[eval_data]['caption_path'], 
        save_path, vp_mode, cfg, part,
        visual_tokenizer, duration_placeholder, fps_placeholder, timesteps,
    )



def one_part_genetion_custom(part): # [4,4]
    ################################ Eval data collection ################################
    eval_data_collection = {
        "custom_t2i": {
            "caption_path": "eval/prompts/custom_t2i_prompts.jsonl",
        },
    }

    ################################# GenEval Generation ##################################
    ### 1B stage 1
    # model = "ckpts/1B/stage-1-image"
    # save_path = f"ckpts/generated_results/1B_custom_images" 
    ### 3B stage 1
    model = "ckpts/3B/stage-1-image"
    save_path = f"ckpts/generated_results/3B_custom_images" 

    model_type = "masked_AR"
    generation_task = "t2v" 
    eval_data = "custom_t2i"
    timesteps = 50
    cfg=16


    vp_mode = None 
    duration_placeholder=1 # duration = 1 and fps = 1 mean image generation.
    fps_placeholder=1
    visual_tokenizer="Cosmos-Tokenizer-DV4x8x8"
    lumos_1_inference_img(
        model, model_type, generation_task, 
        eval_data_collection[eval_data]['caption_path'], 
        save_path, vp_mode, cfg, part,
        visual_tokenizer, duration_placeholder, fps_placeholder, timesteps,
    )



if __name__=="__main__":
    part, prompt_source = parse_argument()
    if prompt_source == "geneval":
        one_part_genetion(part)
    elif prompt_source == "custom":
        one_part_genetion_custom(part)
