from inference_solver_vid import FlexARVidInferenceSolver
from PIL import Image
import cv2
import os
import numpy as np
import json
import glob
import re
from model.sampling import cosine_schedule
import argparse
import ast


##############################################################################
#                      Video prediction pipeline                             #
##############################################################################
def test_one_video_prediction_vbench(
        caption, inference_solver, model_type, generation_task, partial_videos, 
        save_path, vp_mode, cfg, target_size,
        mask_history_ratio=None, caption_as_name=False, caption_file_name=None,
        duration_placeholder = 2, fps_placeholder = 16, timesteps = 18
    ):
    # resolution_placeholder = "448x256"
    frames_placeholder = duration_placeholder * fps_placeholder
    max_gen_len_dict = {1: 1536, 6: 3700, 12: 6720, 16:8192, 24:14000, 32:16384, 36:18500, 48:24576, 64:32768, 96:49152} # 12: 6144 # 24:12288

    ### resolution
    resolution_dict = {656: [768, 480], 352: [448, 256], 528: [672, 384]}
    video_resolution = resolution_dict[target_size] 


    q1 = f"Generate a video with a resolution of <video_resolution_placeholder>, consisting of {frames_placeholder} frames at {fps_placeholder} frames per second, according to the following prompt:\n " \
         + caption
    a1 = "<|partial_video|>"

    if caption_as_name:
        if caption_file_name is not None:
            output_video_path = f"{save_path}/{caption_file_name}"
        else:
            output_video_path = caption.replace(" ", "_") + ".mp4"
            output_video_path = f"{save_path}/{output_video_path}"
    else:
        file_name_prefix = '_'.join(partial_videos[0].split('/')[-2:])
        vp_mode = "t2v" if generation_task == "t2v" else vp_mode
        output_video_path = f"{save_path}/{file_name_prefix}_{duration_placeholder}_{fps_placeholder}_{vp_mode}_cfg{cfg}_topk2000.mp4"
    print(output_video_path)
    
    if model_type=='next_token_prediction':
        if generation_task == "vp":
            # generated: tuple of (generated response, list of generated images)
            generated = inference_solver.generate(
                partial_videos=partial_videos,
                qas=[[q1, a1]],     
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
                video_resolution=video_resolution,
                mask_history_ratio=mask_history_ratio,
            )
        elif generation_task == "t2v":
            ### t2v
            generated = inference_solver.generate_maskedAR_mmrope_v8_newcache(
                partial_videos=[],
                qas=[[q1, ""]],  # [[q1, ""]],     
                max_gen_len=max_gen_len_dict[frames_placeholder],
                temperature=1.0,
                # temperature=6.0,
                timesteps=timesteps,  # ideal number of steps is 18 in maskgit paper
                guidance_scale=cfg,
                noise_schedule=cosine_schedule,
                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=2000),
                output_video_path=output_video_path,
                fps_duration=[fps_placeholder, duration_placeholder],
                generation_task='t2v',
                video_resolution=video_resolution,
                mask_history_ratio=mask_history_ratio,
            )
        else:
            assert False, "The generation task is not supported."
    else:
        assert False, f'{model_type} not implemented'



def lumos_1_t2v(
        model_path, model_type, generation_task, video_path, caption_path, save_path, vp_mode, cfg, target_size,
        mask_history_ratio, caption_part=None, cluster=[1,1], num_vids_per_caption=1,
        visual_tokenizer="Chameleon", duration_placeholder=2, fps_placeholder=16, timesteps=18
    ):
    # Load captions from the JSON file
    with open(caption_path, 'r') as f:
        captions = json.load(f)
    # use_caption_file_name = True if "all_dimension_longer.json" in caption_path else False
    use_caption_file_name = True # Set to true by default for VBench

    ### Split the captions to be a part corresponding to a cluster.
    captions = get_part_of_dictionary(captions, part=cluster)
    print(f"We run on one cluster ({len(captions)} captions), which is part of many clusters ({cluster}).")
    
    os.makedirs(save_path, exist_ok=True)
    
    if isinstance(model_path, FlexARVidInferenceSolver):
        inference_solver = model_path
    elif isinstance(model_path, str):  # Check if model_path is a string
        assert target_size in [352, 656, 528]
        st_compression_ratio = obtain_compression_ratio(visual_tokenizer)
        inference_solver = FlexARVidInferenceSolver(
            model_path = model_path,
            precision="bf16",
            target_fps=fps_placeholder,
            duration=duration_placeholder,
            visual_tokenizer=visual_tokenizer,
            target_size=target_size,
            vae_st_compress=st_compression_ratio,
        )
    
    video_files = glob.glob(os.path.join(video_path, "*.mp4"))

    ### List a running dict.
    # run_dict = list(captions.keys())
    if caption_part is not None:
        run_dict = get_part_of_dictionary(captions, part=caption_part)
    if num_vids_per_caption > 1:
        repeat_run_dict = {}
        for name, cap in run_dict.items():
            for vid_idx in range(num_vids_per_caption):
                suffix = f"-{vid_idx}.mp4"
                repeat_run_dict[name.replace(".mp4", suffix)] = cap
        run_dict = repeat_run_dict
    # Delete completed videos (save time).
    completed_videos = os.listdir(save_path)
    bf_dedup_num = len(run_dict)
    run_dict = {f_name: cap for f_name, cap in run_dict.items() if f_name not in completed_videos}
    print(f"We need to generate {len(run_dict)} videos (before deduplcation {bf_dedup_num}).")

    if len(video_files) == 0 or generation_task == 't2v':
        generation_task = 't2v'
        print(f"Did not find initial frames, setting generation_task to text-to-video generation.")
        for cap_idx, (caption_file_name, caption) in enumerate(run_dict.items()):
            print(f"Start running {cap_idx}/{len(run_dict)}.")
            test_one_video_prediction_vbench(
                caption, inference_solver, model_type, generation_task, [], save_path, vp_mode, cfg, 
                target_size=target_size, mask_history_ratio=mask_history_ratio,
                caption_as_name=True, caption_file_name=caption_file_name if use_caption_file_name else None,
                duration_placeholder=duration_placeholder, fps_placeholder=fps_placeholder, timesteps=timesteps
            )
    
    else:
        assert False


def obtain_compression_ratio(visual_tokenizer):
    if visual_tokenizer == "Chameleon":
        spatial_compression, temporal_compression = 16, 1
    elif "Cosmos-Tokenizer" in visual_tokenizer:
        temporal_compression, spatial_compression = [int(i) for i in visual_tokenizer.split('DV')[1].split('x')[:2]]
    else:
        assert False, f"Visual_tokenizer {visual_tokenizer} is not supported."
    
    return [spatial_compression, temporal_compression]


def get_part_of_dictionary(input_dict, part=[1, 4]):
    """
    Splits the input_dict into 'n_parts' and returns the 'part_index'th part.

    :param input_dict: Dictionary to be split.
    :param part: A list with two elements [part_index, n_parts]. Defaults to [1, 4].
                 part_index is 1-indexed.
    :return: The specified part of the dictionary.
    """
    # Unpack the parameters
    part_index, n_parts = part
    
    # Convert dictionary items to a list of tuples
    items = list(input_dict.items())
    
    # Calculate the approximate size of each part
    part_size = len(items) // n_parts
    
    # Determine the start and end indices for the specified part
    start_index = (part_index - 1) * part_size
    end_index = part_index * part_size
    
    # Handle cases where it's not evenly divisible
    # Start padding with extra elements if there are any leftover
    remainder = len(items) % n_parts
    if part_index <= remainder:
        start_index += part_index - 1
        end_index += part_index
    else:
        start_index += remainder
        end_index += remainder
    
    # Extract the specified part of the items list
    part_items = items[start_index:end_index]
    
    # Return the specified part as a dictionary
    return dict(part_items)



def one_part_genetion(part, cluster, mask_history_ratio):
    ################################ Eval data collection ################################
    eval_data_collection = {
        "vbench-all-dimension-longer": {
            "video_path": "",
            "caption_path": "eval/prompts/vbench_t2v_all_dimension_longer.json",
        },
    }

    # NOTE Uncomment the model you need.
    ### 1B 256p
    # target_size = 352
    # model = "ckpts/1B/stage-2-joint"
    # save_path = "ckpts/generated_results/videos/1B_256p_t2v"

    ### 1B 384p
    # target_size = 528
    # model = "ckpts/1B/stage-2-joint-384p"
    # save_path = "ckpts/generated_results/videos/1B_384p_t2v"
    
    ### 3B 256p
    # target_size = 352
    # model = "ckpts/3B/stage-2-joint"
    # save_path = "ckpts/generated_results/videos/3B_256p_t2v"

    ### 3B 384p
    target_size = 528
    model = "ckpts/3B/stage-2-joint-384p"
    save_path = "ckpts/generated_results/videos/3B_384p_t2v"


    model_type = "masked_AR"
    generation_task = "t2v" 
    eval_data = "vbench-all-dimension-longer" 
    timesteps = 50 
    cfg = 15

    vp_mode = None
    duration_placeholder = 2 
    fps_placeholder = 12
    visual_tokenizer = "Cosmos-Tokenizer-DV4x8x8"
    lumos_1_t2v(
        model, model_type, generation_task, 
        eval_data_collection[eval_data]['video_path'], 
        eval_data_collection[eval_data]['caption_path'], 
        save_path, vp_mode, cfg, target_size, mask_history_ratio, part, cluster, 5,
        visual_tokenizer, duration_placeholder, fps_placeholder, timesteps,
    )


def one_part_genetion_custom(part, mask_history_ratio):
    ################################ Eval data collection ################################
    eval_data_collection = {
        "custom_t2v": {
            "video_path": "",
            "caption_path": "eval/prompts/custom_t2v_prompts.json",
        },
    }

    # NOTE Uncomment the model you need.
    ### 1B 256p
    # target_size = 352
    # model = "ckpts/1B/stage-2-joint"
    # save_path = "ckpts/generated_results/videos/1B_256p_t2v_custom"

    ### 1B 384p
    # target_size = 528
    # model = "ckpts/1B/stage-2-joint-384p"
    # save_path = "ckpts/generated_results/videos/1B_384p_t2v_custom"
    
    ### 3B 256p
    # target_size = 352
    # model = "ckpts/3B/stage-2-joint"
    # save_path = "ckpts/generated_results/videos/3B_256p_t2v_custom"

    ### 3B 384p
    target_size = 528
    model = "ckpts/3B/stage-2-joint-384p"
    save_path = "ckpts/generated_results/videos/3B_384p_t2v_custom"


    model_type = "masked_AR"
    generation_task = "t2v"
    eval_data = "custom_t2v"
    timesteps = 50
    cfg = 15

    vp_mode = None
    duration_placeholder = 2  # 2
    fps_placeholder = 12
    visual_tokenizer = "Cosmos-Tokenizer-DV4x8x8"
    lumos_1_t2v(
        model, model_type, generation_task, 
        eval_data_collection[eval_data]['video_path'], 
        eval_data_collection[eval_data]['caption_path'], 
        save_path, vp_mode, cfg, target_size, mask_history_ratio, part, [1,1], 5,
        visual_tokenizer, duration_placeholder, fps_placeholder, timesteps,
    )



def parse_argument():
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add a 'part' argument that accepts a string
    parser.add_argument('--part', type=str, required=True, help='[part_idx, num_of_parts]')
    parser.add_argument('--mask_history_ratio', type=float, required=True, help='float between [0,1]')
    parser.add_argument('--prompt_source', type=str, required=False, default='vbench', help='vbench or custom')
    parser.add_argument('--cluster', type=str, default='[1,1]', help='[cluster_idx, num_of_cluster]')


    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the string representation of the list to an actual list
    part = ast.literal_eval(args.part)
    cluster = ast.literal_eval(args.cluster)
    
    return part, cluster, args.mask_history_ratio, args.prompt_source


if __name__=="__main__":
    part, cluster, mask_history_ratio, prompt_source = parse_argument()
    if prompt_source == "vbench":
        one_part_genetion(part, cluster, mask_history_ratio)
    elif prompt_source == "custom":
        one_part_genetion_custom(part, mask_history_ratio)