import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from argparse import ArgumentParser
import json
import pandas as pd
import math
import pickle

from data.convertsation import Conversation
# from data.item_processor import FlexARItemProcessor
from data.item_processor import FlexARItemProcessor2
import time
import torch



class ItemProcessor(FlexARItemProcessor2):
    def __init__(
        self,
        target_fps,
        target_duration,
        video_frame_batch_size,
        tokenizer="Alpha-VLLM/Lumina-mGPT-7B-768",
        visual_tokenizer="Chameleon",
        cosmos_dtype=torch.bfloat16,
        conv_template=Conversation,
        target_size=512,
    ):
        super().__init__(
            tokenizer, conv_template, target_size, 
            target_fps=target_fps, duration=target_duration, 
            cosmos_dtype=cosmos_dtype, visual_tokenizer=visual_tokenizer
        )
        print(self.crop_size_list)


    def process_item(self, raw_item, training_mode=False, out_flatten=True):

        # Add custom codes here to convert raw_item to the standard format
        # The standard format contains the "conversations" and "image" keys

        # ********* <start>  Add your custom codes here *******

        # *********  <end>   Add your custom codes here *******

        item = {
            "conversations": raw_item["conversations"],
        }

        if "image" in raw_item:
            item.update({"image": raw_item["image"]})
        if "video" in raw_item:
            item.update({"video": raw_item["video"]})

        return super(ItemProcessor, self).process_item(item, training_mode, out_flatten)
            
    def transform_csv_to_json_dict(self, csv_path):
        '''
        This function reads a CSV file, processes its content, and transforms it into a JSON-like structure.
        Each row of the CSV is converted into a specific format involving conversations and video data.

        Parameters:
        - csv_path (str): The file path to the CSV file that needs to be transformed.

        Returns:
        - output_data (list): A list of dictionaries, where each dictionary represents a JSON-like object derived from a row of the CSV file.
        '''

        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Prepare list to store the output dictionaries
        output_data = []

        # Iterate over each row in the DataFrame to create JSON objects
        for idx, row in df.iterrows():
            data_dict = {
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Generate a video with a resolution of <video_resolution_placeholder>, consisting of "
                            f"<video_frames_placeholder> frames at <video_fps_placeholder> frames per second, according to the following prompt:\n "
                            f"{row['text']}"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": "<|video|>"
                    }
                ],
                "video": [row['path']]
            }
            output_data.append(data_dict)

            # # Write the output to a JSON file
            # output_file = "output.json"
            # with open(output_file, "w", encoding="utf-8") as f:
            #     json.dump(output_data, f, indent=4)
            # print(f"Data has been successfully written to {output_file}")

        return output_data


def format_elapsed_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    return f"{hours}h {minutes}m {seconds}s"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--video_frame_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--in_filename",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
    )
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--target_duration", type=int, default=8)
    parser.add_argument("--visual_tokenizer", type=str, default="Chameleon", choices=["Chameleon", "Cosmos-Tokenizer-DV4x8x8"])
    parser.add_argument("--cosmos_dtype", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")

    parser.add_argument("--all_tasks", type=int, default=None)
    parser.add_argument("--task_id", type=int, default=None) # 1, ..., all_tasks
    args = parser.parse_args()

    print('Before ItemProcessor init.')
    cosmos_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.cosmos_dtype]
    item_processor = ItemProcessor(
        target_size=args.target_size, 
        target_fps=args.target_fps,
        target_duration=args.target_duration,
        video_frame_batch_size=args.video_frame_batch_size,
        visual_tokenizer=args.visual_tokenizer,
        cosmos_dtype=cosmos_dtype,
    )

    process_subtask = False
    if args.all_tasks is not None and args.task_id is not None:
        process_subtask = True


    # with open(args.in_filename) as f:
    #     ori_contents = json.load(f)
    
    ori_contents = item_processor.transform_csv_to_json_dict(args.in_filename)
    global_id_add = 0
    if process_subtask:
        task_len = len(ori_contents)//args.all_tasks
        ori_contents = ori_contents[(args.task_id-1)*task_len : args.task_id*task_len] if args.task_id < args.all_tasks else ori_contents[(args.task_id-1)*task_len : ]
        global_id_add = (args.task_id-1)*task_len
    
    num = len(ori_contents)

    splits = args.splits
    rank = args.rank
    output_dir = args.out_dir
    if process_subtask:
        output_dir = os.path.join(output_dir, f"{args.task_id}-in-{args.all_tasks}")

    save_dir = os.path.join(output_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    num_per_rank = math.ceil(num / splits)

    try:
        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "r") as f:
            start_idx = int(f.read()) + 1
        print(f"resume from {start_idx}")
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}")

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))
    start_time = time.time()
    for i in range(start_idx, end_idx):
        if i % 10 == 0:
            print(f"{start_idx} to {end_idx}: {i}/{end_idx}, consuming {format_elapsed_time(time.time()-start_time)}")

        record = None
        pkl_path = os.path.join(save_dir, f"{i+global_id_add}.pkl")

        tokens, labels = [], []
        try:
            tokens, labels = item_processor.process_item(ori_contents[i], training_mode=True)
            '''ori_contents[i]
            {   "conversations":[
                    {
                        "from": "human",
                        "value": "Generate an image of 768x768 according to the following prompt:\n This image is a promotional poster for Taylor Swift's 'The Eras Tour'. It features a stylized illustration of a blonde woman in a sparkling blue and purple sequined bodysuit with a high-cut leg. The figure is posed dramatically against a swirling pink and purple background. Her hair is long and straight, and she has bright red lips in an open-mouthed expression. The artwork is signed 'Anna W.' in the top right corner. At the bottom of the image, 'TAYLOR SWIFT THE ERAS TOUR' is written in large, bold black text. The overall aesthetic is glamorous and eye-catching, typical of concert tour promotional material."
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>"
                    }
                ],
                "image": ["data/test-image-dataset/00.jpg"]}
            '''
            # new_item = {"token": tokens, "label": labels, "id": i}
            # with open(pkl_path, "wb") as f:
            #     pickle.dump(new_item, f)

            # record = {"file": pkl_path, "len": len(tokens), "id": i}

        except Exception as e:
            from traceback import format_exc

            print(f"item {i} error: {ori_contents[i]}\n")
            print(format_exc())
            if "KeyError: 'size_wh'" in format_exc():
                print("This video is too short.\n")

        if len(tokens)==0 and len(labels)==0:
            print(f"We skip {ori_contents[i]}.")
        else:
            new_item = {"token": tokens, "label": labels, "id": i + global_id_add}
            with open(pkl_path, "wb") as f:
                pickle.dump(new_item, f)

            record = {"file": pkl_path, "len": len(tokens), "id": i + global_id_add}

            if record is not None:
                with open(os.path.join(output_dir, f"{rank}-of-{splits}-record.jsonl"), "a") as f:
                    record_str = json.dumps(record) + "\n"
                    f.write(record_str)

            with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "w") as f:
                if i == end_idx - 1:
                    f.write("finished")
                else:
                    f.write(f"{i}")