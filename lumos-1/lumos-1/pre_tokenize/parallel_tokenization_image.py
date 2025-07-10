import os
from multiprocessing import Process

def run_script(rank, all_ranks):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    print(f"Starting running on {rank}.")

    # Set the environment variable
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # Log in to Hugging Face CLI (note this assumes `huggingface-cli` is on the PATH)

    # Image 256p
    # os.system(f"/mnt/workspace/workgroup/hangjie.yhj/conda-envs/lumina-mgpt/bin/python -u pre_tokenize/pre_tokenize_vid_csv.py "
    #           f"--splits={all_ranks} "
    #           f"--rank={rank} "
    #           f"--in_filename  pre_tokenize/csv_files/test_image.csv "
    #           f"--out_dir  pre_tokenize/data/test_image "              
    #           f"--target_size 352 "
    #           f"--target_fps 1 "
    #           f"--target_duration 1 "
    #           f"--all_tasks 1 "
    #           f"--task_id 1 "
    #           f"--visual_tokenizer Cosmos-Tokenizer-DV4x8x8 "
    #           f"--cosmos_dtype bf16 ")

    # Image 384p
    os.system(f"/mnt/workspace/workgroup/hangjie.yhj/conda-envs/lumina-mgpt/bin/python -u pre_tokenize/pre_tokenize_vid_csv.py "
              f"--splits={all_ranks} "
              f"--rank={rank} "
              f"--in_filename  pre_tokenize/csv_files/test_image.csv "
              f"--out_dir  pre_tokenize/data/test_image "              
              f"--target_size 528 "
              f"--target_fps 1 "
              f"--target_duration 1 "
              f"--all_tasks 1 "
              f"--task_id 1 "
              f"--visual_tokenizer Cosmos-Tokenizer-DV4x8x8 "
              f"--cosmos_dtype bf16 ")



if __name__ == "__main__":
    processes = []
    all_ranks = 8
    for i in range(all_ranks):
        p = Process(target=run_script, args=(i, all_ranks))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
