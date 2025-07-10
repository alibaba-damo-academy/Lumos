export PATH=/mnt/workspace/workgroup/hangjie.yhj/conda-envs/lumos-1-public/bin:$PATH 

start_device=0
num_devices=1  # number of gpus. 
sub_tasks=${num_devices}  # number of tasks. 
mask_history_ratio=0.7
for j in $(seq 0 $((sub_tasks-1))); do
  CUDA_VISIBLE_DEVICES=$(((start_device + j) % num_devices)) python inference_i2v.py --part "[$((j+1)), $sub_tasks]" --mask_history_ratio $mask_history_ratio --prompt_source "custom"  &
done

# Wait for all background jobs to complete
wait
