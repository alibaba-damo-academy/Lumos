export PATH=/mnt/workspace/workgroup/hangjie.yhj/conda-envs/lumos-1-public/bin:$PATH 

start_device=0
i=4             # number of gpus you want to use for generation.
for j in $(seq 0 $((i-1))); do
  CUDA_VISIBLE_DEVICES=$((start_device + j)) python inference_t2i.py --part "[$((j+1)), $i]" --prompt_source "custom" &
  ### For debugging
  # python -m ipdb inference_t2i.py --part "[$((j+1)), $i]"
done
# Wait for all background jobs to complete
wait