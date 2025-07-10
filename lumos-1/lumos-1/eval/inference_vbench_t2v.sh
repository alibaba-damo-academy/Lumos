export PATH=/mnt/workspace/workgroup/hangjie.yhj/conda-envs/lumos-1-public/bin:$PATH 
# CUDA_VISIBLE_DEVICES=0 python inference_vbench.py --part "[1, 1]" --mask_history_ratio 0.3

start_device=0
i=4 # number of gpus
mask_history_ratio=0.7
for j in $(seq 0 $((i-1))); do
  CUDA_VISIBLE_DEVICES=$((start_device + j)) python inference_t2v.py --part "[$((j+1)), $i]" --mask_history_ratio $mask_history_ratio  &
done
# Wait for all background jobs to complete
wait



