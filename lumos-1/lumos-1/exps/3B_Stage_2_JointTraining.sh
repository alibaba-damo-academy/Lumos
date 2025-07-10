#!/bin/bash
lr=2.5e-5
wd=0.1
dropout=0
z_loss_weight=0

########################################### Data configuration #########################################
data_config=configs/data/pre-tokenized-vid-onejson-Cosmos-DV4x8x8.yaml
eval_data_config=configs/data/pre-tokenized-vid-onejson-Cosmos-DV4x8x8.yaml

image_data_config=configs/data/pre-tokenized-img-onejson-Cosmos-DV4x8x8.yaml


######################################### System Configuration #########################################
### V1: From H20
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=3600
export NCCL_LAUNCH_MODE=PARALLEL
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IGNORE_DISABLED_P2P=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### V2: From H100
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=storage_bond
# export NCCL_CROSS_NIC=1
# export NCCL_IB_TIMEOUT=3600
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_NET_PLUGIN=none
# export TOKENIZERS_PARALLELISM=false
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_ENDPOINT=https://hf-mirror.com


######################################### Code Configuration #########################################
exp_name=h100-3B-stage-2-joint-training-384p

mkdir -p output/"$exp_name" 

export PATH=/mnt/workspace/workgroup/hangjie.yhj/conda-envs/lumos-1-public/bin:$PATH

torchrun --nnodes 1  --nproc-per-node 1 --rdzv-conf=timeout=36000 finetune_solver_vid.py \
--init_from  ckpts/3B/stage-2-joint-384p \
--model_size 3B_MMRoPE \
--batch_size 1 \
--accum_iter 1 \
--epochs 100 \
--warmup_epochs 0.0025 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--clip_grad 4 \
--data_config $data_config \
--num_workers 32 \
--output_dir output/"$exp_name" \
--save_iteration_interval 1000 \
--checkpointing \
--max_seq_len 65536 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
--pretrain_task predict_video \
--video_fps 12 \
--video_duration 2 \
--ckpt_max_keep 20 \
--cfg_mode text_only \
--masking_mode  text_to_video  \
--eval_data_config $eval_data_config \
--data_video_fps 12 \
--data_video_frames 84 \
--MaskedAR \
--mask_schedule cosine \
--min_masking_rate 0.7 \
--noise_type mask \
--mask_type tube \
--train_loss CEChunked \
--decay_start_coef 1 \
--visual_tokenizer Cosmos-Tokenizer-DV4x8x8 \
--vocab_size 129536 \
--no_ntp_loss \
--train_with_vis_tok \
--vis_tok_start 65536 \
--frame_closs_recorder \
--no_resume_metric_logger \
--joint_img_video \
--img_batch_size 6 \
--img_data_config $image_data_config \
--video_iter_per_img_iter 1 \
# --eval_in_epoch 200 \
# --eval_mode  text_to_video \
# --run_eval \
