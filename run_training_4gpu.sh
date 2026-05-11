#!/bin/bash
# Training MiniVLA on xArm pick red block data with 4 GPUs

# Export environment variables
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

# Run training on 4 GPUs
uv run torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block" \
  --vla.expected_world_size 4 \
  --vla.global_batch_size 32 \
  --vla.per_device_batch_size 8 \
  --pretrained_checkpoint "checkpoints/minivla-libero90-prismatic/checkpoints/latest-checkpoint.pt" \
  --data_root_dir "data/rlds" \
  --run_root_dir "runs/" \
  --is_resume False \
  --image_aug True \
  --save_interval 1000 \
  --wandb_project "xarm-minivla-4gpu"
