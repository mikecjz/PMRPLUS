#!/bin/bash

# Example script to run inference with sensitivity maps logging to wandb
# 
# Usage: ./run_inference_with_sensmaps.sh <checkpoint_path> <config_path>

set -e

# Default values
CHECKPOINT_PATH=${1:-"path/to/your/checkpoint.ckpt"}
CONFIG_PATH=${2:-"configs/inference/example_with_sensmaps.yaml"}

echo "Running inference with sensitivity maps logging..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"

# Run inference with wandb logging
python main.py predict \
    --config $CONFIG_PATH \
    --predict.ckpt_path $CHECKPOINT_PATH \
    --trainer.logger.init_args.project "promptmr-inference-sensmaps" \
    --trainer.logger.init_args.name "inference_$(date +%Y%m%d_%H%M%S)" \
    --trainer.logger.save_dir "./wandb_logs"

echo "Inference completed! Check your wandb dashboard for sensitivity maps."
