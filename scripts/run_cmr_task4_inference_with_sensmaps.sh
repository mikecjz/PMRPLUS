#!/bin/bash

# Script to run inference with sensitivity maps logging for CMR Task4 validation
# Based on your existing cmr-task4-val_cs.yaml configuration

set -e

echo "Running CMR Task4 inference with sensitivity maps logging..."
echo "Using configuration: configs/inference/pmr-plus/cmr-task4-val_cs_with_sensmaps.yaml"
echo "Using GPU 7"

# Set CUDA device to GPU 7
export CUDA_VISIBLE_DEVICES=7

# Run inference with wandb logging
python main.py predict \
    --config configs/inference/pmr-plus/cmr-task4-val_cs_with_sensmaps.yaml \
    --trainer.logger.init_args.project "promptmr-plus-cmr-task4-inference-sensmaps" \
    --trainer.logger.init_args.name "cmr-task4-val_cs_$(date +%Y%m%d_%H%M%S)" \
    --trainer.logger.save_dir "/common/lidxxlab/chushu/PromptMR-plus-Task4_debug/wandb_logs"

echo "Inference completed! Check your wandb dashboard for sensitivity maps."
echo "Wandb project: promptmr-plus-cmr-task4-inference-sensmaps"
echo "Output directory: /common/lidxxlab/cmrchallenge/code/Inference/inf_augmented_cs1020_intensity_adjusted"
