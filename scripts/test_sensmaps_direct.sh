#!/bin/bash

# Test script to verify sensitivity maps logging during inference
# This approach directly logs from predict_step method (no callback needed)

set -e

echo "Testing sensitivity maps logging directly from predict_step method..."
echo "This approach is much simpler - no callback needed!"

# Run inference with limited batches to test logging
cd /common/lidxxlab/chushu/PromptMR-plus-Task4_debug

CUDA_VISIBLE_DEVICES=7 python main.py predict \
    --config configs/inference/pmr-plus/cmr-task4-val_cs_with_sensmaps.yaml \
    --trainer.limit_predict_batches 25 \
    --trainer.logger.init_args.name "test_sensmaps_direct_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "✅ Test completed!"
echo "Check your wandb dashboard for sensitivity maps logged directly from predict_step method."
echo "Look for images under keys like: inference/batch_000, inference/batch_020, etc."
echo ""
echo "This approach:"
echo "- ✅ No callback needed"
echo "- ✅ Direct logging from predict_step (like validation_step)"
echo "- ✅ Same logging pattern as validation stage"
echo "- ✅ Logs every 20 batches automatically"
echo "- ✅ Includes sensitivity maps, zero-filled, and reconstructed images"
