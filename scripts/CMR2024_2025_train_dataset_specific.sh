#!/bin/bash

# =============================================================================
# CMR 2024-2025 Multi-Dataset Training Script with Dataset-Specific Balancing
# =============================================================================

# SBATCH Configuration
#SBATCH --job-name=cmr24_25_ds
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:H100:4
#SBATCH --mail-user=Yi.Zheng@cshs.org
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Environment Setup
# =============================================================================

echo "=== Setting up environment ==="
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Current host: $(hostname)"

# Activate conda environment
echo "Activating conda environment..."
source activate cmr
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'cmr'"
    exit 1
fi

# Set environment variables
export CMRROOT=/common/lidxxlab/Yi/PromptMR-plus-Task3_large2
export SAVE_DIR=/common/lidxxlab/Yi/training_results_folder/multi_dataset_training_dataset_specific

echo "CMRROOT: $CMRROOT"
echo "SAVE_DIR: $SAVE_DIR"

# Change to project directory
cd $CMRROOT
echo "Changed to directory: $(pwd)"

# =============================================================================
# GPU Information
# =============================================================================

echo "=== GPU Information ==="
nvidia-smi
echo ""

# =============================================================================
# Weights & Biases Setup
# =============================================================================

echo "=== Setting up Weights & Biases ==="
# Replace YOUR_WANDB_API_KEY_HERE with your actual API key
wandb login fac722f823bd0fbc4c5b3b255e9c3e8266aaf362
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to login to wandb, but continuing..."
fi

# =============================================================================
# Dataset Path Verification
# =============================================================================

echo "=== Verifying dataset paths ==="

# Check 2024 dataset
CMR2024_PATH="/common/lidxxlab/cmrchallenge/data/CMR2024/Processed"
if [ -d "$CMR2024_PATH" ]; then
    echo "✓ CMR2024 dataset found at: $CMR2024_PATH"
    echo "  Number of files: $(find $CMR2024_PATH -name "*.h5" | wc -l)"
else
    echo "✗ ERROR: CMR2024 dataset not found at: $CMR2024_PATH"
    exit 1
fi

# Check 2025 dataset
CMR2025_PATH="/common/lidxxlab/cmrchallenge/data/CMR2025/Processed"
if [ -d "$CMR2025_PATH" ]; then
    echo "✓ CMR2025 dataset found at: $CMR2025_PATH"
    echo "  Number of files: $(find $CMR2025_PATH -name "*.h5" | wc -l)"
else
    echo "✗ ERROR: CMR2025 dataset not found at: $CMR2025_PATH"
    exit 1
fi

# =============================================================================
# Create Save Directory
# =============================================================================

echo "=== Creating save directory ==="
mkdir -p $SAVE_DIR
if [ $? -eq 0 ]; then
    echo "✓ Save directory created/verified: $SAVE_DIR"
else
    echo "✗ ERROR: Failed to create save directory: $SAVE_DIR"
    exit 1
fi

# =============================================================================
# Training Configuration
# =============================================================================

echo "=== Training Configuration ==="
CONFIG_FILE="configs/train/pmr-plus/cmr24-25-cardiac-task3-dataset-specific.yaml"
echo "Config file: $CONFIG_FILE"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file found"

# =============================================================================
# Start Training
# =============================================================================

echo "=== Starting Multi-Dataset Training with Dataset-Specific Balancing ==="
echo "Start time: $(date)"
echo ""

# Run training with specific configuration
python main.py fit \
    --config $CONFIG_FILE \
    --trainer.logger.init_args.project cmr2024_2025_phased \
    --trainer.logger.init_args.name pmr_plus_cmr24_25_dataset_specific \
    --trainer.logger.init_args.tags "baseline,promptmr_plus,cmr24_25,dataset_specific" \
    --trainer.logger.init_args.save_dir $SAVE_DIR \
    --trainer.logger.init_args.log_model false \
    --trainer.logger.init_args.offline false \
    --trainer.max_epochs 50 \
    --trainer.devices 4 \
    --trainer.strategy ddp \
    --trainer.gradient_clip_val 0.001 \
    --trainer.log_every_n_steps 50 \
    --trainer.deterministic false \
    --trainer.use_distributed_sampler false \
    --model.init_args.lr 0.0002 \
    --model.init_args.lr_step_size 11

TRAINING_EXIT_CODE=$?

echo ""
echo "=== Training Completed ==="
echo "End time: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
fi

# =============================================================================
# Final Status
# =============================================================================

echo ""
echo "=== Final Status ==="
echo "Save directory: $SAVE_DIR"
echo "Config used: $CONFIG_FILE"
echo "Job completed at: $(date)"

exit $TRAINING_EXIT_CODE
