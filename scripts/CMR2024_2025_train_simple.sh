#!/bin/bash
#SBATCH --job-name=CMR2024_2025_multi
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --mail-user=Yi.Zheng@cshs.org
#SBATCH --mail-type=END,FAIL

# 激活conda环境
source activate cmr

# 设置路径
CMRROOT=/common/lidxxlab/Yi/PromptMR-plus-Task3_large
SAVE_DIR=/common/lidxxlab/Yi/training_results_folder/multi_dataset_training

# 显示GPU信息
echo "GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 登录wandb (请替换为您的wandb API key)
echo "Logging into Weights & Biases..."
wandb login fac722f823bd0fbc4c5b3b255e9c3e8266aaf362

# 切换到项目目录
cd $CMRROOT

# 开始训练
echo "Starting Multi-Dataset Training (CMR2024 + CMR2025)..."

python main.py fit \
    -c $CMRROOT/configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml \
    --trainer.devices=auto \
    --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2024_2025 \
    --model.init_args.pretrain=False

echo "Training completed!"
