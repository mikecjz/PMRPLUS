#!/bin/bash
#SBATCH --job-name=CMR25_T4_5kloss
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:h100:7
#SBATCH --nodelist=esplhpc-cp088
#SBATCH --mail-user=Yi.Zheng@cshs.org
#SBATCH --mail-type=BEGIN,END,FAIL

# 让脚本在出错时立即退出
set -e

# 激活conda环境（确保包含 torch 等依赖）
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate cmr || { echo "[FATAL] Conda 环境 'cmr' 未找到或无法激活"; exit 1; }
else
    echo "[FATAL] 未检测到 conda，请先加载/安装 Anaconda 并创建环境 'cmr'"; exit 1
fi

# 设置项目根目录和保存目录
CMRROOT=/common/lidxxlab/Yi/PromptMR-plus-Task4_kloss
SAVE_DIR=/common/lidxxlab/Yi/training_results_folder/multi_dataset_training_kloss

# 显示GPU信息
echo "=== GPU Information ==="
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 显示训练配置信息
echo "=== Training Configuration ==="
echo "Project Root: $CMRROOT"
echo "Save Directory: $SAVE_DIR"
echo "Config File: $CMRROOT/configs/train/pmr-plus/cmr24-25-cardiac-task4-dataset-specific_xlk.yaml"
echo ""

# 快速检查 PyTorch 是否可用
echo "=== Checking PyTorch in current environment ==="
python - <<'PY'
try:
    import torch
    import torchvision  # noqa: F401
    print("PyTorch OK:", torch.__version__)
except Exception as e:
    import sys
    print("[FATAL] PyTorch import failed:", e)
    sys.exit(1)
PY
echo ""

# 登录wandb (请替换为您的wandb API key)
echo "=== Logging into Weights & Biases ==="
wandb login fac722f823bd0fbc4c5b3b255e9c3e8266aaf362
echo ""

# 检查数据集路径
echo "=== Checking Dataset Paths ==="
DATASET_2024="/common/lidxxlab/cmrchallenge/data/CMR2024/Processed"
# DATASET_2025="/common/lidxxlab/cmrchallenge/data/CMR2025/Processed"
DATASET_2025="/common/lidxxlab/Yifan/PromptDiff/CMR2025/Processed_addmeta"



if [ -d "$DATASET_2024" ]; then
    echo "✓ CMR2024 dataset found: $DATASET_2024"
    echo "  - Train samples: $(ls $DATASET_2024/train/*.h5 | wc -l)"
    echo "  - Val samples: $(ls $DATASET_2024/val/*.h5 | wc -l)"
else
    echo "✗ CMR2024 dataset not found: $DATASET_2024"
    exit 1
fi

if [ -d "$DATASET_2025" ]; then
    echo "✓ CMR2025 dataset found: $DATASET_2025"
    echo "  - Train samples: $(ls $DATASET_2025/train/*.h5 | wc -l)"
    echo "  - Val samples: $(ls $DATASET_2025/val/*.h5 | wc -l)"
else
    echo "✗ CMR2025 dataset not found: $DATASET_2025"
    exit 1
fi
echo ""

# 创建保存目录
mkdir -p $SAVE_DIR/promptmr-plus/CMR2024_2025_dataset_specific
echo "✓ Created save directory: $SAVE_DIR/promptmr-plus/CMR2024_2025_dataset_specific"
echo ""

# 开始训练
echo "=== Starting Multi-Dataset Training with Dataset-Specific Balancing + K-Space Loss ==="
echo "Training on CMR2024 + CMR2025 datasets"
echo "Model: PromptMR+ with dataset-specific balancing and k-space loss"
echo ""

cd $CMRROOT

python main.py fit \
    -c $CMRROOT/configs/train/pmr-plus/cmr24-25-cardiac-task4-dataset-specific_xlk.yaml \
    --trainer.devices=auto \
    --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2024_2025_dataset_specific \
    --trainer.logger.init_args.project=cmr2024_2025_phased \
    --trainer.logger.init_args.name=pmr_plus_cmr24_25_dataset_specific_kloss \
    --trainer.logger.init_args.tags="[baseline, promptmr_plus, cmr24_25, dataset_specific, kspace_loss]" \
    --model.init_args.pretrain=False \
    --model.init_args.n_history=15 \
    --model.init_args.num_cascades=16 \
    --model.init_args.num_adj_slices=7 \
    --model.init_args.kspace_loss_weight=0.01 \
    --ckpt_path=/common/lidxxlab/Yi/training_results_folder/multi_dataset_training_kloss/promptmr-plus/CMR2024_2025_dataset_specific/cmr2024_2025_phased/fbrtyxad/checkpoints/best-epochepoch=08-valvalidation_loss=0.0144.ckpt

echo ""
echo "=== Training Completed ==="
echo "Results saved to: $SAVE_DIR/promptmr-plus/CMR2024_2025_dataset_specific"
echo "Check wandb dashboard for detailed training logs"
