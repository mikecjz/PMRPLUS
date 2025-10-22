#!/bin/bash
#SBATCH --job-name=CMRscratch
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --mail-user=Yi.Zheng@cshs.org
#SBATCH --mail-type=END,FAIL



source activate cmr


CMRROOT=/common/lidxxlab/Yi/PromptMR-plus_CMR17/PromptMR-plus
SAVE_DIR=/common/lidxxlab/Yi/training_results_folder/DG_training_scratch

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login 888ea4d187a6f809d8cf6dda7de79991e057d892


# Run training script with the correct LightningCLI format

python main.py fit \
    -c $CMRROOT/configs/base.yaml \
    -c $CMRROOT/configs/train/pmr-plus/cmr25-cardiac-upd.yaml \
    -c $CMRROOT/configs/model/pmr-plus.yaml \
    --trainer.devices=auto \
    --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2025 \
    --model.init_args.pretrain=False \
    #--model.init_args.pretrain_weights_path=/common/lidxxlab/cmrchallenge/code/chaowei/experiments/cmr25/promptmr-plus/CMR2025/deep_recon/uec2kxvx/checkpoints/last.ckpt
    #/common/lidxxlab/chushu/training_results_folder/num_slice_3_continued/promptmr-plus/CMR2025/deep_recon/c6o5qby4/checkpoints/last.ckpt
    #/common/lidxxlab/chushu/training_results_folder/num_slice_3/promptmr-plus/CMR2025/deep_recon/zm3n3quz/checkpoints/last.ckpt
    #/common/lidxxlab/chushu/PromptMR-plus_CMR17/PromptMR-plus/ckpt_folder/converted_3_channel_new.ckpt
    #continued training with 3-channel model