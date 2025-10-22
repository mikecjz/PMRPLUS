# SBATCH 训练脚本使用说明

## 脚本文件

1. **`CMR2024_2025_train_multi_dataset.sh`** - 完整版本，包含详细检查和日志
2. **`CMR2024_2025_train_simple.sh`** - 简化版本，基本功能

## 使用步骤

### 1. 获取您的 Weights & Biases API Key

1. 访问 [wandb.ai](https://wandb.ai)
2. 登录您的账户
3. 进入 Settings → API Keys
4. 复制您的 API Key

### 2. 修改脚本中的 API Key

编辑脚本文件，将 `YOUR_WANDB_API_KEY_HERE` 替换为您的实际 API Key：

```bash
# 在脚本中找到这一行
wandb login YOUR_WANDB_API_KEY_HERE

# 替换为您的实际 API Key，例如：
wandb login 888ea4d187a6f809d8cf6dda7de79991e057d892
```

### 3. 提交训练任务

```bash
# 方法1: 使用完整版本脚本
sbatch scripts/CMR2024_2025_train_multi_dataset.sh

# 方法2: 使用简化版本脚本
sbatch scripts/CMR2024_2025_train_simple.sh
```

### 4. 监控训练进度

```bash
# 查看任务状态
squeue -u $USER

# 查看任务日志
tail -f slurm-<job_id>.out

# 查看 Weights & Biases 仪表板
# 访问 https://wandb.ai/your-username/cmr2024_2025_phased
```

## 脚本配置说明

### SBATCH 参数
- `--job-name=CMR2024_2025_multi` - 任务名称
- `--partition=gpu` - 使用GPU分区
- `--cpus-per-task=12` - 12个CPU核心
- `--mem=200G` - 200GB内存
- `--time=168:00:00` - 最大运行时间7天
- `--gres=gpu:l40s:4` - 4个L40S GPU
- `--mail-user=Yi.Zheng@cshs.org` - 邮件通知地址
- `--mail-type=END,FAIL` - 任务结束或失败时发送邮件

### 训练参数
- **数据集**: CMR2024 + CMR2025
- **模型**: PromptMR+ with multi-dataset support
- **配置**: `cmr24-25-cardiac-task3.yaml`
- **保存目录**: `/common/lidxxlab/Yi/training_results_folder/multi_dataset_training`
- **Wandb项目**: `cmr2024_2025_phased`

## 自定义修改

### 修改训练时间
```bash
# 修改 --time 参数
--time=72:00:00  # 3天
--time=24:00:00  # 1天
```

### 修改GPU数量
```bash
# 修改 --gres 参数
--gres=gpu:l40s:2  # 2个GPU
--gres=gpu:l40s:8  # 8个GPU
```

### 修改保存目录
```bash
# 修改脚本中的 SAVE_DIR
SAVE_DIR=/your/custom/path
```

### 修改邮件地址
```bash
# 修改 --mail-user 参数
--mail-user=your.email@example.com
```

## 故障排除

### 常见问题

1. **API Key 错误**
   - 确保API Key正确且有效
   - 检查网络连接

2. **数据集路径错误**
   - 确保数据集路径存在
   - 检查文件权限

3. **GPU内存不足**
   - 减少batch_size
   - 使用更少的GPU

4. **训练时间超限**
   - 增加 `--time` 参数
   - 使用检查点恢复训练

### 恢复训练

如果训练中断，可以使用检查点恢复：

```bash
python main.py fit \
    -c $CMRROOT/configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml \
    --trainer.devices=auto \
    --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2024_2025 \
    --model.init_args.pretrain=False \
    --ckpt_path=/path/to/checkpoint.ckpt
```

## 监控和日志

### Wandb 仪表板
- 项目: `cmr2024_2025_phased`
- 运行名称: `pmr_plus_cmr24_25_baseline`
- 标签: `[baseline, promptmr_plus, cmr24_25, multi_dataset]`

### 本地日志
- 训练日志: `slurm-<job_id>.out`
- 模型检查点: `$SAVE_DIR/promptmr-plus/CMR2024_2025/`

## 联系信息

如有问题，请联系开发团队或查看项目文档。
