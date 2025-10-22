# Dataset-Specific Multi-Dataset Training

## 概述

这个版本实现了**数据集特定的平衡策略**，允许2024和2025数据集使用不同的平衡比例，而不是统一的平衡策略。

## 主要特性

### 🔄 数据集特定平衡
- **2024数据集**: 使用2024单独训练时的平衡策略
- **2025数据集**: 使用2025单独训练时的平衡策略
- **自动检测**: 根据文件名自动识别数据集类型

### 📊 平衡策略对比

| 序列类型 | 2024单独训练 | 2025单独训练 | 数据集特定策略 |
|----------|-------------|-------------|----------------|
| `cine_lvot` | 6 | 8 | 2024: 6, 2025: 8 |
| `cine_sax` | 1 | 1 | 2024: 1, 2025: 1 |
| `T1map` | 2 | 3 | 2024: 2, 2025: 3 |
| `T2map` | 6 | 4 | 2024: 6, 2025: 4 |
| `cine_lax` | 2 | 8 | 2024: 2, 2025: 8 |
| `perfusion` | - | 8 | 2024: -, 2025: 8 |
| `T1rho` | - | 8 | 2024: -, 2025: 8 |

## 文件结构

```
PromptMR-plus-Task3_large2/
├── pl_modules/
│   └── multi_dataset_module.py          # 改进的多数据集平衡器
├── configs/train/pmr-plus/
│   └── cmr24-25-cardiac-task3-dataset-specific.yaml  # 新配置文件
└── scripts/
    ├── CMR2024_2025_train_dataset_specific.sh        # 完整训练脚本
    └── CMR2024_2025_train_dataset_specific_simple.sh # 简化训练脚本
```

## 使用方法

### 1. 快速开始

```bash
# 激活环境
source activate cmr

# 运行简化训练脚本
bash scripts/CMR2024_2025_train_dataset_specific_simple.sh
```

### 2. 使用完整脚本

```bash
# 使用sbatch提交作业
sbatch scripts/CMR2024_2025_train_dataset_specific.sh

# 或直接运行
bash scripts/CMR2024_2025_train_dataset_specific.sh
```

### 3. 手动运行

```bash
# 激活环境
source activate cmr

# 设置路径
export CMRROOT=/common/lidxxlab/Yi/PromptMR-plus-Task3_large2
export SAVE_DIR=/common/lidxxlab/Yi/training_results_folder/multi_dataset_training_dataset_specific

# 切换到项目目录
cd $CMRROOT

# 登录wandb
wandb login YOUR_API_KEY

# 创建保存目录
mkdir -p $SAVE_DIR

# 开始训练
python main.py fit \
    --config configs/train/pmr-plus/cmr24-25-cardiac-task3-dataset-specific.yaml \
    --trainer.logger.init_args.save_dir $SAVE_DIR
```

## 技术细节

### 数据集检测
- **2024数据集**: 文件名不包含 "Center" 前缀
- **2025数据集**: 文件名包含 "Center" 前缀

### 平衡逻辑
1. **自动检测**: 根据文件名识别数据集类型
2. **分别处理**: 对每个数据集使用对应的平衡策略
3. **独立平衡**: 2024和2025数据分别进行平衡
4. **合并输出**: 将两个数据集的平衡结果合并

### 日志输出
训练时会显示详细的平衡信息：
```
2024 - Sequence type 'cine_lvot': 100 samples, ratio 6
2025 - Sequence type 'cine_lvot': 150 samples, ratio 8
2024 - Sequence type 'T1map': 50 samples, ratio 2
2025 - Sequence type 'T1map': 80 samples, ratio 3
Total balanced samples: 2340
```

## 配置说明

### 2024数据集平衡策略
```yaml
ratio_dict_2024: {
  'T1map': 2, 
  'T2map': 6, 
  'cine_lax': 2, 
  'cine_sax': 1, 
  'cine_lvot': 6, 
  'aorta_sag': 1, 
  'aorta_tra': 1,
  'tagging': 1
}
```

### 2025数据集平衡策略
```yaml
ratio_dict_2025: {
  'cine_rvot': 8,
  'cine_sax': 1,
  'lge_lax_4ch': 8,
  'flow2d': 3,
  'cine_lax': 8,
  'T1w': 4,
  'lge_sax': 2,
  'T2map': 4,
  'perfusion': 8,
  'T1rho': 8,
  'T1map': 3,
  'cine_lax_3ch': 8,
  'lge_lax_2ch': 8,
  'cine_lax_2ch': 8,
  'T1mappost': 8,
  'T2w': 2,
  'cine_lax_4ch': 8,
  'lge_lax_3ch': 8,
  'blackblood': 8,
  'cine_lvot': 8,
  'cine_ot': 8,
  'lge_lax': 8,
  'cine_lax_r2ch': 8,
  'T2smap': 8,
}
```

## 优势

1. **更精确的平衡**: 每个数据集使用最适合的平衡策略
2. **保持原有性能**: 不破坏单独训练时的优化效果
3. **灵活配置**: 可以轻松调整每个数据集的平衡策略
4. **向后兼容**: 支持统一平衡策略作为后备选项

## 监控训练

### Weights & Biases
- 项目: `cmr2024_2025_phased`
- 运行名称: `pmr_plus_cmr24_25_dataset_specific`
- 标签: `baseline,promptmr_plus,cmr24_25,dataset_specific`

### 检查点保存
- 保存5个最佳模型
- 每10个epoch保存一次最新模型
- 保存目录: `/common/lidxxlab/Yi/training_results_folder/multi_dataset_training_dataset_specific`

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保在正确的环境中
   source activate cmr
   cd /common/lidxxlab/Yi/PromptMR-plus-Task3_large2
   ```

2. **数据集路径错误**
   ```bash
   # 检查数据集路径
   ls /common/lidxxlab/cmrchallenge/data/CMR2024/Processed
   ls /common/lidxxlab/cmrchallenge/data/CMR2025/Processed
   ```

3. **GPU内存不足**
   ```bash
   # 减少batch_size或使用更少的GPU
   --trainer.devices 2
   ```

## 版本历史

- **v1.0**: 初始版本，支持数据集特定平衡策略
- 支持2024和2025数据集的独立平衡
- 自动数据集类型检测
- 详细的日志输出
