# 快速开始 - 多数据集训练

## 快速启动命令

```bash
# 1. SSH到计算节点
ssh esplhpc-cp075

# 2. 切换到项目目录并激活环境
cd /common/lidxxlab/Yi/PromptMR-plus-Task3_large
conda activate cmr

# 3. 开始训练（使用2024+2025数据集）
python main.py --config configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml
```

## 主要改进

✅ **支持多数据集**: 同时使用2024和2025两套数据集  
✅ **智能命名处理**: 自动处理不同数据集的命名差异  
✅ **统一数据平衡**: 25种序列类型的平衡策略  
✅ **保持原有功能**: 所有原有模型参数和训练设置保持不变  

## 配置文件

- **新配置**: `configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml`
- **原配置**: `configs/train/pmr-plus/cmr25-cardiac-task3.yaml` (仅2025数据集)

## 关键参数

- **n_history**: 15 (保持不变)
- **数据路径**: 2024 + 2025 两个数据集
- **序列类型**: 25种 (包含2024和2025的所有类型)
- **训练设置**: 4 GPU, 50 epochs, 学习率 0.0002

## 监控

训练日志会显示：
- 每个序列类型的样本数量
- 数据平衡情况
- 训练进度和损失

详细说明请参考 `MULTI_DATASET_README.md`
