# Multi-Dataset Training for CMR 2024 + 2025

这个项目已经成功修改为支持同时使用2024和2025两套CMR数据集进行训练。

## 主要修改

### 1. 新增多数据集数据模块
- **文件**: `pl_modules/multi_dataset_module.py`
- **功能**: 支持同时加载和处理多个数据集
- **特点**: 
  - 自动处理不同数据集的命名差异
  - 统一的数据平衡策略
  - 支持不同的序列类型映射

### 2. 智能数据平衡器
- **类**: `MultiDatasetBalanceSampler`
- **功能**: 
  - 自动识别2024和2025数据集的命名格式
  - 将不同命名方式的序列类型映射到统一的平衡策略
  - 支持25种不同的序列类型

### 3. 新的训练配置
- **文件**: `configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml`
- **特点**:
  - 使用新的多数据集数据模块
  - 配置了两个数据路径
  - 统一的模型参数和训练设置

## 数据集命名差异处理

### 2024数据集格式
```
P001_cine_lvot.h5
P002_cine_sax.h5
P003_T1map.h5
```

### 2025数据集格式
```
Center001_UIH_30T_umr780_P001_cine_lax_3ch.h5
Center001_UIH_30T_umr780_P002_cine_sax.h5
Center001_UIH_30T_umr780_P003_T1map.h5
```

### 序列类型映射
系统会自动将以下序列类型进行映射：
- `cine_lvot` ↔ `cine_lvot`
- `cine_sax` ↔ `cine_sax`
- `cine_lax` ↔ `cine_lax`
- `T1map` ↔ `T1map`
- `T2map` ↔ `T2map`
- 等等...

## 使用方法

### 1. 环境准备
```bash
# SSH到计算节点
ssh esplhpc-cp075

# 切换到项目目录
cd /common/lidxxlab/Yi/PromptMR-plus-Task3_large

# 激活conda环境
conda activate cmr
```

### 2. 测试配置
```bash
# 测试多数据集平衡器
python test_multi_dataset.py

# 测试训练配置
python test_training_config.py

# 验证完整配置（dry-run模式）
python start_multi_dataset_training.py --dry-run
```

### 3. 开始训练
```bash
# 使用新的多数据集配置开始训练
python start_multi_dataset_training.py

# 或者直接使用main.py
python main.py --config configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml
```

## 配置参数说明

### 数据配置
- **数据路径**: 同时使用2024和2025两个数据集
- **数据平衡**: 25种序列类型的平衡策略
- **数据变换**: 使用2025的mask函数（兼容2024数据）

### 模型配置
- **n_history**: 15 (历史特征数量)
- **num_cascades**: 16 (级联块数量)
- **num_adj_slices**: 7 (相邻切片数量)

### 训练配置
- **设备**: 4个GPU
- **最大轮数**: 50
- **学习率**: 0.0002
- **项目名**: cmr2024_2025_phased

## 监控和日志

训练过程中会记录：
- 每个序列类型的样本数量和平衡比例
- 数据集加载状态
- 训练和验证损失
- 模型检查点保存

## 文件结构

```
PromptMR-plus-Task3_large/
├── pl_modules/
│   └── multi_dataset_module.py          # 多数据集数据模块
├── configs/train/pmr-plus/
│   └── cmr24-25-cardiac-task3.yaml     # 多数据集训练配置
├── test_multi_dataset.py               # 多数据集测试脚本
├── test_training_config.py             # 配置测试脚本
├── start_multi_dataset_training.py     # 训练启动脚本
└── MULTI_DATASET_README.md             # 本说明文件
```

## 注意事项

1. **数据路径**: 确保两个数据集路径都存在且可访问
2. **内存使用**: 多数据集训练会增加内存使用量
3. **训练时间**: 由于数据量增加，训练时间会相应增加
4. **存储空间**: 确保有足够的存储空间保存模型检查点

## 故障排除

### 常见问题
1. **导入错误**: 确保在正确的conda环境中运行
2. **数据路径错误**: 检查数据集路径是否正确
3. **内存不足**: 考虑减少batch_size或使用更少的GPU

### 调试方法
1. 使用dry-run模式测试配置
2. 检查日志输出中的错误信息
3. 验证数据集文件是否完整

## 性能优化建议

1. **数据缓存**: 启用数据集缓存以提高加载速度
2. **多进程**: 适当增加num_workers参数
3. **混合精度**: 考虑使用混合精度训练
4. **梯度累积**: 如果GPU内存不足，可以使用梯度累积

## 联系信息

如有问题，请联系开发团队或查看项目文档。
