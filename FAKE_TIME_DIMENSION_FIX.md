# Fake Time Dimension 处理改进

## 问题描述

原代码在加载4D数据时会人为添加一个fake time dimension（通过复制数据），但在推理保存结果时没有正确移除这个假的维度，导致4D输入数据错误地输出4D结果而非期望的3D结果。

## 解决方案

### 1. 数据加载时添加标记 (`data/mri_data.py`)

在 `_load_volume` 方法中：
- 当检测到4D数据时，添加 `has_fake_time_dim = True` 标志
- 将此标志添加到 `attrs` 字典中

```python
# 第787行附近
elif len(kspace_volume.shape) == 4:
    kspace_volume = np.stack([kspace_volume,kspace_volume])
    has_fake_time_dim = True  # 新增标志
```

### 2. 数据样本中传递标志 (`data/transforms.py`)

在 `PromptMRSample` 类中：
- 添加 `has_fake_time_dim: bool` 字段
- 在 `CmrxReconDataTransform.__call__` 中传递此标志

### 3. 推理过程中保留标志 (`pl_modules/promptmr_module.py`)

在 `predict_step` 方法的返回值中：
- 添加 `'has_fake_time_dim': batch.has_fake_time_dim`

### 4. 保存时移除fake dimension (`main.py`)

在 `CustomWriter.write_on_epoch_end` 中：
- 检查 `has_fake_time_dim` 标志
- 如果为 `True` 且 `num_time_frames == 2`，则移除时间维度
- 保存3D结果而非4D结果

```python
# 第215行附近
if has_fake_time_dim and num_time_frames == 2:
    final_volume = final_4d_volume[0]  # 移除时间维度
    save_reconstructions(final_volume, fname, save_dir)  # 保存3D
else:
    save_reconstructions(final_4d_volume, fname, save_dir)  # 保存4D
```

## 效果

- ✅ 4D输入数据（无时间维度）→ 3D输出结果（正确移除fake time dimension）
- ✅ 5D输入数据（有真实时间维度）→ 4D输出结果（保持原有行为）
- ✅ 保持了对原有真实时间序列数据的兼容性

## 修改的文件

1. `data/mri_data.py` - 数据加载时添加标志
2. `data/transforms.py` - 样本类中添加标志字段
3. `pl_modules/promptmr_module.py` - 推理时传递标志
4. `main.py` - 保存时检查并移除fake dimension

## 测试

核心逻辑已通过测试验证，确保：
- 4D数据正确标记和处理
- 3D输出结果正确生成
- 原有5D数据功能不受影响






