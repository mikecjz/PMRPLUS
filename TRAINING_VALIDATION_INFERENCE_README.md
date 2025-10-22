# Training Validation Inference 功能说明

## 功能概述

这个新功能允许您在使用training的validation data进行推理时，同时保存：
1. 重建后的图像（正常推理结果）
2. 被mask过的k-space数据
3. 使用的mask数据

## 配置文件

### 1. 正常推理模式
使用 `configs/inference/pmr-plus/cmr25-cardiac-training.yaml`
- 只保存重建图像
- 不保存masked k-space

### 2. Training Validation Inference模式
使用 `configs/inference/pmr-plus/cmr25-cardiac-training-validation.yaml`
- 保存重建图像
- 保存masked k-space和mask数据

## 输出文件结构

### 正常推理输出
```
TaskS2/MultiCoil/
└── original_file.mat          # 重建图像（仅包含重建结果）
```

### Training Validation Inference输出

#### 1. 重建图像（正常输出）
```
TaskS2/MultiCoil/
└── original_file.mat          # 重建图像（仅包含重建结果）
```

#### 2. Masked K-space数据
```
{配置的masked_kspace_output_dir}/train_val/TaskS2/MultiCoil/
├── LGE/ValidationSet/Mask_TaskS2/Center010/UIH_30T_umr790/P060/
│   └── lge_lax_4ch_mask_ktUniform16.mat
└── LGE/ValidationSet/UnderSample_TaskS2/Center010/UIH_30T_umr790/P060/
    └── lge_lax_4ch_kus_ktUniform16.mat
```

**示例**（使用默认配置）：
```
/common/lidxxlab/chaowei/data/CMR2025/TrainingValidationData/train_val/TaskS2/MultiCoil/
├── LGE/ValidationSet/Mask_TaskS2/Center010/UIH_30T_umr790/P060/
│   └── lge_lax_4ch_mask_ktUniform16.mat
└── LGE/ValidationSet/UnderSample_TaskS2/Center010/UIH_30T_umr790/P060/
    └── lge_lax_4ch_kus_ktUniform16.mat
```

## 文件名格式

### Mask文件命名
- 格式：`{原文件名}_mask_{mask类型}{低频频率}.mat`
- 示例：`lge_lax_4ch_mask_ktUniform16.mat`

### Masked K-space文件命名
- 格式：`{原文件名}_kus_{mask类型}{低频频率}.mat`
- 示例：`lge_lax_4ch_kus_ktUniform16.mat`

### Mask类型映射
- `kt_uniform` → `ktUniform`
- `kt_random` → `ktRandom`
- `kt_radial` → `ktRadial`
- `uniform` → `Uniform`

## 使用方法

### 1. 正常推理
```bash
python main.py --config configs/inference/pmr-plus/cmr25-cardiac-training.yaml predict
```

### 2. Training Validation Inference
```bash
python main.py --config configs/inference/pmr-plus/cmr25-cardiac-training-validation.yaml predict
```

## 配置参数

在YAML配置文件中，通过以下参数控制功能：

```yaml
callbacks:
  - class_path: __main__.CustomWriter
    init_args:
      output_dir: /path/to/output                    # 正常推理结果输出路径
      write_interval: batch_and_epoch
      save_masked_kspace: true                       # 设置为true启用masked k-space保存
      masked_kspace_output_dir: /path/to/masked_data # masked k-space数据输出路径（可选）
```

## 注意事项

1. **路径解析**：代码会自动从输入文件路径中提取Center、Scanner、Patient信息
2. **序列类型检测**：根据文件名自动检测序列类型（LGE、CINE、T1、T2等）
3. **目录创建**：会自动创建所需的目录结构
4. **数据格式**：保存的.mat文件包含相应的元数据信息

## 数据内容

### Mask文件内容
```python
{
    'mask': numpy_array,           # mask数据
    'mask_type': str,              # mask类型
    'num_low_frequencies': int     # 低频频率数量
}
```

### Masked K-space文件内容
```python
{
    'kus': numpy_array,            # masked k-space数据 [Ny, Nx, Ncoil, Nz, Nt]
    'mask_type': str,              # mask类型
    'num_low_frequencies': int,    # 低频频率数量
    'acceleration': int            # 加速因子
}
```

## 数据格式说明

### Masked K-space格式
- **数组名**: `kus`
- **维度**: `[Ny, Nx, Ncoil, Nz, Nt]`
  - `Ny`: 频率编码方向
  - `Nx`: 相位编码方向  
  - `Ncoil`: 线圈数量
  - `Nz`: 切片数量
  - `Nt`: 时间帧数（如果有时间维度）

### Mask格式
- **数组名**: `mask`
- **维度**: 
  - 3D数据: `[Ny, Nx]` (2D mask)
  - 4D数据: `[Ny, Nx, Nt]` (3D mask with time dimension)

### 与Validation Inference的兼容性
保存的masked k-space和mask数据格式与标准validation inference（如`cmr-task4-val.yaml`）读取的数据格式完全一致，确保数据兼容性。
