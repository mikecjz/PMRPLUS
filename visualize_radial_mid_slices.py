#!/usr/bin/env python3
"""
Convert all MATLAB .mat files (both v7.3/HDF5 and legacy formats) under a hard‑coded input directory into grayscale PNG images, preserving directory structure.

For multi-dimensional arrays (>2D), collapses all leading dimensions via mean to produce a single 2D slice.
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import scipy.io
from PIL import Image

def load_mat(path: Path) -> np.ndarray:
    """Load a .mat file (legacy v5 or HDF5-based v7.3)."""
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No valid variables in {path}")
            return f[keys[0]][()]
    try:
        mat = scipy.io.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No valid variables in {path}")
        return mat[keys[0]]
    except Exception as e:
        raise ValueError(f"Cannot load MAT file: {path} ({e})")

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 range (0-255)."""
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255
    return arr.astype(np.uint8)

def find_radial_files_by_pattern(output_path):
    """
    根据特定模式查找radial文件
    """
    radial_files = []
    
    # 查找所有.mat文件
    mat_files = list(output_path.rglob("*.mat"))
    
    # 过滤出包含"ktRadial"的文件
    for file_path in mat_files:
        if "ktRadial" in file_path.name:
            radial_files.append(file_path)
    
    return radial_files

def visualize_radial_mid_slices(inference_output_dir, save_dir=None, max_images_per_row=4):
    """
    展示inference结果中所有radial数据的中间切片
    
    Args:
        inference_output_dir: inference结果的输出目录
        save_dir: 保存图片的目录，如果为None则显示图片
        max_images_per_row: 每行最多显示的图片数量
    """
    
    # 设置输出目录路径
    output_path = Path(inference_output_dir) 
    
    if not output_path.exists():
        print(f"错误：目录不存在 {output_path}")
        return
    
    # 查找所有radial相关文件
    radial_files = find_radial_files_by_pattern(output_path)
    
    if not radial_files:
        print("未找到任何radial文件")
        return
    
    print(f"找到 {len(radial_files)} 个radial文件")
    
    # 按类型分组
    file_groups = {}
    for file_path in radial_files:
        # 提取文件类型（如T1w, perfusion, T1map等）
        filename = file_path.name
        if "T1w" in filename:
            file_type = "T1w"
        elif "perfusion" in filename:
            file_type = "perfusion"
        elif "T1map" in filename:
            file_type = "T1map"
        elif "T2map" in filename:
            file_type = "T2map"
        elif "lge" in filename:
            file_type = "lge"
        elif "T2w" in filename:
            file_type = "T2w"
        elif "cine" in filename:
            file_type = "cine"
        else:
            file_type = "other"
        
        if file_type not in file_groups:
            file_groups[file_type] = []
        file_groups[file_type].append(file_path)
    
    # 处理每种类型的文件
    successful_files = []
    failed_files = []
    
    for file_type, files in file_groups.items():
        print(f"\n=== 处理 {file_type} 类型文件 ({len(files)} 个) ===")
        
        for i, file_path in enumerate(files[:10]):  # 每种类型最多处理10个文件
            try:
                print(f"处理文件 {i+1}/{min(len(files), 10)}: {file_path.name}")
                
                # 加载数据
                data = load_mat(file_path)
                
                # 分析数据形状
                print(f"  数据形状: {data.shape}")
                print(f"  数据类型: {data.dtype}")
                print(f"  数值范围: [{data.min():.4f}, {data.max():.4f}]")
                
                # 确定中间切片
                if len(data.shape) == 3:
                    mid_slice = data.shape[2] // 2
                    slice_data = data[:, :, mid_slice]
                elif len(data.shape) == 4:
                    mid_slice = data.shape[3] // 2
                    slice_data = data[:, :, :, mid_slice]
                    # 如果是4D数据，取第一个通道
                    if slice_data.shape[0] > 1:
                        slice_data = slice_data[0, :, :]
                else:
                    print(f"  跳过：不支持的维度 {len(data.shape)}")
                    continue
                
                successful_files.append({
                    'path': file_path,
                    'type': file_type,
                    'data': slice_data,
                    'original_shape': data.shape
                })
                
            except Exception as e:
                print(f"  处理文件 {file_path.name} 时出错: {e}")
                failed_files.append(file_path)
    
    # 可视化结果
    if not successful_files:
        print("没有成功读取任何radial文件")
        return
    
    print(f"\n=== 可视化Radial中间切片 ===")
    print(f"成功处理: {len(successful_files)} 个文件")
    print(f"失败: {len(failed_files)} 个文件")
    
    # 按类型分组可视化
    for file_type in file_groups.keys():
        type_files = [f for f in successful_files if f['type'] == file_type]
        if not type_files:
            continue
        
        print(f"\n可视化 {file_type} 类型文件 ({len(type_files)} 个)")
        
        # 计算子图布局
        num_files = len(type_files)
        num_rows = (num_files + max_images_per_row - 1) // max_images_per_row
        num_cols = min(num_files, max_images_per_row)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, file_info in enumerate(type_files):
            row = i // max_images_per_row
            col = i % max_images_per_row
            
            ax = axes[row, col]
            
            # 显示图像
            im = ax.imshow(file_info['data'], cmap='gray', aspect='auto')
            ax.set_title(f"{file_info['path'].name}\n{file_info['original_shape']}", 
                        fontsize=8, pad=5)
            ax.axis('off')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(num_files, num_rows * num_cols):
            row = i // max_images_per_row
            col = i % max_images_per_row
            axes[row, col].axis('off')
        
        plt.suptitle(f'{file_type} 类型 Radial 数据中间切片', fontsize=16)
        plt.tight_layout()
        
        # 保存或显示
        if save_dir:
            save_path = Path(save_dir) / f'radial_mid_slices_{file_type}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"保存图片到: {save_path}")
        else:
            plt.show()
        
        plt.close()

if __name__ == "__main__":
    # 设置inference输出目录
    inference_output_dir = "/common/lidxxlab/cmrchallenge/code/Inference/inf_0814_xl2balancer_epoch1/"
    
    # 设置保存目录（可选）
    save_dir = "/common/lidxxlab/cmrchallenge/code/Inference/infcheck_png/radial_visualization_results"
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 运行可视化
    visualize_radial_mid_slices(inference_output_dir, save_dir=save_dir)