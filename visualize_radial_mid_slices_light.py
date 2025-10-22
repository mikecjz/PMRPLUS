import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import scipy.io

def load_mat_file(file_path):
    """
    加载.mat文件，支持新旧格式
    """
    try:
        # 首先尝试使用scipy.io.loadmat (旧格式)
        mat_data = scipy.io.loadmat(file_path)
        # 移除系统变量
        keys_to_remove = ['__header__', '__version__', '__globals__']
        for key in keys_to_remove:
            if key in mat_data:
                del mat_data[key]
        
        # 获取第一个数据键
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if data_keys:
            return mat_data[data_keys[0]]
        else:
            raise ValueError("No data found in .mat file")
            
    except NotImplementedError:
        # 如果是新格式(v7.3)，使用h5py
        try:
            with h5py.File(file_path, 'r') as f:
                # 获取所有键
                keys = list(f.keys())
                if keys:
                    # 获取第一个数据键
                    data_key = keys[0]
                    data = f[data_key][:]
                    # 如果是MATLAB格式，需要转置
                    if len(data.shape) >= 2:
                        data = np.transpose(data)
                    return data
                else:
                    raise ValueError("No data found in .mat file")
        except Exception as e:
            print(f"h5py读取失败: {e}")
            raise
    except Exception as e:
        print(f"scipy.io.loadmat读取失败: {e}")
        # 尝试使用h5py作为备选方案
        try:
            with h5py.File(file_path, 'r') as f:
                # 获取所有键
                keys = list(f.keys())
                if keys:
                    # 获取第一个数据键
                    data_key = keys[0]
                    data = f[data_key][:]
                    # 如果是MATLAB格式，需要转置
                    if len(data.shape) >= 2:
                        data = np.transpose(data)
                    return data
                else:
                    raise ValueError("No data found in .mat file")
        except Exception as h5py_error:
            print(f"h5py备选方案也失败: {h5py_error}")
            raise ValueError(f"无法读取.mat文件: {e}")

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

def visualize_radial_mid_slices_light(inference_output_dir, save_dir=None, max_files_per_type=3):
    """
    轻量级版本：展示inference结果中radial数据的中间切片
    """
    
    # 设置输出目录路径
    output_path = Path(inference_output_dir) / "TaskR2" / "MultiCoil"
    
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
    
    # 处理每种类型的文件（限制数量）
    successful_files = []
    
    for file_type, files in file_groups.items():
        print(f"\n=== 处理 {file_type} 类型文件 ({len(files)} 个) ===")
        
        # 每种类型只处理前几个文件
        files_to_process = files[:max_files_per_type]
        
        for i, file_path in enumerate(files_to_process):
            try:
                print(f"处理文件 {i+1}/{len(files_to_process)}: {file_path.name}")
                
                # 加载数据
                data = load_mat_file(file_path)
                
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
    
    # 可视化结果
    if not successful_files:
        print("没有成功读取任何radial文件")
        return
    
    print(f"\n=== 可视化Radial中间切片 ===")
    print(f"成功处理: {len(successful_files)} 个文件")
    
    # 按类型分组可视化
    for file_type in file_groups.keys():
        type_files = [f for f in successful_files if f['type'] == file_type]
        if not type_files:
            continue
        
        print(f"\n可视化 {file_type} 类型文件 ({len(type_files)} 个)")
        
        # 计算子图布局
        num_files = len(type_files)
        num_cols = min(num_files, 3)  # 最多3列
        num_rows = (num_files + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        
        # 确保axes是二维数组
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, file_info in enumerate(type_files):
            row = i // num_cols
            col = i % num_cols
            
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
            row = i // num_cols
            col = i % num_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'{file_type} Radial Data Mid Slices', fontsize=16)
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
    inference_output_dir = "/common/lidxxlab/cmrchallenge/code/Inference/inf_0814_xl2balancer_epoch1"
    
    # 设置保存目录（可选）
    save_dir = "./radial_visualization_results"
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 运行可视化（每种类型最多处理3个文件）
    visualize_radial_mid_slices_light(inference_output_dir, save_dir=save_dir, max_files_per_type=3)


