import torch
import numpy as np

def batch_augment(points: torch.Tensor) -> torch.Tensor:
    """
    对批处理事件点云数据进行增强处理，适用于(t,x,y)格式的归一化数据
    
    该函数适用于形状为 [B, N, 3] 的事件点云数据，其中：
    - B: batch size (批大小)
    - N: 事件点数
    - 3: (t,x,y)坐标维度，t为时间，x,y为空间坐标
    
    数据特点：
    - 所有坐标都已归一化到(0,1)范围内
    - 时间维度t具有时序特性，需要特殊处理
    - 空间坐标x,y可以应用几何变换
    
    增强包含三个步骤：
    1. 时间维度添加微小噪声 (±0.01)
    2. 空间坐标随机平移 (±0.05范围)
    3. 空间坐标添加微小高斯噪声 (标准差0.005)
    
    参数:
        points (torch.Tensor): 输入事件点云数据，形状应为 [B, N, 3]，格式为(t,x,y)
        
    返回:
        torch.Tensor: 增强后的事件点云数据，形状与输入相同
        
    示例:
        >>> # 创建模拟事件点云数据 (batch_size=4, 点数=2048)
        >>> points = torch.rand(4, 2048, 3)  # (t,x,y)格式，范围(0,1)
        >>> 
        >>> # 应用增强
        >>> augmented_points = batch_augment(points)
        >>> 
        >>> # 检查输出形状
        >>> print(augmented_points.shape)  # 输出: torch.Size([4, 2048, 3])
        
    增强效果说明:
        原始事件点云   -> 时间噪声     -> 空间平移     -> 空间噪声
        [t,x,y]       [t±Δt,x,y]    [t±Δt,x±Δx,y±Δy] [t±Δt,x±Δx±ε,y±Δy±ε]
    """
    # 验证输入形状
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError("输入张量形状应为 [B, N, 3], 实际形状: {}".format(points.shape))
    
    # 验证数据范围 (应该在0-1之间)
    if points.min() < 0 or points.max() > 1:
        print("警告: 输入数据不在(0,1)范围内，可能影响增强效果")
    
    # 1. 时间维度添加微小噪声 (±0.01)
    # 时间噪声应该很小，保持时序特性
    time_noise = (torch.rand_like(points[..., 0:1]) * 0.02 - 0.01)  # ±0.01
    points_with_time_noise = points + time_noise
    
    # 2. 空间坐标随机平移 (±0.05)
    # 为每个batch生成独立的空间偏移量，仅影响x,y坐标
    spatial_shift = torch.zeros_like(points)
    spatial_shift[..., 1:] = (torch.rand(points.size(0), 1, 2, device=points.device) * 0.1 - 0.05)  # ±0.05
    points_spatial_shifted = points_with_time_noise + spatial_shift
    
    # 3. 空间坐标添加微小高斯噪声 (标准差0.005)
    # 仅对x,y坐标添加噪声
    spatial_noise = torch.zeros_like(points_spatial_shifted)
    spatial_noise[..., 1:] = torch.randn_like(points_spatial_shifted[..., 1:]) * 0.005
    augmented_points = points_spatial_shifted + spatial_noise
    
    # 4. 确保结果仍在(0,1)范围内
    augmented_points = torch.clamp(augmented_points, 0.0, 1.0)
    
    return augmented_points


# 示例使用代码
if __name__ == "__main__":
    # 创建模拟事件点云数据 (batch_size=2, 点数=2048)
    # 生成(t,x,y)格式的数据，范围在(0,1)
    test_points = torch.rand(2, 2048, 3)  # t,x,y都在(0,1)范围内
    
    print("原始事件点云示例:")
    print(f"形状: {test_points.shape}")
    print(f"第一个事件: t={test_points[0, 0, 0]:.3f}, x={test_points[0, 0, 1]:.3f}, y={test_points[0, 0, 2]:.3f}")
    print(f"时间范围: ({test_points[...,0].min():.3f}-{test_points[...,0].max():.3f})")
    print(f"X范围: ({test_points[...,1].min():.3f}-{test_points[...,1].max():.3f})")
    print(f"Y范围: ({test_points[...,2].min():.3f}-{test_points[...,2].max():.3f})")
    
    # 应用增强
    augmented = batch_augment(test_points)
    
    print("\n增强后事件点云示例:")
    print(f"形状: {augmented.shape}")
    print(f"第一个事件: t={augmented[0, 0, 0]:.3f}, x={augmented[0, 0, 1]:.3f}, y={augmented[0, 0, 2]:.3f}")
    print(f"时间范围: ({augmented[...,0].min():.3f}-{augmented[...,0].max():.3f})")
    print(f"X范围: ({augmented[...,1].min():.3f}-{augmented[...,1].max():.3f})")
    print(f"Y范围: ({augmented[...,2].min():.3f}-{augmented[...,2].max():.3f})")
    
    # 计算变换量
    time_displacement = (augmented[..., 0] - test_points[..., 0]).abs()
    spatial_displacement = (augmented[..., 1:] - test_points[..., 1:]).norm(dim=-1)
    
    print(f"\n时间维度平均变化: {time_displacement.mean():.5f} ± {time_displacement.std():.5f}")
    print(f"空间维度平均位移: {spatial_displacement.mean():.5f} ± {spatial_displacement.std():.5f}")
    
    # 验证数据范围
    print(f"\n增强后数据范围验证:")
    print(f"最小值: {augmented.min():.5f}, 最大值: {augmented.max():.5f}")
    print(f"是否在(0,1)范围内: {augmented.min() >= 0 and augmented.max() <= 1}")