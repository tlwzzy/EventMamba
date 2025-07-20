import torch
import numpy as np

def batch_augment(points: torch.Tensor) -> torch.Tensor:
    """
    对批处理点云数据进行增强处理，包含随机旋转、平移和高斯噪声
    
    该函数适用于形状为 [B, N, 3] 的点云数据，其中：
    - B: batch size (批大小)
    - N: 点数 (如2048)
    - 3: XYZ坐标维度
    
    增强包含三个步骤：
    1. 随机绕Z轴旋转 (0-360度)
    2. 随机平移 (±0.1米范围)
    3. 添加高斯噪声 (标准差0.01)
    
    参数:
        points (torch.Tensor): 输入点云数据，形状应为 [B, N, 3]
        
    返回:
        torch.Tensor: 增强后的点云数据，形状与输入相同
        
    示例:
        >>> # 创建模拟点云数据 (batch_size=4, 点数=2048)
        >>> points = torch.randn(4, 2048, 3)
        >>> 
        >>> # 应用增强
        >>> augmented_points = batch_point_cloud_augmentation(points)
        >>> 
        >>> # 检查输出形状
        >>> print(augmented_points.shape)  # 输出: torch.Size([4, 2048, 3])
        
    增强效果可视化:
        原始点云       -> 旋转后       -> 平移后       -> 添加噪声后
        [● 整齐排列]   [↻ 旋转角度]   [⇳ 位置偏移]   [~ 点位置扰动]
    """
    # 验证输入形状
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError("输入张量形状应为 [B, N, 3], 实际形状: {}".format(points.shape))
    
    # 1. 随机旋转 (绕Z轴)
    # 生成每个batch的随机旋转角度 (0 ~ 2π)
    angle = torch.rand(points.size(0), device=points.device) * 2 * np.pi
    
    # 构造旋转矩阵 (仅影响XY平面)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rot_mat = torch.stack([cos_a, -sin_a, 
                          sin_a, cos_a], dim=1).view(-1, 2, 2)
    
    # 应用旋转 (仅对XY坐标)
    xy_rotated = torch.matmul(points[..., :2], rot_mat)
    points_rot = torch.cat([xy_rotated, points[..., 2:]], dim=-1)
    
    # 2. 随机平移 (±0.1米)
    # 为每个batch生成独立偏移量
    shift = (torch.rand(points.size(0), 1, 3, device=points.device) * 0.2 - 0.1)
    points_shift = points_rot + shift
    
    # 3. 添加高斯噪声
    # 标准差0.01 (约99.7%的点偏移小于0.03)
    noise = torch.randn_like(points_shift) * 0.01
    augmented_points = points_shift + noise
    
    return augmented_points


# 示例使用代码
if __name__ == "__main__":
    # 创建模拟点云数据 (batch_size=2, 点数=2048)
    test_points = torch.rand(2, 2048, 3) * 2 - 1  # 生成(-1,1)范围内的点
    
    print("原始点云示例:")
    print(f"形状: {test_points.shape}")
    print(f"第一个点: {test_points[0, 0]}")
    print(f"范围: X({test_points[...,0].min():.3f}-{test_points[...,0].max():.3f}) "
          f"Y({test_points[...,1].min():.3f}-{test_points[...,1].max():.3f}) "
          f"Z({test_points[...,2].min():.3f}-{test_points[...,2].max():.3f})")
    
    # 应用增强
    augmented = batch_point_cloud_augmentation(test_points)
    
    print("\n增强后点云示例:")
    print(f"形状: {augmented.shape}")
    print(f"第一个点: {augmented[0, 0]}")
    print(f"范围: X({augmented[...,0].min():.3f}-{augmented[...,0].max():.3f}) "
          f"Y({augmented[...,1].min():.3f}-{augmented[...,1].max():.3f}) "
          f"Z({augmented[...,2].min():.3f}-{augmented[...,2].max():.3f})")
    
    # 计算变换量
    displacement = (augmented - test_points).norm(dim=-1)
    print(f"\n平均点位移: {displacement.mean():.5f} ± {displacement.std():.5f} 米")