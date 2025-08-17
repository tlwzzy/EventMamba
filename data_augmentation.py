import torch
import numpy as np

def random_scale(data, scale_low=0.8, scale_high=1.25):
    """
    对一个批次的数据进行随机缩放。
    只缩放 x, y 坐标 (维度 1 和 2)，时间戳 (维度 0) 保持不变。
    """
    B, N, C = data.shape
    scales = torch.FloatTensor(B).uniform_(scale_low, scale_high).to(data.device)
    for batch_index in range(B):
        data[batch_index, :, 1:3] *= scales[batch_index]
    return data

def random_shift(data, shift_range=0.1):
    """
    对一个批次的数据进行随机平移。
    只平移 x, y 坐标 (维度 1 和 2)，时间戳 (维度 0) 保持不变。
    """
    B, N, C = data.shape
    shifts = torch.FloatTensor(B, 1, 2).uniform_(-shift_range, shift_range).to(data.device)
    data[:, :, 1:3] += shifts
    return data

# ==============================================================================
# ======================== START: 核心修改 =====================================
# ==============================================================================
def random_dropout(data, dropout_ratio=0.2):
    """
    在一个批次的数据中随机丢弃一部分点。
    通过将这些点复制为第一个点来实现，模拟信息丢失。
    """
    B, N, C = data.shape
    for batch_index in range(B):
        dropout_indices = np.random.choice(N, int(N * dropout_ratio), replace=False)
        # (核心修改) 对源张量进行 .clone()
        data[batch_index, dropout_indices, :] = data[batch_index, 0, :].clone() 
    return data
# ==============================================================================
# ========================= END: 核心修改 ======================================
# ==============================================================================

def augment_point_cloud(data, do_scale=True, do_shift=True, do_dropout=True):
    """
    对一个批次的数据随机应用增强。
    """
    if do_scale and np.random.random() > 0.5:
        data = random_scale(data)
    if do_shift and np.random.random() > 0.5:
        data = random_shift(data)
    if do_dropout and np.random.random() > 0.5:
        data = random_dropout(data, dropout_ratio=np.random.uniform(0.1, 0.3))
    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("--- 正在测试数据增强模块 ---")

    num_points = 1024
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = 0.5 * np.cos(angles) + 0.5
    y = 0.2 * np.sin(angles) + 0.5
    x += np.random.normal(0, 0.02, num_points)
    y += np.random.normal(0, 0.02, num_points)

    sample_point_cloud = np.hstack([t, x.reshape(-1, 1), y.reshape(-1, 1)])
    
    batch_data = np.stack([sample_point_cloud, sample_point_cloud], axis=0)
    batch_tensor = torch.from_numpy(batch_data).float()

    original_tensor = batch_tensor.clone()
    augmented_tensor = augment_point_cloud(batch_tensor.clone())

    print(f"原始Tensor形状: {original_tensor.shape}")
    print(f"增强后Tensor形状: {augmented_tensor.shape}")
    
    print(f"\n原始数据[0] x 范围: {original_tensor[0, :, 1].min():.2f} -> {original_tensor[0, :, 1].max():.2f}")
    print(f"增强后数据[0] x 范围: {augmented_tensor[0, :, 1].min():.2f} -> {augmented_tensor[0, :, 1].max():.2f}")
    
    original_sample = original_tensor[0].numpy()
    augmented_sample = augmented_tensor[0].numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(original_sample[:, 1], original_sample[:, 2], c=original_sample[:, 0], cmap='viridis', s=5)
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2.scatter(augmented_sample[:, 1], augmented_sample[:, 2], c=augmented_sample[:, 0], cmap='viridis', s=5)
    ax2.set_title("Augmented Point Cloud")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.suptitle("数据增强效果对比 (颜色代表时间)")
    plt.tight_layout()
    print("\n正在显示可视化结果... 关闭图像窗口以继续。")
    plt.show()

    print("\n--- 测试完成 ---")