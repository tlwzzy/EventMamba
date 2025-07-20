#!/usr/bin/env python3
"""
事件相机事件流去噪测试脚本
演示不同去噪方法的效果
"""

import numpy as np
import matplotlib.pyplot as plt
from event_denoising import EventDenoiser, denoise_events_simple, remove_outliers

def generate_synthetic_events(num_events=1000, noise_ratio=0.3):
    """
    生成合成的事件数据用于测试
    Args:
        num_events: 事件总数
        noise_ratio: 噪声比例
    Returns:
        events: 合成事件数据 [N, 3] (t, x, y)
    """
    # 生成有效事件（模拟运动轨迹）
    valid_events = int(num_events * (1 - noise_ratio))
    t_valid = np.linspace(0, 1000, valid_events)
    x_valid = 64 + 32 * np.sin(t_valid / 100) + np.random.normal(0, 2, valid_events)
    y_valid = 64 + 32 * np.cos(t_valid / 100) + np.random.normal(0, 2, valid_events)
    
    # 生成噪声事件
    noise_events = num_events - valid_events
    t_noise = np.random.uniform(0, 1000, noise_events)
    x_noise = np.random.uniform(0, 128, noise_events)
    y_noise = np.random.uniform(0, 128, noise_events)
    
    # 合并事件
    events = np.vstack([
        np.column_stack([t_valid, x_valid, y_valid]),
        np.column_stack([t_noise, x_noise, y_noise])
    ])
    
    # 随机打乱
    np.random.shuffle(events)
    
    return events

def visualize_events(events, title="Events", save_path=None):
    """
    可视化事件数据
    Args:
        events: 事件数据 [N, 3] (t, x, y)
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 时间-空间图
    plt.subplot(1, 3, 1)
    plt.scatter(events[:, 0], events[:, 1], c=events[:, 0], cmap='viridis', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.title(f'{title} - Time vs X')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.scatter(events[:, 0], events[:, 2], c=events[:, 0], cmap='viridis', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.title(f'{title} - Time vs Y')
    plt.colorbar()
    
    # 空间分布图
    plt.subplot(1, 3, 3)
    plt.scatter(events[:, 1], events[:, 2], c=events[:, 0], cmap='viridis', alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{title} - Spatial Distribution')
    plt.colorbar()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def test_denoising_methods():
    """
    测试不同的去噪方法
    """
    print("生成合成事件数据...")
    events = generate_synthetic_events(num_events=1000, noise_ratio=0.3)
    print(f"原始事件数量: {len(events)}")
    
    # 可视化原始数据
    visualize_events(events, "Original Events", "original_events.png")
    
    # 测试不同的去噪方法
    methods = {
        'Simple': lambda e: denoise_events_simple(e, min_events=50, max_time_gap=100),
        'Spatial Filter': lambda e: remove_outliers(e, w=128, h=128, min_neighbors=3),
        'Temporal Filter': lambda e: EventDenoiser(method='temporal_filter').denoise(e, w=128, h=128),
        'SNN': lambda e: EventDenoiser(method='snn').denoise(e, w=128, h=128),
        'Combined': lambda e: EventDenoiser(method='combined').denoise(e, w=128, h=128)
    }
    
    results = {}
    for method_name, denoise_func in methods.items():
        print(f"\n测试 {method_name} 去噪方法...")
        try:
            denoised_events = denoise_func(events.copy())
            results[method_name] = denoised_events
            print(f"{method_name} 去噪后事件数量: {len(denoised_events)}")
            print(f"去噪率: {100 * (1 - len(denoised_events) / len(events)):.1f}%")
            
            # 可视化去噪结果
            visualize_events(denoised_events, f"{method_name} Denoised", f"{method_name.lower().replace(' ', '_')}_denoised.png")
            
        except Exception as e:
            print(f"{method_name} 去噪失败: {e}")
            results[method_name] = events
    
    return results

def compare_denoising_performance():
    """
    比较不同去噪方法的性能
    """
    print("\n=== 去噪方法性能比较 ===")
    
    # 生成不同噪声水平的数据
    noise_levels = [0.1, 0.3, 0.5, 0.7]
    methods = ['Simple', 'Spatial Filter', 'Temporal Filter', 'SNN', 'Combined']
    
    results = {}
    for noise_level in noise_levels:
        print(f"\n噪声水平: {noise_level:.1f}")
        events = generate_synthetic_events(num_events=1000, noise_ratio=noise_level)
        
        for method in methods:
            try:
                if method == 'Simple':
                    denoised = denoise_events_simple(events.copy())
                elif method == 'Spatial Filter':
                    denoised = remove_outliers(events.copy())
                else:
                    denoiser = EventDenoiser(method=method.lower().replace(' ', '_'))
                    denoised = denoiser.denoise(events.copy())
                
                denoising_ratio = 1 - len(denoised) / len(events)
                print(f"  {method}: 去噪率 {denoising_ratio:.2f}")
                
            except Exception as e:
                print(f"  {method}: 失败 - {e}")

if __name__ == "__main__":
    print("事件相机事件流去噪测试")
    print("=" * 50)
    
    # 测试去噪方法
    results = test_denoising_methods()
    
    # 比较性能
    compare_denoising_performance()
    
    print("\n测试完成！")
    print("生成的可视化图像已保存到当前目录。") 