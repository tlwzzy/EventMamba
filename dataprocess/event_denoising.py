import numpy as np
import cv2
import torch
import random
from typing import Optional, Tuple

class EventDenoiser:
    """事件相机事件流去噪器"""
    
    def __init__(self, method='temporal_filter', **kwargs):
        """
        初始化去噪器
        Args:
            method: 去噪方法 ('temporal_filter', 'spatial_filter', 'snn', 'combined')
            **kwargs: 方法特定参数
        """
        self.method = method
        self.kwargs = kwargs
        
    def denoise(self, events: np.ndarray, w: int = 128, h: int = 128) -> np.ndarray:
        """
        对事件进行去噪
        Args:
            events: 事件数组 [N, 3] (t, x, y) 或 [N, 4] (t, x, y, p)
            w, h: 图像宽度和高度
        Returns:
            denoised_events: 去噪后的事件
        """
        if self.method == 'temporal_filter':
            return self._temporal_filter(events)
        elif self.method == 'spatial_filter':
            return self._spatial_filter(events, w, h)
        elif self.method == 'snn':
            return self._snn_denoise(events, w, h)
        elif self.method == 'combined':
            return self._combined_denoise(events, w, h)
        else:
            raise ValueError(f"Unknown denoising method: {self.method}")
    
    def _temporal_filter(self, events: np.ndarray, 
                        min_events: int = 50, 
                        max_time_gap: int = 1000) -> np.ndarray:
        """
        时间域过滤，去除时间间隔过大的事件
        """
        if len(events) < min_events:
            return events
        
        # 按时间排序
        sorted_indices = np.argsort(events[:, 0])
        events = events[sorted_indices]
        
        # 计算时间间隔
        time_diffs = np.diff(events[:, 0])
        
        # 找到时间间隔过大的位置
        large_gaps = np.where(time_diffs > max_time_gap)[0]
        
        if len(large_gaps) == 0:
            return events
        
        # 保留最大连续段
        max_start = 0
        max_length = 0
        
        start = 0
        for gap_idx in large_gaps:
            length = gap_idx - start
            if length > max_length:
                max_length = length
                max_start = start
            start = gap_idx + 1
        
        # 检查最后一段
        length = len(events) - start
        if length > max_length:
            max_length = length
            max_start = start
        
        return events[max_start:max_start + max_length]
    
    def _spatial_filter(self, events: np.ndarray, w: int, h: int,
                       min_neighbors: int = 3, 
                       radius: float = 2.0) -> np.ndarray:
        """
        空间域过滤，去除孤立的噪声事件
        """
        if len(events) < 10:
            return events
        
        # 创建空间网格
        grid = np.zeros((h, w), dtype=np.int32)
        
        # 将事件映射到网格
        for event in events:
            x, y = int(event[1]), int(event[2])
            if 0 <= x < w and 0 <= y < h:
                grid[y, x] += 1
        
        # 计算每个位置的邻居数量
        kernel = np.ones((3, 3), dtype=np.int32)
        neighbor_count = cv2.filter2D(grid, -1, kernel)
        
        # 过滤孤立事件
        filtered_events = []
        for event in events:
            x, y = int(event[1]), int(event[2])
            if 0 <= x < w and 0 <= y < h:
                if neighbor_count[y, x] >= min_neighbors:
                    filtered_events.append(event)
        
        return np.array(filtered_events) if filtered_events else events
    
    def _snn_denoise(self, events: np.ndarray, w: int, h: int,
                     threshold: float = 1.2,
                     decay: float = 0.02,
                     margin: int = 3,
                     spike_val: float = 1.0) -> np.ndarray:
        """
        基于脉冲神经网络的去噪
        """
        if len(events) < 10:
            return events
        
        # 初始化SNN网络
        network = np.zeros((h, w), dtype=np.float64)
        timenet = np.zeros((h, w), dtype=np.int64)
        firing = np.zeros((h, w), dtype=np.int64)
        
        # 按时间排序事件
        sorted_indices = np.argsort(events[:, 0])
        events = events[sorted_indices]
        
        # 初始化时间网络
        if len(events) > 0:
            timenet[:] = events[0, 0]
        
        denoised_events = []
        
        for event in events:
            t, x, y = event[0], int(event[1]), int(event[2])
            
            if 0 <= x < w and 0 <= y < h:
                # 计算时间衰减
                escape_time = (t - timenet[y, x]) / 1000.0
                residual = max(network[y, x] - decay * escape_time, 0)
                
                # 更新膜电位
                network[y, x] = residual + spike_val
                timenet[y, x] = t
                
                # 检查是否触发
                if network[y, x] > threshold:
                    firing[y, x] += 1
                    denoised_events.append(event)
                    
                    # 侧抑制：清除周围区域
                    for i in range(-margin, margin + 1):
                        for j in range(-margin, margin + 1):
                            ny, nx = y + i, x + j
                            if 0 <= nx < w and 0 <= ny < h:
                                network[ny, nx] = 0.0
        
        return np.array(denoised_events) if denoised_events else events
    
    def _combined_denoise(self, events: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        组合多种去噪方法
        """
        # 1. 时间域过滤
        events = self._temporal_filter(events)
        
        # 2. 空间域过滤
        events = self._spatial_filter(events, w, h)
        
        # 3. SNN去噪
        events = self._snn_denoise(events, w, h)
        
        return events

def denoise_events_simple(events: np.ndarray, 
                         min_events: int = 50,
                         max_time_gap: int = 1000) -> np.ndarray:
    """
    简单的事件去噪函数
    Args:
        events: 事件数组 [N, 3] (t, x, y)
        min_events: 最小事件数
        max_time_gap: 最大时间间隔
    Returns:
        denoised_events: 去噪后的事件
    """
    if len(events) < min_events:
        return events
    
    # 按时间排序
    sorted_indices = np.argsort(events[:, 0])
    events = events[sorted_indices]
    
    # 计算时间间隔
    time_diffs = np.diff(events[:, 0])
    
    # 找到时间间隔过大的位置
    large_gaps = np.where(time_diffs > max_time_gap)[0]
    
    if len(large_gaps) == 0:
        return events
    
    # 保留最大连续段
    max_start = 0
    max_length = 0
    
    start = 0
    for gap_idx in large_gaps:
        length = gap_idx - start
        if length > max_length:
            max_length = length
            max_start = start
        start = gap_idx + 1
    
    # 检查最后一段
    length = len(events) - start
    if length > max_length:
        max_length = length
        max_start = start
    
    return events[max_start:max_start + max_length]

def remove_outliers(events: np.ndarray, 
                   w: int = 128, 
                   h: int = 128,
                   min_neighbors: int = 3) -> np.ndarray:
    """
    去除空间离群点
    Args:
        events: 事件数组 [N, 3] (t, x, y)
        w, h: 图像宽度和高度
        min_neighbors: 最小邻居数
    Returns:
        filtered_events: 过滤后的事件
    """
    if len(events) < 10:
        return events
    
    # 创建空间网格
    grid = np.zeros((h, w), dtype=np.int32)
    
    # 将事件映射到网格
    for event in events:
        x, y = int(event[1]), int(event[2])
        if 0 <= x < w and 0 <= y < h:
            grid[y, x] += 1
    
    # 计算每个位置的邻居数量
    kernel = np.ones((3, 3), dtype=np.int32)
    neighbor_count = cv2.filter2D(grid, -1, kernel)
    
    # 过滤孤立事件
    filtered_events = []
    for event in events:
        x, y = int(event[1]), int(event[2])
        if 0 <= x < w and 0 <= y < h:
            if neighbor_count[y, x] >= min_neighbors:
                filtered_events.append(event)
    
    return np.array(filtered_events) if filtered_events else events 