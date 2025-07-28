import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import random
from event_denoising import EventDenoiser, denoise_events_simple, remove_outliers

def shuffle_downsample(data,num=None):
    ''' data is a numpy array '''
    if num == None:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
    elif num > data.shape[0]:
        idx = np.random.choice(np.arange(data.shape[0]), size=num, replace=True)
        idx.sort()
    else:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        idx = idx[0:num]
        idx.sort()
    return data[idx,...]

def normaliztion(orinal_events,w,h,process_p = False):
    """
    Normalize events.
    """
    events = orinal_events.copy()
    events = events.astype('float32')
    events[:, 0] = (events[:, 0] - events[:, 0].min(axis=0)) / (events[:, 0].max(axis=0) - events[:, 0].min(axis=0)+1e-6)
    events[:, 1] = events[:, 1] / w
    events[:, 2] = events[:, 2] / h
    if process_p:
        events[:, 3] = events[:, 3]*2-1
    return events

def process_split(split_dir, export_path, num_points=4096, max_classes=100, enable_denoising=True, denoise_method='combined'):
    data, labels = [], []
    class_names = sorted(os.listdir(split_dir))
    if max_classes is not None:
        class_names = class_names[:max_classes]
    print('类别顺序:', class_names)
    
    # 初始化去噪器
    if enable_denoising:
        denoiser = EventDenoiser(method=denoise_method)
    
    for class_idx, class_name in enumerate(tqdm(class_names, desc="类别进度")):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        file_list = os.listdir(class_dir)
        for file_idx, file_name in enumerate(tqdm(file_list, desc=f"{class_name}文件进度", leave=False)):
            file_path = os.path.join(class_dir, file_name)
            events = np.load(file_path)
            t, x, y = events['t'], events['x'], events['y']
            window_events = np.stack([t, x, y], axis=1)
            
            if window_events.shape[0] > 100:
                # 应用去噪
                if enable_denoising:
                    if denoise_method == 'simple':
                        # 简单时间过滤
                        window_events = denoise_events_simple(window_events, min_events=50, max_time_gap=1000)
                        # 空间离群点去除
                        # if len(window_events) > 10:
                        #     window_events = remove_outliers(window_events, w=128, h=128, min_neighbors=3)
                    else:
                        # 使用完整的去噪器
                        window_events = denoiser.denoise(window_events, w=128, h=128)
                
                if window_events.shape[0] > 100:  # 再次检查事件数量
                    extracted = shuffle_downsample(window_events, num_points)
                    if extracted.shape[0] == num_points:
                        events_normed = normaliztion(extracted, 128, 128, False)
                        # print(f'样本: {file_name}, 路径: {file_path}, 类别: {class_name}, 标签: {class_idx}')
                        data.append(events_normed)
                        labels.append(class_idx)
    data = np.array(data)
    labels = np.array(labels)
    print(f"data: {len(data)}, labels: {len(labels)}")
    with h5py.File(export_path, 'w') as hf:
        hf.create_dataset('data', data=data, maxshape=(None, num_points, 3), chunks=True, dtype='float32')
        hf.create_dataset('label', data=labels, maxshape=(None,), chunks=True, dtype='int16')

if __name__ == "__main__":
    # 处理train和test，启用去噪
    # 可以选择不同的去噪方法：
    # 'simple': 简单的时间过滤 + 空间离群点去除
    # 'temporal_filter': 时间域过滤
    # 'spatial_filter': 空间域过滤  
    # 'snn': 脉冲神经网络去噪
    # 'combined': 组合多种方法
    process_split('data/DVS-Lip/train', 'data/DVS-Lip/train.h5', enable_denoising=True, denoise_method='snn', num_points=8196)
    process_split('data/DVS-Lip/test', 'data/DVS-Lip/test.h5', enable_denoising=True, denoise_method='snn', num_points=8196)