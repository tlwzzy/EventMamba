# 事件相机事件流去噪方法

本项目提供了多种事件相机事件流去噪方法，用于提高事件数据的质量和后续处理的准确性。

## 去噪方法概述

### 1. 时间域过滤 (Temporal Filter)
- **原理**: 基于事件时间间隔进行过滤，去除时间上不连续的事件
- **适用场景**: 去除时间间隔过大的噪声事件
- **参数**: 
  - `min_events`: 最小事件数 (默认: 50)
  - `max_time_gap`: 最大时间间隔 (默认: 1000)

### 2. 空间域过滤 (Spatial Filter)
- **原理**: 基于空间邻域关系过滤孤立事件
- **适用场景**: 去除空间上孤立的噪声事件
- **参数**:
  - `min_neighbors`: 最小邻居数 (默认: 3)
  - `radius`: 搜索半径 (默认: 2.0)

### 3. 脉冲神经网络去噪 (SNN Denoising)
- **原理**: 模拟生物神经元的脉冲机制，通过阈值触发和侧抑制去除噪声
- **适用场景**: 复杂噪声环境下的高级去噪
- **参数**:
  - `threshold`: 神经元触发阈值 (默认: 1.2)
  - `decay`: 膜电位衰减系数 (默认: 0.02)
  - `margin`: 侧抑制范围 (默认: 3)
  - `spike_val`: 每次事件的膜电位增量 (默认: 1.0)

### 4. 组合去噪 (Combined)
- **原理**: 结合多种去噪方法的优势
- **流程**: 时间域过滤 → 空间域过滤 → SNN去噪

## 使用方法

### 基本使用

```python
from event_denoising import EventDenoiser, denoise_events_simple, remove_outliers

# 方法1: 使用去噪器类
denoiser = EventDenoiser(method='combined')
denoised_events = denoiser.denoise(events, w=128, h=128)

# 方法2: 使用简单函数
denoised_events = denoise_events_simple(events, min_events=50, max_time_gap=1000)
denoised_events = remove_outliers(denoised_events, w=128, h=128, min_neighbors=3)
```

### 在数据处理中使用

```python
# 在 generate_dvslip.py 中使用
from event_denoising import EventDenoiser

def process_split(split_dir, export_path, num_points=4096, max_classes=70, 
                 enable_denoising=True, denoise_method='combined'):
    # 初始化去噪器
    if enable_denoising:
        denoiser = EventDenoiser(method=denoise_method)
    
    # 处理事件数据
    for events in event_streams:
        if enable_denoising:
            events = denoiser.denoise(events, w=128, h=128)
        # 继续处理...
```

### 不同去噪方法的选择

```python
# 简单快速去噪
process_split(..., denoise_method='simple')

# 时间域过滤
process_split(..., denoise_method='temporal_filter')

# 空间域过滤
process_split(..., denoise_method='spatial_filter')

# SNN高级去噪
process_split(..., denoise_method='snn')

# 组合多种方法
process_split(..., denoise_method='combined')
```

## 参数调优建议

### 根据数据特征调整参数

1. **高噪声数据**:
   ```python
   # 使用更严格的参数
   denoiser = EventDenoiser(method='combined')
   denoised = denoiser.denoise(events, w=128, h=128)
   ```

2. **低噪声数据**:
   ```python
   # 使用简单方法
   denoised = denoise_events_simple(events, min_events=30, max_time_gap=500)
   ```

3. **实时处理**:
   ```python
   # 使用快速的空间过滤
   denoised = remove_outliers(events, min_neighbors=2)
   ```

### 性能优化

- **计算效率**: `simple` > `spatial_filter` > `temporal_filter` > `snn` > `combined`
- **去噪效果**: `combined` > `snn` > `temporal_filter` > `spatial_filter` > `simple`

## 测试和验证

### 运行测试脚本

```bash
cd dataprocess
python test_denoising.py
```

### 测试内容

1. **合成数据测试**: 生成包含噪声的合成事件数据
2. **可视化比较**: 生成去噪前后的对比图
3. **性能评估**: 比较不同方法的去噪率和计算时间

### 评估指标

- **去噪率**: 去除的事件比例
- **保留率**: 保留的有效事件比例
- **计算时间**: 处理速度
- **内存使用**: 内存消耗

## 实际应用案例

### 1. DVS-Lip数据集处理

```python
# 在 generate_dvslip.py 中
process_split('data/DVS-Lip/train', 'data/DVS-Lip/train.h5', 
             enable_denoising=True, denoise_method='simple')
```

### 2. 实时事件流处理

```python
# 实时处理事件流
denoiser = EventDenoiser(method='temporal_filter')
for event_batch in event_stream:
    denoised_batch = denoiser.denoise(event_batch)
    # 继续处理...
```

### 3. 离线数据预处理

```python
# 批量处理历史数据
denoiser = EventDenoiser(method='combined')
for file_path in data_files:
    events = load_events(file_path)
    denoised_events = denoiser.denoise(events)
    save_events(denoised_events, output_path)
```

## 注意事项

1. **数据格式**: 确保输入事件数据格式为 `[N, 3]` (t, x, y) 或 `[N, 4]` (t, x, y, p)
2. **坐标范围**: 确保事件坐标在图像范围内 (0 ≤ x < w, 0 ≤ y < h)
3. **时间戳**: 时间戳应该是单调递增的
4. **内存使用**: 对于大数据集，考虑分批处理

## 故障排除

### 常见问题

1. **导入错误**: 确保安装了所需的依赖包
   ```bash
   pip install numpy opencv-python torch matplotlib
   ```

2. **内存不足**: 对于大文件，使用分批处理
   ```python
   # 分批处理
   batch_size = 10000
   for i in range(0, len(events), batch_size):
       batch = events[i:i+batch_size]
       denoised_batch = denoiser.denoise(batch)
   ```

3. **参数调优**: 根据具体数据特征调整参数
   ```python
   # 调整参数
   denoiser = EventDenoiser(method='snn', threshold=1.5, decay=0.01)
   ```

## 扩展和定制

### 添加新的去噪方法

```python
class CustomDenoiser(EventDenoiser):
    def _custom_denoise(self, events, w, h):
        # 实现自定义去噪逻辑
        return denoised_events
```

### 集成到现有流程

```python
# 在现有的数据处理流程中添加去噪步骤
def process_events(events, enable_denoising=True):
    if enable_denoising:
        denoiser = EventDenoiser(method='combined')
        events = denoiser.denoise(events)
    return events
```

## 参考文献

1. SNN去噪方法参考了 `dataprocess/generate_action.py` 中的实现
2. 点云去噪方法参考了 `dataprocess/generate_thu.py` 中的实现
3. 时间域过滤方法基于事件时间连续性原理

## 更新日志

- v1.0: 初始版本，包含基本的去噪方法
- v1.1: 添加SNN去噪和组合方法
- v1.2: 优化性能和添加测试脚本 