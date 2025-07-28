# EventMamba Hydra配置使用指南

本项目已将参数管理从传统的argparse迁移到Hydra + OmegaConf，使用YAML配置文件来管理所有训练参数。

## 文件结构

```
EventMamba-main/
├── configs/
│   └── config.yaml          # 主配置文件
├── train_classification.py  # 重构后的训练脚本
├── test_config.py           # 配置测试脚本
└── README_hydra_config.md   # 本文档
```

## 配置文件说明

### 主配置文件 (`configs/config.yaml`)

配置文件包含以下几个主要部分：

1. **设备配置** (`device`)
   - `use_cpu`: 是否使用CPU模式
   - `gpu`: 指定GPU设备

2. **数据配置** (`data`)
   - `data_path`: 数据集路径
   - `num_category`: 分类类别数
   - `num_point`: 点云点数

3. **训练配置** (`training`)
   - `batch_size`: 批次大小
   - `epoch`: 训练轮数
   - `learning_rate`: 学习率
   - `optimizer`: 优化器类型
   - `decay_rate`: 权重衰减率
   - `augment`: 是否启用数据增强

4. **模型配置** (`model`)
   - `bignet`: 是否使用大网络

5. **日志配置** (`logging`)
   - `log_path`: TensorBoard日志路径
   - `log_name`: 日志名称
   - `log_dir`: 日志目录

## 使用方法

### 1. 基本使用

```bash
# 使用默认配置运行训练
python train_classification.py
```

### 2. 命令行覆盖参数

```bash
# 修改批次大小
python train_classification.py training.batch_size=64

# 修改学习率和优化器
python train_classification.py training.learning_rate=0.01 training.optimizer=SGD

# 启用数据增强
python train_classification.py training.augment=true

# 使用大网络
python train_classification.py model.bignet=true

# 修改数据集路径
python train_classification.py data.data_path=/path/to/your/dataset/
```

### 3. 多参数组合

```bash
# 同时修改多个参数
python train_classification.py \
    training.batch_size=64 \
    training.learning_rate=0.01 \
    training.optimizer=SGD \
    training.augment=true \
    model.bignet=true
```

### 4. 使用不同数据集

```bash
# 切换到THU数据集
python train_classification.py \
    data.data_path=/root/autodl-tmp/EventMamba-main/data/THU/dataset/ \
    data.num_category=50 \
    logging.log_name=/thu/
```

## 配置测试

运行配置测试脚本来验证配置是否正确：

```bash
python test_config.py
```

## Hydra输出管理

Hydra会自动创建输出目录来保存运行结果：

```
outputs/
├── 2024-01-01/           # 按日期组织
│   └── 12-34-56/         # 按时间组织
│       ├── .hydra/       # Hydra配置信息
│       └── ...           # 其他输出文件
```

## 优势

1. **配置集中管理**: 所有参数都在YAML文件中，便于管理和版本控制
2. **类型安全**: OmegaConf提供类型检查和验证
3. **命令行覆盖**: 可以通过命令行轻松覆盖配置参数
4. **实验管理**: Hydra自动管理实验输出和配置
5. **可读性强**: YAML格式比命令行参数更易读和维护

## 迁移对比

### 原来的使用方式 (argparse)
```bash
python train_classification.py \
    --batch_size 64 \
    --learning_rate 0.01 \
    --optimizer SGD \
    --augment \
    --bignet
```

### 现在的使用方式 (Hydra)
```bash
python train_classification.py \
    training.batch_size=64 \
    training.learning_rate=0.01 \
    training.optimizer=SGD \
    training.augment=true \
    model.bignet=true
```

## 注意事项

1. 确保安装了必要的依赖：
   ```bash
   pip install hydra-core omegaconf
   ```

2. 配置文件路径是相对于脚本的，确保`configs/`目录在正确位置

3. 布尔值在命令行中使用`true/false`，不是`True/False`

4. 字符串值如果包含特殊字符，需要用引号包围