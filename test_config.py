#!/usr/bin/env python3
"""
测试Hydra配置文件是否正常工作
"""

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config")
def test_config(cfg: DictConfig):
    print("=" * 50)
    print("Hydra配置测试")
    print("=" * 50)
    
    # 打印完整配置
    print("完整配置:")
    print(OmegaConf.to_yaml(cfg))
    
    print("\n" + "=" * 50)
    print("关键参数验证:")
    print("=" * 50)
    
    # 验证关键参数
    print(f"设备配置:")
    print(f"  - use_cpu: {cfg.device.use_cpu}")
    print(f"  - gpu: {cfg.device.gpu}")
    
    print(f"\n数据配置:")
    print(f"  - data_path: {cfg.data.data_path}")
    print(f"  - num_category: {cfg.data.num_category}")
    print(f"  - num_point: {cfg.data.num_point}")
    
    print(f"\n训练配置:")
    print(f"  - batch_size: {cfg.training.batch_size}")
    print(f"  - epoch: {cfg.training.epoch}")
    print(f"  - learning_rate: {cfg.training.learning_rate}")
    print(f"  - optimizer: {cfg.training.optimizer}")
    print(f"  - augment: {cfg.training.augment}")
    
    print(f"\n模型配置:")
    print(f"  - bignet: {cfg.model.bignet}")
    
    print(f"\n日志配置:")
    print(f"  - log_path: {cfg.logging.log_path}")
    print(f"  - log_name: {cfg.logging.log_name}")
    
    print("\n" + "=" * 50)
    print("配置测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    test_config()