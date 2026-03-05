"""
BasePolicy — 所有策略模型的统一抽象接口

所有自定义策略（ACT, VITA, 等）都继承此基类，实现统一的
训练 (compute_loss)、推理 (select_action)、保存/加载 接口。
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BasePolicy(ABC, nn.Module):
    """所有策略模型的统一接口。"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """添加模型特有的命令行参数。子类可覆盖。"""
        pass

    @abstractmethod
    def build(self, args, dataset_stats: dict,
              input_features: dict, output_features: dict) -> None:
        """构建模型内部组件并移动到 device。

        Args:
            args: 命令行参数 (含 device, lr 等通用参数 + 模型特有参数)
            dataset_stats: 数据集统计信息 (均值/标准差, 用于归一化)
            input_features: 输入特征描述 {key: FeatureInfo}
            output_features: 输出特征描述 {key: FeatureInfo}
        """
        pass

    @abstractmethod
    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """前向传播 + 计算训练 loss。

        Args:
            batch: 数据字典 {key: Tensor}，已在 device 上

        Returns:
            (loss, metrics_dict) — loss 用于 backward, metrics_dict 用于日志
        """
        pass

    @abstractmethod
    def select_action(self, batch: dict) -> torch.Tensor:
        """推理: 给定当前观测，返回 action。

        Args:
            batch: 观测字典 {key: Tensor}，batch_size=1

        Returns:
            action tensor, shape (1, action_dim)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置内部状态 (action queue 等)。每个 episode 开始时调用。"""
        pass

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """返回优化器实例。"""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """保存 checkpoint 到指定目录。"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path, dataset_stats: dict, device: torch.device) -> "BasePolicy":
        """从 checkpoint 加载模型。

        Args:
            path: checkpoint 目录
            dataset_stats: 数据集统计信息
            device: 目标设备

        Returns:
            加载好的策略实例
        """
        pass

    def get_delta_timestamps(self, fps: float) -> dict:
        """返回数据集加载所需的 delta_timestamps 配置。

        子类可覆盖以定义不同的 observation/action 时间窗口。
        默认只返回 action chunk。

        Args:
            fps: 数据集帧率

        Returns:
            {key: [timestamp_offsets]} 字典
        """
        chunk_size = getattr(self, 'chunk_size', 50)
        return {
            "action": [i / fps for i in range(chunk_size)],
        }
