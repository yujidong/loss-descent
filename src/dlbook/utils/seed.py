"""随机种子工具：全书所有实验的可复现随机性都从这里出发。"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """统一设置 Python / NumPy（及已安装的 PyTorch）的随机种子。

    每章可执行笔记本的第一格固定调用 ``set_seed(42)``，
    保证读者运行结果与书中图表一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 是可选依赖，卷二起才会安装
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
