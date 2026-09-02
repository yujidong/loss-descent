"""随机种子工具：全书所有实验的可复现随机性都从这里出发。"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """统一设置 Python / NumPy（及已安装的 PyTorch）的随机种子。

    每章可执行笔记本的第一格固定调用 ``set_seed(42)``，
    保证读者运行结果与书中图表一致。

    注意：这里设置的是 legacy 全局 RNG（np.random.*）。dlbook 内部
    各模块统一走 ``np.random.default_rng(seed)`` 并以显式 seed 参数
    传入，不受本函数影响——读者自己用 default_rng() 做的实验，
    请自行固定种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 是可选依赖，卷二起才会安装
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
