"""随书实验的玩具数据集。"""
from __future__ import annotations

import numpy as np

# XOR 四点，标签取 ±1（与感知机/MLP 的符号输出同域）
XOR = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([-1.0, 1.0, 1.0, -1.0])


def make_blobs(n_per_class: int = 30, distance: float = 2.0, noise: float = 0.25, seed: int = 0):
    """两个高斯团簇，中心分别在 (-d/2,-d/2) 与 (+d/2,+d/2)。

    用于展示可分数据上的收敛与决策边界。
    """
    rng = np.random.default_rng(seed)
    c = distance / 2.0
    centers = np.array([[-c, -c], [c, c]])
    X = np.vstack(
        [rng.normal(center, noise, size=(n_per_class, 2)) for center in centers]
    )
    y = np.array([-1.0] * n_per_class + [1.0] * n_per_class)
    return X, y


def make_slab(n_per_class: int = 20, margin: float = 0.5, extent: float = 5.0, noise: float = 0.05, seed: int = 0):
    """两条平行"板"：+1 类在 x2 > +margin，-1 类在 x2 < -margin。

    显式控制几何 margin gamma = margin（到真实边界 x2=0 的距离），
    同时 extent 控制数据半径 R——用于验证 Novikoff 收敛上界 (R/gamma)^2：
    margin 每减半，错分次数应约增至 4 倍。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-extent, extent, size=n_per_class * 2)
    x2 = np.concatenate(
        [
            margin + rng.uniform(0.0, noise, size=n_per_class),
            -margin - rng.uniform(0.0, noise, size=n_per_class),
        ]
    )
    X = np.column_stack([x1, x2])
    y = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    return X, y
