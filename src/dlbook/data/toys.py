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


def make_moons(n_per_class: int = 100, noise: float = 0.1, seed: int = 0):
    """双月数据集：非线性但温和的二分类，1990s 风格的"真实任务"难度。"""
    rng = np.random.default_rng(seed)
    t1, t2 = rng.uniform(0, np.pi, n_per_class), rng.uniform(0, np.pi, n_per_class)
    upper = np.column_stack([np.cos(t1), np.sin(t1)])
    lower = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (2 * n_per_class, 2))
    y = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    return X, y


def make_spiral(n_per_class: int = 100, turns: float = 1.75, noise: float = 0.1, seed: int = 0):
    """双螺旋：需要多层网络才能分开的经典难题（2.3/2.5 章的主力数据）。"""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for label, sign in ((1.0, 0.0), (-1.0, np.pi)):
        t = np.linspace(0.05, 1.0, n_per_class)
        r = t * np.sqrt(2.0)
        theta = sign + t * 2 * np.pi * turns
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        X += rng.normal(0, noise, X.shape)
        Xs.append(X)
        ys.append(np.full(n_per_class, label))
    return np.vstack(Xs), np.concatenate(ys)


def make_shifted_pattern(n: int = 400, size: int = 12, noise: float = 0.3, seed: int = 0):
    """平移图案分类：图像里有一个 3x3 图案（十字或方块）出现在随机位置。

    二分类 ±1（图案类型）。教学点：平移不变性——卷积的归纳偏置主场。
    """
    rng = np.random.default_rng(seed)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    square = np.ones((3, 3), dtype=float)
    X = np.zeros((n, 1, size, size))
    y = np.empty(n)
    for i in range(n):
        pat = cross if i % 2 == 0 else square
        top, left = rng.integers(0, size - 2, size=2)
        X[i, 0, top : top + 3, left : left + 3] = pat
        y[i] = 1.0 if i % 2 == 0 else -1.0
    X += rng.normal(0, noise, X.shape)
    return X, y


def make_seq_task(n: int = 400, length: int = 16, dist: int = 8, seed: int = 0):
    """序列回声任务：y = x[length - dist]（回看 dist 步之前的比特）。

    用于演示固定感受野（1D 卷积窗口）的长程依赖之痛：窗口 < dist 则不可解。
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, length)).astype(float)
    y = X[:, length - dist].copy() * 2 - 1  # 映射为 ±1
    return X, y
