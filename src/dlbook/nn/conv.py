"""numpy 手写卷积层与卷积网络（Part 3，1989 年的 LeNet 一脉）。

实现刻意朴素：单通道输入、stride=1、可选 padding、逐卷积核偏移的
向量化循环——规模小但梯度链完整，供 3.1/3.3 的"平移不变性/迁移"
实验使用。权重按扇入缩放初始化（2.1 的规则照常生效）。
"""
from __future__ import annotations

import numpy as np

from dlbook.nn.mlp_numpy import MLPNumpy


class Conv2D:
    """单输入通道卷积：X (n, 1, H, W) -> (n, out_ch, H', W')。"""

    def __init__(self, out_ch: int, kernel: int, padding: int = 0, seed: int = 0):
        self.out_ch, self.k, self.pad = out_ch, kernel, padding
        rng = np.random.default_rng(seed)
        fan_in = kernel * kernel
        self.W = rng.uniform(-1.0, 1.0, (out_ch, fan_in)) / np.sqrt(fan_in)
        self.b = np.zeros(out_ch)
        self.grad_W = self.grad_b = None

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def _pad(self, X):
        if self.pad == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))

    def forward(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X[:, None]  # (H,W) -> (1,H,W)
        X = self._pad(X)
        n, _, H, W = X.shape
        Ho, Wo = H - self.k + 1, W - self.k + 1
        Z = np.zeros((n, self.out_ch, Ho, Wo))
        Xp = X[:, 0]  # (n, H, W)
        # 把每个卷积位置的 patch 展平成 (n, k*k)，与 W (out_ch, k*k) 做矩阵乘
        patches = np.zeros((n, Ho * Wo, self.k * self.k))
        idx = 0
        for di in range(self.k):
            for dj in range(self.k):
                block = Xp[:, di : di + Ho, dj : dj + Wo]  # (n, Ho, Wo)
                patches[:, :, self.k * di + dj] = block.reshape(n, Ho * Wo)
        Z = (patches @ self.W.T).transpose(0, 2, 1).reshape(n, self.out_ch, Ho, Wo)
        Z += self.b[None, :, None, None]
        self._cache = (patches, X.shape)
        return Z

    def backward(self, dZ: np.ndarray) -> None:
        patches, _ = self._cache
        n = patches.shape[0]
        dz = dZ.reshape(n, self.out_ch, -1).transpose(0, 2, 1)  # (n, HoWo, out_ch)
        self.grad_W = np.einsum("npd,npo->od", patches, dz)
        self.grad_b = dz.sum(axis=(0, 1))
        # 输入梯度对上层（卷积特征层通常不需要回传给输入图像），此处省略 dX

    def step(self, lr: float) -> None:
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b


class SimpleConvNet:
    """单卷积层 + relu + 全连接头：LeNet 的最小重演版。

    任务限定为二分类（±1）。backward 把卷积层与 MLPNumpy 头的
    梯度链用 backward_from 缝起来——1.3 的三条递推贯穿两种层。
    """

    def __init__(self, in_hw: tuple[int, int], out_ch: int = 8, kernel: int = 3,
                 padding: int = 1, hidden: int = 32, seed: int = 0, pool: bool = True):
        """pool=True 时对卷积特征做全局平均池化——位置不变性在架构里（3.1 的正题）。"""
        self.conv = Conv2D(out_ch, kernel, padding=padding, seed=seed)
        self.pool = pool
        Ho, Wo = in_hw[0] + 2 * padding - kernel + 1, in_hw[1] + 2 * padding - kernel + 1
        feat = out_ch if pool else out_ch * Ho * Wo
        self.head = MLPNumpy(feat, [hidden, 1], activation="relu", seed=seed + 1)
        self._spatial = (Ho, Wo)
        self._cache = None

    def forward(self, X, training: bool = False):
        Z = self.conv.forward(X)
        Hr = np.maximum(Z, 0.0)
        if self.pool:
            flat = Hr.mean(axis=(2, 3))  # (n, out_ch)：每个滤波器在全图的平均响应
        else:
            flat = Hr.reshape(len(Hr), -1)
        out = self.head.forward(flat, training=training)
        self._cache = (Z, Hr, flat)
        return out

    def backward(self, X, Y) -> float:
        Y = np.asarray(Y, dtype=float).reshape(-1, 1)
        pred = self.forward(X, training=True)
        loss = float(np.mean((pred - Y) ** 2))
        dA = 2.0 * (pred - Y) / len(pred)
        Z, Hr, flat = self._cache
        # 全连接头：backward_from 填充头部梯度并返回 dL/d(flat)（梯度链缝合口）
        d_flat = self.head.backward_from(flat, dA)
        if self.pool:  # 全局平均的梯度：均分回每个空间位置
            dHr = np.broadcast_to(
                d_flat[:, :, None, None] / (self._spatial[0] * self._spatial[1]), Hr.shape
            ).copy()
        else:
            dHr = d_flat.reshape(Hr.shape)
        dZ = dHr * (Z > 0.0)
        self.conv.backward(dZ)
        return loss

    def step(self, lr: float, momentum: float = 0.0) -> None:
        self.conv.step(lr)
        self.head.step(lr, momentum=momentum)


class Conv1D:
    """一维卷积：X (n, L) -> (n, out_ch, L-k+1)。取末位输出即可做因果式预测头（3.4 章）。"""

    def __init__(self, out_ch: int, kernel: int, seed: int = 0):
        self.out_ch, self.k = out_ch, kernel
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-1.0, 1.0, (out_ch, kernel)) / np.sqrt(kernel)
        self.b = np.zeros(out_ch)
        self.grad_W = self.grad_b = None

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, L = X.shape
        Lo = L - self.k + 1
        patches = np.zeros((n, Lo, self.k))
        for d in range(self.k):
            patches[:, :, d] = X[:, d : d + Lo]
        Z = patches @ self.W.T + self.b
        self._cache = (patches,)
        return Z.transpose(0, 2, 1)  # (n, out_ch, Lo)

    def backward(self, dZ):
        patches, = self._cache
        dz = dZ.transpose(0, 2, 1)  # (n, Lo, out_ch)
        self.grad_W = np.einsum("npd,npo->od", patches, dz)
        self.grad_b = dz.sum(axis=(0, 1))

    def step(self, lr: float) -> None:
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b
