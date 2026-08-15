"""numpy 向量版 MLP：手写反向传播。

不再建计算图——直接按 1.3 章的三条递推实现层间梯度：
    delta_L = 2(A_L - Y)/n
    delta_l = (delta_{l+1} @ W_{l+1}) * f'(Z_l)
    dL/dW_l = delta_l^T @ A_{l-1}
这也是自动微分普及前（1986–2010s）实践者的真实写法。
标量引擎（dlbook.autodiff）负责讲原理；从这里起实验改用向量引擎。
"""
from __future__ import annotations

import numpy as np

_ACT = {
    "tanh": (np.tanh, lambda z, a: 1.0 - a**2),
    "relu": (lambda z: np.maximum(0.0, z), lambda z, a: (z > 0).astype(float)),
    "sigmoid": (
        lambda z: 1.0 / (1.0 + np.exp(-z)),
        lambda z, a: a * (1.0 - a),
    ),
}


class MLPNumpy:
    """全连接 MLP：隐藏层用 activation，输出层线性（回归）。"""

    def __init__(self, nin: int, nouts: list[int], activation: str = "tanh", seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [nin] + list(nouts)
        # 按扇入缩放初始化幅度（为什么必须这样做，2.1 章揭晓）
        self.Ws = [
            rng.uniform(-1.0, 1.0, (b, a)) / np.sqrt(a)
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        self.bs = [np.zeros(b) for b in sizes[1:]]
        self.activation = activation
        self.grad_Ws: list[np.ndarray] | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        acts = [X]  # A_0 .. A_L
        zs = []
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            Z = acts[-1] @ W.T + b
            if i < len(self.Ws) - 1:
                Z = _ACT[self.activation][0](Z)
            zs.append(Z)
            acts.append(Z)
        self._acts, self._zs = acts, zs
        return acts[-1]

    def __call__(self, X):
        return self.forward(X)

    def backward(self, X, Y) -> float:
        """对全批 MSE 做反向传播，填充 grad_Ws/grad_bs，返回 loss。"""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        pred = self.forward(X)
        loss = float(np.mean((pred - Y) ** 2))

        delta = 2.0 * (pred - Y) / len(X)  # dL/dZ_L（输出层线性）
        grad_Ws, grad_bs = [None] * len(self.Ws), [None] * len(self.bs)
        for i in reversed(range(len(self.Ws))):
            grad_Ws[i] = delta.T @ self._acts[i]
            grad_bs[i] = delta.sum(axis=0)
            if i > 0:
                df = _ACT[self.activation][1]
                delta = (delta @ self.Ws[i]) * df(self._zs[i - 1], self._acts[i])
        self.grad_Ws, self.grad_bs = grad_Ws, grad_bs
        return loss

    def step(self, lr: float, momentum: float = 0.0) -> None:
        """可选动量（Polyak 1964）：本引擎默认关，实验里按需开启。"""
        if momentum > 0 and not hasattr(self, "_vel"):
            self._vel = [
                (np.zeros_like(W), np.zeros_like(b)) for W, b in zip(self.Ws, self.bs)
            ]
        for i in range(len(self.Ws)):
            if momentum > 0:
                vW, vb = self._vel[i]
                vW *= momentum
                vW -= lr * self.grad_Ws[i]
                vb *= momentum
                vb -= lr * self.grad_bs[i]
                self.Ws[i] += vW
                self.bs[i] += vb
            else:
                self.Ws[i] -= lr * self.grad_Ws[i]
                self.bs[i] -= lr * self.grad_bs[i]

    def grad_norms(self):
        """各层参数梯度的 L2 范数（backward 之后调用）。"""
        return [
            float(np.sqrt(np.sum(gW**2) + np.sum(gb**2)))
            for gW, gb in zip(self.grad_Ws, self.grad_bs)
        ]
