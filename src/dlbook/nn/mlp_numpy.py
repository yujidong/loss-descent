"""numpy 向量版 MLP：手写反向传播（含 Part 2/3 工具箱的全部部件）。

部件按历史顺序逐步启用：
    扇入缩放初始化（2.1）、动量（2.3）、ReLU（2.3）、Dropout（2.3）、
    BatchNorm（2.3）、残差连接（3.2）。
默认关闭（初始化除外）——每章按需打开，重演发明史。

反向传播仍是 1.3 章的三条递推，只是在每层多了 BN/.dropout/残差的
局部导数。backward_from() 允许外部把梯度链续进来（3.1 的卷积头用它）。
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
    "linear": (lambda z: z, lambda z, a: np.ones_like(z)),
}


class MLPNumpy:
    """全连接 MLP：隐藏层 activation（可 BN/dropout/残差），输出层线性。

    - init="scaled" 按扇入缩放（2.1 的理论），"uniform" 保留朴素初始化（做对照）；
    - residual=True 时，宽度相同的相邻隐藏层之间加恒等跳连（3.2）。
    """

    def __init__(
        self,
        nin: int,
        nouts: list[int],
        activation: str = "tanh",
        seed: int = 0,
        *,
        init: str = "scaled",
        dropout: float = 0.0,
        batchnorm: bool = False,
        residual: bool = False,
    ):
        rng = np.random.default_rng(seed)
        sizes = [nin] + list(nouts)
        self.n_layers = len(sizes) - 1
        hidden = sizes[1:-1]
        if residual and len(set(hidden)) > 1:
            raise ValueError("残差连接要求所有隐藏层等宽")
        self.activation, self.dropout, self.batchnorm, self.residual = (
            activation, dropout, batchnorm, residual,
        )
        self._rng = np.random.default_rng(seed + 1)

        self.Ws = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = rng.uniform(-1.0, 1.0, (b, a))
            if init == "scaled":
                W = W / np.sqrt(a)
            self.Ws.append(W)
        self.bs = [np.zeros(b) for b in sizes[1:]]

        # BatchNorm 参数（仅隐藏层）：y = gamma * xhat + beta
        self.gamma = [np.ones(b) for b in hidden] if batchnorm else []
        self.beta = [np.zeros(b) for b in hidden] if batchnorm else []
        self.running_mean = [np.zeros(b) for b in hidden] if batchnorm else []
        self.running_var = [np.ones(b) for b in hidden] if batchnorm else []

        self.grad_Ws: list[np.ndarray] | None = None
        self._cache: list | None = None
        self._vel: list | None = None

    # ---- 前向 ----

    def forward(self, X, training: bool = False) -> np.ndarray:
        """training=True 时启用 dropout 与 BN 批统计（并更新 running stats）。"""
        A = np.atleast_2d(np.asarray(X, dtype=float))
        self._cache = []
        for i in range(self.n_layers):
            last = i == self.n_layers - 1
            Z = A @ self.Ws[i].T + self.bs[i]

            xhat = var_inv = preact = None
            if not last and self.batchnorm:
                if training:
                    mu, var = Z.mean(axis=0), Z.var(axis=0)
                    self.running_mean[i] = 0.9 * self.running_mean[i] + 0.1 * mu
                    self.running_var[i] = 0.9 * self.running_var[i] + 0.1 * var
                else:
                    mu, var = self.running_mean[i], self.running_var[i]
                var_inv = 1.0 / np.sqrt(var + 1e-5)
                xhat = (Z - mu) * var_inv
                preact = xhat * self.gamma[i] + self.beta[i]
            else:
                preact = Z

            act = _ACT["linear"][0] if last else _ACT[self.activation][0]
            H = act(preact)

            mask = None
            Hout = H
            if not last and self.dropout > 0.0 and training:
                keep = 1.0 - self.dropout
                mask = (self._rng.random(H.shape) < keep) / keep
                Hout = H * mask

            A_new = Hout
            if not last and self.residual and A.shape == Hout.shape:
                A_new = Hout + A  # 恒等跳连（3.2）

            self._cache.append((A, preact, xhat, var_inv, mask, H, Hout))
            A = A_new
        return A

    def __call__(self, X):
        return self.forward(X)

    # ---- 反向 ----

    def backward(self, X, Y) -> float:
        """全批 MSE 的反向传播：填充梯度，返回 loss。
        loss 与梯度都按 (batch × 输出维) 平均，多输出时二者口径一致。"""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        pred = self.forward(X, training=True)
        loss = float(np.mean((pred - Y) ** 2))
        self.backward_from(X, 2.0 * (pred - Y) / pred.shape[0] / pred.shape[1])
        return loss

    def backward_from(self, X, dA) -> np.ndarray:
        """从 dL/d(输出) 续接反向传播，返回 dL/d(输入)（外部梯度链的缝合口）。"""
        if self._cache is None or len(self._cache) != self.n_layers:
            self.forward(X, training=True)
        L = self.n_layers
        self.grad_Ws = [None] * L
        self.grad_bs = [None] * L
        self.grad_gamma = [None] * (L - 1)
        self.grad_beta = [None] * (L - 1)
        for i in reversed(range(L)):
            A, preact, xhat, var_inv, mask, H, Hout = self._cache[i]
            last = i == L - 1
            d_ident = None
            if not last and self.residual and A.shape == Hout.shape:
                d_ident = dA  # 恒等路径：梯度原样回传给上一层输出

            dH = dA * mask if mask is not None else dA

            fprime = _ACT["linear" if last else self.activation][1]
            dpre = dH * fprime(preact, H)  # 先过激活的局部导数
            if xhat is not None:  # BatchNorm 反向（标准公式）
                self.grad_gamma[i] = (dpre * xhat).sum(axis=0)
                self.grad_beta[i] = dpre.sum(axis=0)
                dZ = self.gamma[i] * var_inv * (
                    dpre - dpre.mean(axis=0) - xhat * (dpre * xhat).mean(axis=0)
                )
            else:
                dZ = dpre

            self.grad_Ws[i] = dZ.T @ A
            self.grad_bs[i] = dZ.sum(axis=0)
            dA = dZ @ self.Ws[i]
            if d_ident is not None:
                dA = dA + d_ident
        return dA

    # ---- 更新 ----

    def step(self, lr: float, momentum: float = 0.0) -> None:
        """可选动量（Polyak 1964）。"""
        groups = [
            (self.Ws, self.grad_Ws),
            (self.bs, self.grad_bs),
            (self.gamma, self.grad_gamma),
            (self.beta, self.grad_beta),
        ]
        flat = [p for group, _ in groups for p in group]
        grads = [g for _, glist in groups for g in glist]
        if momentum > 0 and self._vel is None:
            self._vel = [np.zeros_like(p) for p in flat]
        for k, p in enumerate(flat):
            if grads[k] is None:
                continue
            if momentum > 0:
                v = self._vel[k]
                v *= momentum
                v -= lr * grads[k]
                p += v
            else:
                p -= lr * grads[k]

    def grad_norms(self):
        """各层权重梯度的 L2 范数（backward 之后调用）。"""
        return [
            float(np.sqrt(np.sum(gW**2) + np.sum(gb**2)))
            for gW, gb in zip(self.grad_Ws, self.grad_bs)
        ]
