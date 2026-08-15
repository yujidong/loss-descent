"""Transformer 的积木：多头注意力、LayerNorm、前馈块（5.2 章）。

反向传播依然是 1.3 的三条递推——只是路径穿过了 softmax 注意力、
残差与层归一化。所有形状约定：X 为 (B, T, D)。
"""
from __future__ import annotations

import numpy as np


def softmax_lastdim(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def gelu_tanh(x):
    """GELU 的 tanh 近似（原论文公式），反向用乘法法则推出。"""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    dy = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * c * (1.0 + 3 * 0.044715 * x**2)
    return y, dy


class Linear:
    def __init__(self, d_in, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.02, (d_in, d_out)) * np.sqrt(2.0 / (d_in + d_out))
        self.b = np.zeros(d_out)
        self.grad_W = self.grad_b = None
        self._vel = {}

    def forward(self, X):
        self._X = X
        return X @ self.W + self.b

    def backward(self, dY):
        self.grad_W = self._X.reshape(-1, self._X.shape[-1]).T @ dY.reshape(-1, dY.shape[-1])
        self.grad_b = dY.reshape(-1, dY.shape[-1]).sum(axis=0)
        return dY @ self.W.T

    def step(self, lr, momentum=0.9):
        for k, p in (("W", self.W), ("b", self.b)):
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * getattr(self, f"grad_{k}")
            p += self._vel[k]


class LayerNorm:
    def __init__(self, d):
        self.g = np.ones(d)
        self.b = np.zeros(d)
        self.grad_g = self.grad_b = None
        self._vel = {}

    def forward(self, X):
        mu = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + 1e-5)
        self._xhat = (X - mu) * self._inv
        return self._xhat * self.g + self.b

    def backward(self, dY):
        self.grad_g = (dY * self._xhat).sum(axis=(0, 1))
        self.grad_b = dY.sum(axis=(0, 1))
        dxhat = dY * self.g
        return self._inv * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self._xhat * (dxhat * self._xhat).mean(axis=-1, keepdims=True)
        )

    def step(self, lr, momentum=0.9):
        for k, p in (("g", self.g), ("b", self.b)):
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * getattr(self, f"grad_{k}")
            p += self._vel[k]


class MultiHeadAttention:
    """缩放点积注意力 + 输出投影。causal=True 时屏蔽未来位置。"""

    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.dh = d_model, n_heads, d_model // n_heads
        self.Wq = Linear(d_model, d_model, seed)
        self.Wk = Linear(d_model, d_model, seed + 1)
        self.Wv = Linear(d_model, d_model, seed + 2)
        self.Wo = Linear(d_model, d_model, seed + 3)

    def _split(self, X):
        B, T, _ = X.shape
        return X.reshape(B, T, self.n_heads, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, X):
        B, nh, T, dh = X.shape
        return X.transpose(0, 2, 1, 3).reshape(B, T, nh * dh)

    def forward(self, X, causal=True):
        Q, K, V = self._split(self.Wq.forward(X)), self._split(self.Wk.forward(X)), self._split(self.Wv.forward(X))
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dh)  # (B, nh, T, T)
        if causal:
            T = X.shape[1]
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            S = np.where(mask, -1e9, S)
        A = softmax_lastdim(S)
        heads = A @ V
        out = self.Wo.forward(self._merge(heads))
        self._cache = (X, Q, K, V, S if causal else S, A)
        return out

    def backward(self, d_out):
        X, Q, K, V, S, A = self._cache
        d_heads = self._split(self.Wo.backward(d_out))  # (B,nh,T,dh)
        dA = d_heads @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ d_heads
        P = A
        dS = P * (dA - (dA * P).sum(axis=-1, keepdims=True))  # softmax 反向
        dQ = dS @ K / np.sqrt(self.dh)
        dK = dS.transpose(0, 1, 3, 2) @ Q / np.sqrt(self.dh)
        dX_q = self.Wq.backward(self._merge(dQ))
        dX_k = self.Wk.backward(self._merge(dK))
        dX_v = self.Wv.backward(self._merge(dV))
        return dX_q + dX_k + dX_v

    def step(self, lr, momentum=0.9):
        for lin in (self.Wq, self.Wk, self.Wv, self.Wo):
            lin.step(lr, momentum)

    def attention_map(self, X, causal=True):
        """取第一个样本的平均注意力图 (T, T)。"""
        _, _, _, _, _, A = self._cache if self._cache else (None,) * 6
        if A is None:
            self.forward(X, causal)
            A = self._cache[-1]
        return A[0].mean(axis=0)


class MLP:
    def __init__(self, d_model, d_hidden, seed=0):
        self.fc1 = Linear(d_model, d_hidden, seed)
        self.fc2 = Linear(d_hidden, d_model, seed + 1)

    def forward(self, X):
        self._z1 = self.fc1.forward(X)
        self._a1, self._dz1 = gelu_tanh(self._z1)
        return self.fc2.forward(self._a1)

    def backward(self, dY):
        da1 = self.fc2.backward(dY)
        return self.fc1.backward(da1 * self._dz1)

    def step(self, lr, momentum=0.9):
        self.fc1.step(lr, momentum)
        self.fc2.step(lr, momentum)


class Block:
    """pre-LN Transformer 块（GPT-2 风格）：x += attn(LN(x)); x += mlp(LN(x))。"""

    def __init__(self, d_model, n_heads, seed=0):
        self.ln1, self.attn = LayerNorm(d_model), MultiHeadAttention(d_model, n_heads, seed)
        self.ln2, self.mlp = LayerNorm(d_model), MLP(d_model, 4 * d_model, seed + 10)
        self.attn_map = None

    def forward(self, X, causal=True):
        h = self.ln1.forward(X)
        att = self.attn.forward(h, causal=causal)
        self.attn_map = self.attn.attention_map(h)
        X = X + att
        X = X + self.mlp.forward(self.ln2.forward(X))
        return X

    def backward(self, dX):
        dX2 = self.mlp.backward(self.ln2.backward(dX))
        dX = dX + dX2
        d_att = self.ln1.backward(self.attn.backward(dX))
        return dX + d_att

    def step(self, lr, momentum=0.9):
        self.attn.step(lr, momentum)
        self.mlp.step(lr, momentum)
        self.ln1.step(lr, momentum)
        self.ln2.step(lr, momentum)
