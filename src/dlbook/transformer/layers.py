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
    """线性层，可选 LoRA 适配器（7.2 章）：冻结 W，只训低秩增量 A@B。"""

    def __init__(self, d_in, d_out, seed=0, lora_r: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.02, (d_in, d_out)) * np.sqrt(2.0 / (d_in + d_out))
        self.b = np.zeros(d_out)
        self.grad_W = self.grad_b = None
        self._vel = {}
        self.lora_r = lora_r
        if lora_r > 0:
            self.lora_A = rng.normal(0, 0.01, (d_in, lora_r))
            self.lora_B = np.zeros((lora_r, d_out))  # B 零初始化：增量从恒等起步
            self.grad_lora_A = self.grad_lora_B = None

    def forward(self, X):
        self._X = X
        out = X @ self.W + self.b
        if self.lora_r > 0:
            out = out + (X @ self.lora_A) @ self.lora_B
        return out

    def backward(self, dY):
        dYf = dY.reshape(-1, dY.shape[-1])
        Xf = self._X.reshape(-1, self._X.shape[-1])
        if self.lora_r > 0:
            AX = Xf @ self.lora_A                      # (N, r)
            self.grad_lora_B = AX.T @ dYf              # (r, d_out)
            self.grad_lora_A = Xf.T @ (dYf @ self.lora_B.T)  # (d_in, r)
            self.grad_W = self.grad_b = None           # 基座冻结
        else:
            self.grad_W = Xf.T @ dYf
            self.grad_b = dYf.sum(axis=0)
        return dY @ (self.W + (self.lora_A @ self.lora_B if self.lora_r > 0 else 0)).T

    def step(self, lr, momentum=0.9):
        if self.lora_r > 0:
            for k, p in (("A", self.lora_A), ("B", self.lora_B)):
                if k not in self._vel:
                    self._vel[k] = np.zeros_like(p)
                self._vel[k] = momentum * self._vel[k] - lr * getattr(self, f"grad_lora_{k}")
                p += self._vel[k]
            return
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

    def __init__(self, d_model, n_heads, seed=0, lora_r: int = 0):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.dh = d_model, n_heads, d_model // n_heads
        self.Wq = Linear(d_model, d_model, seed, lora_r=lora_r)
        self.Wk = Linear(d_model, d_model, seed + 1, lora_r=lora_r)
        self.Wv = Linear(d_model, d_model, seed + 2, lora_r=lora_r)
        self.Wo = Linear(d_model, d_model, seed + 3, lora_r=lora_r)

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
    def __init__(self, d_model, d_hidden, seed=0, lora_r=0):
        self.fc1 = Linear(d_model, d_hidden, seed, lora_r=lora_r)
        self.fc2 = Linear(d_hidden, d_model, seed + 1, lora_r=lora_r)

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

    def __init__(self, d_model, n_heads, seed=0, lora_r=0):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, seed, lora_r=lora_r)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model, seed + 10, lora_r=lora_r)
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
class MoEMLP:
    """混合专家前馈块（6.4 章）：top-1 稀疏路由 + 直通门梯度。

    与 MLP 同接口。n_experts 位专家各是 2*d 隐层的小 MLP，每个位置
    只经得分最高的那位——活跃 FLOPs 与 dense(4d) 相当，参数加倍。
    门梯度用"软混合的直通近似"：dL/ds 经由 softmax 雅可比回传
    <dY, 专家输出> 的对齐度。
    """

    def __init__(self, d_model, n_experts=2, seed=0):
        self.n_experts = n_experts
        self.experts = [MLP(d_model, 2 * d_model, seed + 100 * i) for i in range(n_experts)]
        rng = np.random.default_rng(seed + 7)
        self.gate_w = rng.normal(0, 0.02, (d_model, n_experts))
        self.grad_gate_w = None
        self._vel = {}
        self.usage = np.full(n_experts, 1.0 / n_experts)

    def forward(self, X):
        p = softmax_lastdim(X @ self.gate_w)  # (B, T, E)
        top = np.argmax(p, axis=-1)           # (B, T)
        Y = np.zeros_like(X)
        for e, expert in enumerate(self.experts):
            sel = top == e
            if sel.any():
                Y[sel] = expert.forward(X[sel])
        self._X, self._p, self._top, self._Y = X, p, top, Y
        self.usage = np.array([(top == e).mean() for e in range(self.n_experts)])
        return Y

    def backward(self, dY):
        X, p, top, Y = self._X, self._p, self._top, self._Y
        dX = np.zeros_like(X)
        for e, expert in enumerate(self.experts):
            sel = top == e
            if sel.any():
                dX[sel] = expert.backward(dY[sel])
        # 直通门梯度：视作软混合 Σ p_e Y_e 的导数（未选专家输出记 0 的近似）
        indicator = (top[..., None, None] == np.arange(self.n_experts))  # (B,T,1,E)
        Y_e = Y[..., None] * indicator
        align = np.einsum("btd,btde->bte", dY, Y_e)
        ds = p * (align - (p * align).sum(axis=-1, keepdims=True))
        self.grad_gate_w = np.einsum("btd,bte->de", X, ds)
        return dX

    def step(self, lr, momentum=0.9):
        for e in self.experts:
            e.step(lr, momentum)
        if "g" not in self._vel:
            self._vel["g"] = np.zeros_like(self.gate_w)
        self._vel["g"] = momentum * self._vel["g"] - lr * self.grad_gate_w
        self.gate_w += self._vel["g"]
