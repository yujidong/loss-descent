"""循环网络：SimpleRNN、LSTM 与字符级语言模型（Part 4）。

BPTT（沿时间反向传播）= 1.3 的三条递推在时间维上的展开：
    delta_t = (delta_{t+1} @ Whh) * f'(z_t)
LSTM 的细胞状态给梯度加了一条恒等传送带（constant error carousel），
是 3.2 残差思想在时间轴上的先声。
"""
from __future__ import annotations

import numpy as np


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class SimpleRNN:
    """h_t = tanh(Wxh[x_t] + Whh @ h_{t-1} + bh)"""

    def __init__(self, vocab: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab, self.hidden = vocab, hidden
        self.Wxh = rng.normal(0, 0.1, (vocab, hidden)) / np.sqrt(vocab)
        self.Whh = rng.normal(0, 0.1, (hidden, hidden)) / np.sqrt(hidden)
        self.bh = np.zeros(hidden)
        self.grads = {}

    def forward(self, tokens, h0=None):
        """tokens: (T,) 词 id。返回 h_1..h_T，形状 (T, hidden)。"""
        h = np.zeros(self.hidden) if h0 is None else h0
        hs = np.zeros((len(tokens), self.hidden))
        for t, tok in enumerate(tokens):
            h = np.tanh(self.Wxh[tok] + self.Whh @ h + self.bh)
            hs[t] = h
        self._cache = (tokens, hs)
        return hs

    def backward(self, dhs, dh_init=None):
        """dhs: (T, hidden) 各时刻 dL/dh_t；dh_init：从序列末端注入的梯度
        （初始状态携带的梯度，seq2seq 编码器链的缝合口）。"""
        tokens, hs = self._cache
        T = len(tokens)
        gWxh = np.zeros_like(self.Wxh)
        gWhh = np.zeros_like(self.Whh)
        gbh = np.zeros_like(self.bh)
        dh_next = np.zeros(self.hidden) if dh_init is None else dh_init
        for t in reversed(range(T)):
            dh = dhs[t] + dh_next
            dz = dh * (1.0 - hs[t] ** 2)
            gWxh[tokens[t]] += dz
            gWhh += np.outer(dz, hs[t - 1] if t > 0 else np.zeros(self.hidden))
            gbh += dz
            # 前向是 Whh @ h，链式法则回传 h_{t-1} 要用 Whh 的转置
            dh_next = self.Whh.T @ dz
        self.grads = {"Wxh": gWxh, "Whh": gWhh, "bh": gbh}
        return dh_next

    def step(self, lr: float, momentum: float = 0.9) -> None:
        if not hasattr(self, "_vel"):
            self._vel = {k: np.zeros_like(v) for k, v in
                         (("Wxh", self.Wxh), ("Whh", self.Whh), ("bh", self.bh))}
        for k, p in (("Wxh", self.Wxh), ("Whh", self.Whh), ("bh", self.bh)):
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]


class LSTM:
    """门控循环单元：i/f/g/o 四门，细胞状态 c 是梯度的恒等传送带。"""

    def __init__(self, vocab: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab, self.hidden = vocab, hidden
        # 拼接四门权重，按 [i, f, g, o] 切分
        self.Wx = rng.normal(0, 0.1, (vocab, 4 * hidden)) / np.sqrt(vocab)
        self.Wh = rng.normal(0, 0.1, (hidden, 4 * hidden)) / np.sqrt(hidden)
        self.b = np.zeros(4 * hidden)
        self.b[hidden : 2 * hidden] = 1.0  # 遗忘门偏置设 1（初始化为"记住"）
        self.grads = {}

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, tokens, state=None):
        h = np.zeros(self.hidden) if state is None else state[0]
        c = np.zeros(self.hidden) if state is None else state[1]
        h_start, c_start = h, c
        H = self.hidden
        hs = np.zeros((len(tokens), H))
        cs = np.zeros((len(tokens), H))
        gates = np.zeros((len(tokens), 4 * H))
        for t, tok in enumerate(tokens):
            z = self.Wx[tok] + h @ self.Wh + self.b
            i, f, g, o = self._sig(z[:H]), self._sig(z[H : 2 * H]), np.tanh(z[2 * H : 3 * H]), self._sig(z[3 * H :])
            c = f * c + i * g
            h = o * np.tanh(c)
            hs[t], cs[t], gates[t] = h, c, np.concatenate([i, f, g, o])
        self._cache = (tokens, hs, cs, gates, h_start, c_start)
        return hs, cs

    def backward(self, dhs, dh_init=None, dc_init=None):
        tokens, hs, cs, gates, h_start, c_start = self._cache
        T, H = len(tokens), self.hidden
        gWx = np.zeros_like(self.Wx)
        gWh = np.zeros_like(self.Wh)
        gb = np.zeros_like(self.b)
        dh_next = np.zeros(H) if dh_init is None else dh_init
        dc_next = np.zeros(H) if dc_init is None else dc_init
        for t in reversed(range(T)):
            i, f, g, o = gates[t, :H], gates[t, H : 2 * H], gates[t, 2 * H : 3 * H], gates[t, 3 * H :]
            c_prev = cs[t - 1] if t > 0 else c_start
            tanh_c = np.tanh(cs[t])
            dh = dhs[t] + dh_next
            do = dh * tanh_c * o * (1 - o)
            dc = dh * o * (1 - tanh_c**2) + dc_next
            df = dc * c_prev * f * (1 - f)
            di = dc * g * i * (1 - i)
            dg = dc * i * (1 - g**2)
            dz = np.concatenate([di, df, dg, do])
            gWx[tokens[t]] += dz
            gWh += np.outer(hs[t - 1] if t > 0 else h_start, dz)
            gb += dz
            dc_next = dc * f
            # 前向是 h @ Wh，回传 h_{t-1} 用 Wh（不转置）
            dh_next = self.Wh @ dz
        self.grads = {"Wx": gWx, "Wh": gWh, "b": gb}
        return dh_next, dc_next

    def step(self, lr: float, momentum: float = 0.9) -> None:
        if not hasattr(self, "_vel"):
            self._vel = {k: np.zeros_like(v) for k, v in
                         (("Wx", self.Wx), ("Wh", self.Wh), ("b", self.b))}
        for k, p in (("Wx", self.Wx), ("Wh", self.Wh), ("b", self.b)):
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]


def rnn_grad_flow_norms(cell, dhs):
    """BPTT 中每个时间步的 ||dh_t||（2.2 的梯度剖面在时间轴上的版本）。"""
    tokens, hs = cell._cache
    T = len(tokens)
    norms = np.zeros(T)
    dh_next = np.zeros(cell.hidden)
    for t in reversed(range(T)):
        dh = dhs[t] + dh_next
        norms[t] = float(np.linalg.norm(dh))
        dz = dh * (1.0 - hs[t] ** 2)
        dh_next = cell.Whh.T @ dz
    return norms


def lstm_grad_flow_norms(cell, dhs):
    tokens, hs, cs, gates, _, _ = cell._cache
    T, H = len(tokens), cell.hidden
    norms = np.zeros(T)
    dh_next = np.zeros(H)
    dc_next = np.zeros(H)
    for t in reversed(range(T)):
        i, f, g, o = gates[t, :H], gates[t, H : 2 * H], gates[t, 2 * H : 3 * H], gates[t, 3 * H :]
        tanh_c = np.tanh(cs[t])
        dh = dhs[t] + dh_next
        norms[t] = float(np.linalg.norm(dh))
        do = dh * tanh_c * o * (1 - o)
        dc = dh * o * (1 - tanh_c**2) + dc_next
        di, df, dg = dc * g * i * (1 - i), dc * (cs[t - 1] if t > 0 else 0) * f * (1 - f), dc * i * (1 - g**2)
        dz = np.concatenate([di, df, dg, do])
        dc_next = dc * f
        dh_next = cell.Wh @ dz
    return norms
