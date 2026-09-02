"""seq2seq：编码器-解码器（4.5 章）与序列分类器（4.3 章的远距试金石）。"""
from __future__ import annotations

import numpy as np

from dlbook.rnn.core import LSTM, SimpleRNN, softmax


class SeqClassifier:
    """把整条序列压进 h_T 后做线性二分类——远距依赖的试金石。"""

    def __init__(self, vocab: int, hidden: int, cell: str = "rnn", seed: int = 0):
        rng = np.random.default_rng(seed + 77)
        self.cell = (SimpleRNN(vocab, hidden, seed) if cell == "rnn"
                     else LSTM(vocab, hidden, seed))
        self.vocab, self.hidden, self.cell_type = vocab, hidden, cell
        self.w = rng.normal(0, 0.1, hidden) / np.sqrt(hidden)
        self.b0 = 0.0
        self.grads = {}
        self._vel = {}

    def loss_and_backward(self, tokens, y: float) -> float:
        hs = (self.cell.forward(tokens) if self.cell_type == "rnn"
              else self.cell.forward(tokens)[0])
        z = float(hs[-1] @ self.w + self.b0)
        loss = (z - y) ** 2
        dz = 2.0 * (z - y)
        self.grads = {"w": dz * hs[-1], "b0": dz}
        dhs = np.zeros_like(hs)
        dhs[-1] = dz * self.w
        self.cell.backward(dhs)
        return loss

    def predict(self, tokens) -> float:
        hs = (self.cell.forward(tokens) if self.cell_type == "rnn"
              else self.cell.forward(tokens)[0])
        return float(np.sign(hs[-1] @ self.w + self.b0))

    def step(self, lr: float, momentum: float = 0.9) -> None:
        for k, p in (("w", self.w), ("b0", self.b0)):
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p) if hasattr(p, "shape") else 0.0
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]
        self.cell.step(lr, momentum=momentum)


class Seq2Seq:
    """编码器-解码器：LSTM 压缩 + LSTM 解码（teacher forcing 训练）。

    编码器的终态 (h, c) 成为解码器的初态——"固定维度向量装下整句"
    的瓶颈（4.5 章的主角与 5.1 章的靶子）。
    """

    def __init__(self, vocab: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed + 55)
        self.enc = LSTM(vocab, hidden, seed)
        self.dec = LSTM(vocab, hidden, seed + 1)
        self.vocab, self.hidden = vocab, hidden
        self.Why = rng.normal(0, 0.1, (hidden, vocab)) / np.sqrt(hidden)
        self.by = np.zeros(vocab)
        self.grads = {}
        self._vel = {}

    def loss_and_backward(self, inp, out) -> float:
        """inp/out 都以 BOS 开头、EOS 结尾；teacher forcing。"""
        hs_e, cs_e = self.enc.forward(inp)
        dec_in, dec_tg = np.asarray(out[:-1]), np.asarray(out[1:])
        hs_d, _ = self.dec.forward(dec_in, (hs_e[-1], cs_e[-1]))
        logits = hs_d @ self.Why + self.by
        probs = softmax(logits)
        T = len(dec_in)
        loss = float(-np.mean(np.log(probs[np.arange(T), dec_tg] + 1e-12)))

        dlogits = probs.copy()
        dlogits[np.arange(T), dec_tg] -= 1.0
        dlogits /= T
        self.grads = {"Why": hs_d.T @ dlogits, "by": dlogits.sum(axis=0)}
        dhs = dlogits @ self.Why.T
        dh0, dc0 = self.dec.backward(dhs)
        dhs_e = np.zeros_like(hs_e)
        dhs_e[-1] = dh0  # 编码器终态的梯度
        self.enc.backward(dhs_e, dc_init=dc0)
        return loss

    def greedy_decode(self, inp, bos: int = 0, eos: int = 1, max_len: int = 12):
        hs_e, cs_e = self.enc.forward(inp)
        h, c = hs_e[-1], cs_e[-1]
        out = [bos]
        for _ in range(max_len):
            hs, cs = self.dec.forward([out[-1]], (h, c))
            h, c = hs[-1], cs[-1]
            nxt = int(np.argmax(h @ self.Why + self.by))
            out.append(nxt)
            if nxt == eos:
                break
        return out

    def sample_decode(self, inp, bos: int = 0, eos: int = 1, max_len: int = 12,
                      temperature: float = 0.8, seed: int = 0):
        """带温度的自回归采样（7.4 章测试时计算的原料）。"""
        rng = np.random.default_rng(seed)
        hs_e, cs_e = self.enc.forward(inp)
        h, c = hs_e[-1], cs_e[-1]
        out = [bos]
        for _ in range(max_len):
            hs, cs = self.dec.forward([out[-1]], (h, c))
            h, c = hs[-1], cs[-1]
            logits = (h @ self.Why + self.by) / max(temperature, 1e-6)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            if nxt == eos:
                break
        return out

    def step(self, lr: float, momentum: float = 0.9) -> None:
        for k, p in (("Why", self.Why), ("by", self.by)):
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]
        self.enc.step(lr, momentum=momentum)
        self.dec.step(lr, momentum=momentum)


class AttentionSeq2Seq:
    """Bahdanau 式注意力（5.1 章）：解码每步对编码器各状态做点积注意力。

    与 Seq2Seq 的唯一区别：解码器每步不再只依赖编码器终态，
    而是拿到一个随步变化的上下文 ctx_t = Σ_s α_{t,s} h_s^enc。
    反向链穿过注意力权重回到每一个编码器状态——"让梯度与信息
    都能直达任意源位置"。
    """

    def __init__(self, vocab: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed + 66)
        self.enc = LSTM(vocab, hidden, seed)
        self.dec = LSTM(vocab, hidden, seed + 1)
        self.vocab, self.hidden = vocab, hidden
        self.Why = rng.normal(0, 0.1, (2 * hidden, vocab)) / np.sqrt(2 * hidden)
        self.by = np.zeros(vocab)
        self.grads = {}
        self._vel = {}

    def _attend(self, hs_d, hs_e):
        scores = hs_d @ hs_e.T / np.sqrt(self.hidden)  # (T, S)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        alpha = e / e.sum(axis=-1, keepdims=True)
        return alpha, alpha @ hs_e  # (T,S), (T,H)

    def loss_and_backward(self, inp, out) -> float:
        hs_e, cs_e = self.enc.forward(inp)
        dec_in, dec_tg = np.asarray(out[:-1]), np.asarray(out[1:])
        hs_d, _ = self.dec.forward(dec_in, (hs_e[-1], cs_e[-1]))
        alpha, ctx = self._attend(hs_d, hs_e)
        concat = np.concatenate([hs_d, ctx], axis=1)
        logits = concat @ self.Why + self.by
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        T = len(dec_in)
        loss = float(-np.mean(np.log(probs[np.arange(T), dec_tg] + 1e-12)))

        dlogits = probs.copy()
        dlogits[np.arange(T), dec_tg] -= 1.0
        dlogits /= T
        self.grads = {"Why": concat.T @ dlogits, "by": dlogits.sum(axis=0)}
        dconcat = dlogits @ self.Why.T
        dhs_d = dconcat[:, : self.hidden].copy()
        dctx = dconcat[:, self.hidden :]

        dalpha = dctx @ hs_e.T
        # 前向 scores 乘过 1/sqrt(hidden)，反向要乘回同一系数
        dscores = alpha * (dalpha - (dalpha * alpha).sum(axis=-1, keepdims=True)) / np.sqrt(self.hidden)
        dhs_d += dscores @ hs_e
        dhs_e = alpha.T @ dctx + dscores.T @ hs_d

        dh0, dc0 = self.dec.backward(dhs_d)
        dhs_e[-1] += dh0
        self.enc.backward(dhs_e, dc_init=dc0)
        return loss

    def greedy_decode(self, inp, bos: int = 0, eos: int = 1, max_len: int = 14):
        hs_e, cs_e = self.enc.forward(inp)
        h, c = hs_e[-1], cs_e[-1]
        out = [bos]
        for _ in range(max_len):
            hs_d, cs_d = self.dec.forward([out[-1]], (h, c))
            h, c = hs_d[-1], cs_d[-1]
            _, ctx = self._attend(hs_d, hs_e)
            concat = np.concatenate([hs_d[-1], ctx[-1]])
            nxt = int(np.argmax(concat @ self.Why + self.by))
            out.append(nxt)
            if nxt == eos:
                break
        return out

    def sample_decode(self, inp, bos: int = 0, eos: int = 1, max_len: int = 14,
                      temperature: float = 0.8, seed: int = 0):
        """带温度的自回归采样（7.4 章测试时计算的原料）。"""
        rng = np.random.default_rng(seed)
        hs_e, cs_e = self.enc.forward(inp)
        h, c = hs_e[-1], cs_e[-1]
        out = [bos]
        for _ in range(max_len):
            hs_d, cs_d = self.dec.forward([out[-1]], (h, c))
            h, c = hs_d[-1], cs_d[-1]
            _, ctx = self._attend(hs_d, hs_e)
            concat = np.concatenate([hs_d[-1], ctx[-1]])
            logits = (concat @ self.Why + self.by) / max(temperature, 1e-6)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            if nxt == eos:
                break
        return out

    def alignment(self, inp, out):
        """对齐热图 (T_dec, S_enc)：训练后可视化"每步看哪里"。"""
        hs_e, cs_e = self.enc.forward(inp)
        h, c = hs_e[-1], cs_e[-1]
        hs_list = []
        for tok in out[:-1]:
            hs_d, cs_d = self.dec.forward([tok], (h, c))
            h, c = hs_d[-1], cs_d[-1]
            hs_list.append(hs_d[-1])
        alpha, _ = self._attend(np.stack(hs_list), hs_e)
        return alpha

    def step(self, lr: float, momentum: float = 0.9) -> None:
        for k, p in (("Why", self.Why), ("by", self.by)):
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]
        self.enc.step(lr, momentum=momentum)
        self.dec.step(lr, momentum=momentum)
