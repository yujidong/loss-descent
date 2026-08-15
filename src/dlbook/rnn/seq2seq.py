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

    def step(self, lr: float, momentum: float = 0.9) -> None:
        for k, p in (("Why", self.Why), ("by", self.by)):
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]
        self.enc.step(lr, momentum=momentum)
        self.dec.step(lr, momentum=momentum)
