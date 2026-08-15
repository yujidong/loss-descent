"""字符级语言模型：RNN/LSTM + softmax 交叉熵 + 采样生成（Part 4）。"""
from __future__ import annotations

import numpy as np

from dlbook.rnn.core import LSTM, SimpleRNN, softmax


class RNNLM:
    """循环语言模型：cell 可选 "rnn" / "lstm"，输出层 softmax 交叉熵。"""

    def __init__(self, vocab: int, hidden: int, cell: str = "rnn", seed: int = 0):
        rng = np.random.default_rng(seed + 99)
        self.vocab, self.hidden, self.cell_type = vocab, hidden, cell
        self.cell = (SimpleRNN(vocab, hidden, seed) if cell == "rnn"
                     else LSTM(vocab, hidden, seed))
        self.Why = rng.normal(0, 0.1, (hidden, vocab)) / np.sqrt(hidden)
        self.by = np.zeros(vocab)
        self.grads = {}
        self._vel = {}

    def forward(self, tokens):
        if self.cell_type == "rnn":
            return self.cell.forward(tokens)
        return self.cell.forward(tokens)[0]  # 只取 h 序列

    def loss_and_backward(self, tokens) -> float:
        """输入 tokens[:-1] 预测 tokens[1:]，返回平均交叉熵（nat/字符）。"""
        tokens = np.asarray(tokens)
        inputs, targets = tokens[:-1], tokens[1:]
        hs = self.forward(inputs)
        logits = hs @ self.Why + self.by
        probs = softmax(logits)
        T = len(inputs)
        loss = float(-np.mean(np.log(probs[np.arange(T), targets] + 1e-12)))

        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T
        self.grads = {
            "Why": hs.T @ dlogits,
            "by": dlogits.sum(axis=0),
        }
        dhs = dlogits @ self.Why.T
        self.cell.backward(dhs)
        return loss

    def hidden_state(self, tokens):
        return self.forward(tokens)

    def final_state(self, tokens):
        if self.cell_type == "lstm":
            return self.cell.forward(tokens)[1][-1]  # 细胞状态（seq2seq 编码用）
        return self.forward(tokens)[-1]

    def step(self, lr: float, momentum: float = 0.9) -> None:
        params = [("Why", self.Why), ("by", self.by)]
        for k, p in params:
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * self.grads[k]
            p += self._vel[k]
        self.cell.step(lr, momentum=momentum)

    def loss_on(self, tokens) -> float:
        tokens = np.asarray(tokens)
        inputs, targets = tokens[:-1], tokens[1:]
        logits = self.forward(inputs) @ self.Why + self.by
        probs = softmax(logits)
        return float(-np.mean(np.log(probs[np.arange(len(inputs)), targets] + 1e-12)))

    def sample(self, prompt_ids, n_new: int = 200, temperature: float = 0.8, seed: int = 0):
        """从 prompt 续写 n_new 个字符（带温度采样）。"""
        rng = np.random.default_rng(seed)
        out = list(prompt_ids)
        if self.cell_type == "lstm":
            hs, cs = self.cell.forward(out)
            h, c = hs[-1], cs[-1]
        else:
            h, c = self.cell.forward(out)[-1], None
        for _ in range(n_new):
            if self.cell_type == "lstm":
                hs, cs = self.cell.forward([out[-1]], (h, c))
                h, c = hs[-1], cs[-1]
            else:
                h = self.cell.forward([out[-1]], h)[-1]
            logits = h @ self.Why + self.by
            p = softmax(logits / temperature)
            out.append(int(rng.choice(len(p), p=p)))
        return out
