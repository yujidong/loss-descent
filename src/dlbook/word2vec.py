"""word2vec：skip-gram + 负采样（4.4 章）。

"语义被压进向量"的最小实现：中心词与上下文词的嵌入内积要大、
与随机负样本的内积要小——一个把共现统计压进几何结构的 loss。
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class SkipGram:
    def __init__(self, vocab: int, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Win = rng.normal(0, 0.1, (vocab, dim)) / np.sqrt(dim)  # 中心词嵌入
        self.Wout = rng.normal(0, 0.1, (vocab, dim)) / np.sqrt(dim)  # 上下文嵌入
        self.vocab, self.dim = vocab, dim
        self._rng = np.random.default_rng(seed + 1)
        self._vel = {}

    def train(self, encoded_corpus, epochs: int = 15, lr: float = 0.05,
              window: int = 2, n_neg: int = 4):
        """encoded_corpus: 词 id 句子列表。返回每 epoch 的平均损失。"""
        losses = []
        for ep in range(epochs):
            total, count = 0.0, 0
            order = self._rng.permutation(len(encoded_corpus))
            for si in order:
                sent = encoded_corpus[si]
                for i, center in enumerate(sent):
                    lo, hi = max(0, i - window), min(len(sent), i + window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        context = sent[j]
                        # 教学版负采样：均匀分布且不排除与正样本碰撞
                        # （原论文用 unigram^0.75 并过滤碰撞；小词表上差异有限）
                        negs = self._rng.integers(0, self.vocab, size=n_neg)
                        loss = self._sgd_step(center, context, negs, lr)
                        total += loss
                        count += 1
            losses.append(total / max(count, 1))
        return losses

    def _sgd_step(self, center, context, negatives, lr):
        v = self.Win[center]
        pairs = [(context, 1.0)] + [(int(n), 0.0) for n in negatives]
        loss = 0.0
        gv = np.zeros_like(v)
        for word, label in pairs:
            u = self.Wout[word]
            p = _sigmoid(float(u @ v))
            loss += -np.log(p + 1e-12) if label else -np.log(1 - p + 1e-12)
            g = p - label
            gv += g * u
            self.Wout[word] -= lr * g * v
        self.Win[center] -= lr * gv
        return float(loss)

    def most_similar(self, word: str, vocab: dict, topk: int = 4):
        inv = {i: w for w, i in vocab.items()}
        target = self.Win[vocab[word]]
        norm = np.linalg.norm(self.Win, axis=1) * np.linalg.norm(target) + 1e-12
        sims = self.Win @ target / norm
        order = np.argsort(-sims)
        out = []
        for idx in order:
            w = inv[int(idx)]
            if w != word:
                out.append((w, float(sims[idx])))
            if len(out) >= topk:
                break
        return out

    def analogy(self, a: str, b: str, c: str, vocab: dict):
        """b - a + c 的最近邻（排除 a/b/c 自身）。"""
        inv = {i: w for w, i in vocab.items()}
        target = self.Win[vocab[b]] - self.Win[vocab[a]] + self.Win[vocab[c]]
        norm = np.linalg.norm(self.Win, axis=1) * np.linalg.norm(target) + 1e-12
        sims = self.Win @ target / norm
        order = np.argsort(-sims)
        for idx in order:
            w = inv[int(idx)]
            if w not in (a, b, c):
                return w, float(sims[idx])
        return None, 0.0
