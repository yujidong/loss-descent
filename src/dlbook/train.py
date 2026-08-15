"""训练器：Part 2 中叶（2.5 章）提炼出的"让深度可训练"的标准流程。

小批量 SGD + 动量 + 每 epoch 的训练/验证记录。它没有魔法——
每一件装备都在前面各章被单独发明过，这里只是组装。
"""
from __future__ import annotations

import numpy as np


def accuracy(model, X, y) -> float:
    """二分类（±1）准确率。"""
    pred = np.asarray(model.forward(X)).ravel()
    y = np.asarray(y).ravel()
    return float(np.mean((pred > 0) == (y > 0)))


class Trainer:
    def __init__(self, model, lr: float = 0.05, momentum: float = 0.9,
                 batch_size: int = 32, seed: int = 0):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.batch_size, self._rng = batch_size, np.random.default_rng(seed)

    def _epoch_loss(self, X, Y):
        return float(np.mean((self.model.forward(X) - np.asarray(Y)) ** 2))

    def fit(self, X, Y, epochs: int, val=None) -> dict[str, list]:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        history: dict[str, list] = {"loss": [], "val_loss": []}
        n = len(X)
        for _ in range(epochs):
            order = self._rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s : s + self.batch_size]
                self.model.backward(X[idx], Y[idx])
                self.model.step(self.lr, momentum=self.momentum)
            history["loss"].append(self._epoch_loss(X, Y))
            history["val_loss"].append(
                self._epoch_loss(*val) if val is not None else float("nan")
            )
        return history
