"""Rosenblatt 感知机（1958）：深度学习史的第一台"学习机器"。

规则是错误驱动的在线更新，而非显式 loss 的梯度下降——
这个差别正是 1.3 章（反向传播）要补上的视角。
"""
from __future__ import annotations

import numpy as np


class Perceptron:
    """单层阈值神经元：预测 y = sign(w·x + b)。"""

    def __init__(self, nin: int):
        self.w = np.zeros(nin)
        self.b = 0.0

    def __call__(self, x) -> float:
        return 1.0 if float(np.dot(self.w, x)) + self.b > 0 else -1.0

    def train(self, X, y, epochs: int = 100, lr: float = 1.0) -> list[int]:
        """Rosenblatt 学习规则：只在预测错误时更新 w += lr·y·x。

        返回每个 epoch 的错分数；若某 epoch 错分为 0，提前停止（已收敛）。
        数据线性不可分时永不收敛——调用方需自行判断列表是否提前截断。
        """
        X, y = np.asarray(X), np.asarray(y)
        mistakes_per_epoch: list[int] = []
        for _ in range(epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                if self(xi) != yi:
                    self.w += lr * yi * xi
                    self.b += lr * yi
                    mistakes += 1
            mistakes_per_epoch.append(mistakes)
            if mistakes == 0:
                break
        return mistakes_per_epoch
