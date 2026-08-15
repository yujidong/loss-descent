"""用 Value 组装多层感知机：Neuron → Layer → MLP。

结构刻意贴近 1986 年 Nature 论文的 Fig. 2：
全连接层 + sigmoid 族激活（这里取 tanh）+ 平方误差。
"""
from __future__ import annotations

import random

from dlbook.autodiff.scalar import Value


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum(w * xi for w, xi in zip(self.w, x)) + self.b
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """多层感知机：MLP(2, [4, 1]) 即 2-4-1 网络。"""

    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + list(nouts)
        self.layers = [Layer(a, b) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def zero_grad(params: list[Value]) -> None:
    for p in params:
        p.grad = 0.0


def sgd_step(params: list[Value], lr: float) -> None:
    """最朴素的梯度下降：沿负梯度走一小步。"""
    for p in params:
        p.data -= lr * p.grad
