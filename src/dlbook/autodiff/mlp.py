"""用 Value 组装多层感知机：Neuron → Layer → MLP。

结构贴近 1986 年 Nature 论文的 Fig. 2：全连接层 + sigmoid 族激活 + 平方误差。
激活可选（tanh/relu/sigmoid/linear），输出层可单独指定——
回归任务用 linear 输出，分类任务沿用 tanh。
"""
from __future__ import annotations

import math
import random

from dlbook.autodiff.scalar import Value

ACTIVATIONS = {
    "tanh": lambda v: v.tanh(),
    "relu": lambda v: v.relu(),
    "sigmoid": lambda v: v.sigmoid(),
    "linear": lambda v: v,  # 恒等：用于回归输出层
}


class Neuron:
    def __init__(self, nin: int, activation: str = "tanh"):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum(w * xi for w, xi in zip(self.w, x)) + self.b
        return ACTIVATIONS[self.activation](act)

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, activation: str = "tanh"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """多层感知机：MLP(2, [4, 1]) 即 2-4-1 网络。

    hidden 激活由 activation 指定；output_activation 为 None 时
    输出层沿用同一激活（1.3 章 XOR 的 ±1 目标即如此），回归任务
    可显式传 "linear"。
    """

    def __init__(
        self,
        nin: int,
        nouts: list[int],
        activation: str = "tanh",
        output_activation: str | None = None,
    ):
        sizes = [nin] + list(nouts)
        acts = [activation] * (len(sizes) - 2) + [output_activation or activation]
        self.layers = [Layer(a, b, act) for a, b, act in zip(sizes, sizes[1:], acts)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def grad_norms(self):
        """各隐藏/输出层的参数梯度 L2 范数（backward() 之后调用）。

        1.6 章用它实测梯度消失，2.2 章将展开完整分析。
        """
        return [
            math.sqrt(sum(p.grad**2 for p in layer.parameters()))
            for layer in self.layers
        ]


def zero_grad(params: list[Value]) -> None:
    for p in params:
        p.grad = 0.0


def sgd_step(params: list[Value], lr: float) -> None:
    """最朴素的梯度下降：沿负梯度走一小步。"""
    for p in params:
        p.data -= lr * p.grad
