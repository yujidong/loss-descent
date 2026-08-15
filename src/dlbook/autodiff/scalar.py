"""标量反向传播引擎（micrograd 风格）。

1986 年的论文把链式法则组织成了算法；本模块把它组织成数据结构：
每个 Value 是计算图上的一个节点，前向计算时记录子节点与局部算子，
backward() 时按拓扑逆序累加梯度。全书后续的每一块积木都从这里长出来。
"""
from __future__ import annotations

import math


class Value:
    """可自动求导的标量：data 是值，grad 是 loss 对它的偏导。"""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op  # 生成该节点的算子，仅用于调试与可视化

    # ---- 算术：每个算子做两件事——前向算值，登记反向的局部梯度 ----

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: int | float):
        assert isinstance(power, (int, float)), "只支持常数幂（链式法则不需要穿过指数）"
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1.0

    def __rtruediv__(self, other):
        return Value(other) * self**-1.0

    # ---- 激活与超越函数 ----

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    # ---- 发动机本体 ----

    def backward(self):
        """反向传播：拓扑排序 + 逆序传播梯度。

        复杂度与前向同阶——「求全部梯度」与「求一个梯度」同价，
        这正是反向模式自动微分对多参数单输出问题的胜利。
        """
        topo: list[Value] = []
        visited: set[int] = set()

        def build(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
