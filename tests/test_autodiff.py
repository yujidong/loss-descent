"""dlbook.autodiff 的正确性测试。

test_backward_matches_finite_differences 是全书代码的信任基石：
任何手写求导都必须过有限差分这一关。
"""
import math

from dlbook.autodiff import MLP, Value, sgd_step, zero_grad
from dlbook.utils import set_seed


def _numeric_grad(f, xs, h=1e-6):
    """中心差分数值梯度，作为手写 backward 的裁判。"""
    grads = []
    for i in range(len(xs)):
        xp, xm = list(xs), list(xs)
        xp[i] += h
        xm[i] -= h
        grads.append((f(*xp) - f(*xm)) / (2 * h))
    return grads


def test_backward_matches_finite_differences():
    def f(a, b):
        return (a * b + math.tanh(a)) * b + a * a * 2.0

    x, y = Value(1.7), Value(-0.8)
    z = (x * y + x.tanh()) * y + x * x * 2.0
    z.backward()

    g = _numeric_grad(f, [1.7, -0.8])
    assert abs(x.grad - g[0]) < 1e-5
    assert abs(y.grad - g[1]) < 1e-5


def test_relu_and_exp_gradients():
    x = Value(0.3)
    y = x.relu() + x.exp()
    y.backward()

    def f(a):
        return max(0.0, a) + math.exp(a)

    assert abs(x.grad - _numeric_grad(f, [0.3])[0]) < 1e-5


def test_mlp_learns_xor():
    set_seed(42)
    model = MLP(2, [4, 1])
    xor = [((0.0, 0.0), -1.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), -1.0)]

    first = None
    for _ in range(400):
        loss = sum((model(list(xs)) - Value(t)) ** 2 for xs, t in xor) * (1.0 / 4)
        if first is None:
            first = loss.data
        zero_grad(model.parameters())
        loss.backward()
        sgd_step(model.parameters(), lr=0.1)
    assert first > 0.5
    assert loss.data < 0.02
    for xs, t in xor:
        assert model(list(xs)).data * t > 0
