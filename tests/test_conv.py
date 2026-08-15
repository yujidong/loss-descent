"""Part 3 卷积部件测试。"""
import numpy as np

from dlbook.data import make_seq_task, make_shifted_pattern
from dlbook.nn.conv import Conv2D, SimpleConvNet


def test_conv_forward_shape_and_padding():
    conv = Conv2D(out_ch=4, kernel=3, padding=1, seed=0)
    X = np.random.default_rng(0).normal(size=(5, 1, 8, 8))
    Z = conv.forward(X)
    assert Z.shape == (5, 4, 8, 8)  # padding=1 保持尺寸


def test_conv_weight_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 1, 6, 6))
    conv = Conv2D(out_ch=3, kernel=3, seed=2)

    def loss():
        Z = conv.forward(X)
        return float((Z * np.arange(Z.size).reshape(Z.shape) % 7 - 3.0 / 7).sum() * 0 + (Z**2).mean())

    dZ_scale = rng.normal(size=(6, 3, 4, 4))
    conv.forward(X)
    conv.backward(dZ_scale)  # 填充 grad_W/grad_b

    h = 1e-6
    flat = conv.W.ravel()
    for idx in (0, 5, 17, conv.W.size - 1):
        old = flat[idx]
        flat[idx] = old + h
        lp = float((conv.forward(X) * dZ_scale).sum())
        flat[idx] = old - h
        lm = float((conv.forward(X) * dZ_scale).sum())
        flat[idx] = old
        num = (lp - lm) / (2 * h)
        auto = conv.grad_W.ravel()[idx]
        assert abs(num - auto) < 1e-4, f"W[{idx}]: {num} vs {auto}"


def test_simple_conv_net_learns_shifted_pattern():
    Xtr, ytr = make_shifted_pattern(n=300, noise=0.15, seed=0)
    Xte, yte = make_shifted_pattern(n=300, noise=0.15, seed=1)
    net = SimpleConvNet((12, 12), out_ch=8, kernel=3, padding=1, hidden=16, seed=0, pool=True)
    rng = np.random.default_rng(0)
    first = None
    for _ in range(80):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 32):
            i = order[s : s + 32]
            l = net.backward(Xtr[i], ytr[i])
            if first is None:
                first = l
            net.step(0.03, momentum=0.9)
    acc = float(np.mean((net.forward(Xte).ravel() > 0) == (yte > 0)))
    assert l < first and acc > 0.95, (first, l, acc)


def test_seq_task_shape():
    X, y = make_seq_task(n=50, length=16, dist=8, seed=0)
    assert X.shape == (50, 16) and set(np.unique(y)) <= {-1.0, 1.0}
    # 标签确为 dist 步之前的比特
    assert np.array_equal((y > 0).astype(int), X[:, 16 - 8].astype(int))
