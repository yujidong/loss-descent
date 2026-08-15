import numpy as np

from dlbook.data import XOR, XOR_LABELS, make_blobs, make_slab
from dlbook.nn import Perceptron


def test_converges_on_linearly_separable_data():
    X, y = make_blobs(distance=2.0, seed=0)
    p = Perceptron(2)
    mistakes = p.train(X, y, epochs=100)
    assert mistakes[-1] == 0  # 提前停止：最后一个 epoch 零错误
    assert all(p(x) == t for x, t in zip(X, y))


def test_oscillates_forever_on_xor():
    p = Perceptron(2)
    mistakes = p.train(XOR, XOR_LABELS, epochs=50)
    assert len(mistakes) == 50  # 从未提前停止
    assert mistakes[-1] > 0  # 最后一轮仍有错分
    assert not all(p(x) == t for x, t in zip(XOR, XOR_LABELS))


def test_novikoff_square_law():
    """Novikoff 上界 (R/gamma)^2 的定性验证：margin 减半，错分次数约 ×4。"""
    wide = Perceptron(2).train(*make_slab(margin=1.0, seed=0), epochs=2000)
    tight = Perceptron(2).train(*make_slab(margin=0.25, seed=0), epochs=5000)
    assert wide[-1] == 0 and tight[-1] == 0
    assert sum(tight) > 3 * sum(wide)
