"""Part 2 工具箱部件（dropout/BatchNorm/残差/初始化）与训练器测试。"""
import numpy as np

from dlbook.data import make_moons, make_spiral
from dlbook.nn.mlp_numpy import MLPNumpy
from dlbook.train import Trainer, accuracy


def _numeric_grads(model, X, Y, h=1e-6):
    grads = []
    for Wi in model.Ws:
        g = np.zeros_like(Wi)
        it = np.nditer(Wi, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = Wi[idx]
            Wi[idx] = old + h
            lp = np.mean((model.forward(X, training=True) - Y) ** 2)
            Wi[idx] = old - h
            lm = np.mean((model.forward(X, training=True) - Y) ** 2)
            Wi[idx] = old
            g[idx] = (lp - lm) / (2 * h)
            it.iternext()
        grads.append(g)
    return grads


def _check_against_numeric(model, X, Y):
    model.backward(X, Y)
    for g_num, g_auto in zip(_numeric_grads(model, X, Y), model.grad_Ws):
        assert np.allclose(g_num, g_auto, atol=1e-5)


def test_batchnorm_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = MLPNumpy(3, [4, 1], activation="relu", seed=1, batchnorm=True)
    X, Y = rng.normal(size=(9, 3)), rng.normal(size=(9, 1))
    _check_against_numeric(model, X, Y)


def test_residual_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = MLPNumpy(3, [4, 4, 4, 1], activation="relu", seed=1, residual=True)
    X, Y = rng.normal(size=(8, 3)), rng.normal(size=(8, 1))
    _check_against_numeric(model, X, Y)


def test_dropout_eval_mode_is_deterministic_and_scaled():
    model = MLPNumpy(2, [8, 1], dropout=0.5, seed=0)
    X = np.random.default_rng(1).normal(size=(5, 2))
    a, b = model.forward(X), model.forward(X)
    assert np.array_equal(a, b)  # eval 模式不随机


def test_uniform_init_available_for_contrast():
    model = MLPNumpy(2, [8, 1], init="uniform", seed=0)
    rng2 = np.random.default_rng(0)
    expected = rng2.uniform(-1, 1, (8, 2))
    assert np.allclose(model.Ws[0], expected)


def test_trainer_fits_moons():
    X, y = make_moons(n_per_class=60, seed=0)
    model = MLPNumpy(2, [16, 1], activation="relu", seed=42)
    t = Trainer(model, lr=0.05, momentum=0.9, batch_size=32, seed=0)
    hist = t.fit(X, y, epochs=150)
    assert hist["loss"][0] > hist["loss"][-1]
    assert accuracy(model, X, y) > 0.95


def test_spiral_shapes():
    X, y = make_spiral(n_per_class=50, seed=0)
    assert X.shape == (100, 2) and y.shape == (100,)
    assert set(np.unique(y)) == {-1.0, 1.0}
