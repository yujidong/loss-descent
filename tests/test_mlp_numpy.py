import numpy as np

from dlbook.nn.mlp_numpy import MLPNumpy


def _numeric_grads(model, X, Y, h=1e-6):
    grads = []
    for Wi in model.Ws:
        g = np.zeros_like(Wi)
        it = np.nditer(Wi, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = Wi[idx]
            Wi[idx] = old + h
            lp = np.mean((model.forward(X) - Y) ** 2)
            Wi[idx] = old - h
            lm = np.mean((model.forward(X) - Y) ** 2)
            Wi[idx] = old
            g[idx] = (lp - lm) / (2 * h)
            it.iternext()
        grads.append(g)
    return grads


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(1)
    model = MLPNumpy(2, [3, 1], activation="tanh", seed=2)
    X = rng.normal(size=(7, 2))
    Y = rng.normal(size=(7, 1))
    model.backward(X, Y)
    for g_num, g_auto in zip(_numeric_grads(model, X, Y), model.grad_Ws):
        assert np.allclose(g_num, g_auto, atol=1e-5)


def test_fits_smooth_function():
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    Y = np.sin(2 * X + 0.7) + 0.5 * np.sin(4 * X - 0.3)  # 相移避免对称鞍点
    model = MLPNumpy(1, [30, 1], seed=42)
    for _ in range(8000):
        model.backward(X, Y)
        model.step(0.05, momentum=0.9)
    assert model.backward(X, Y) < 1e-2


def test_grad_norms_shape():
    model = MLPNumpy(1, [8, 8, 1], seed=0)
    model.backward([[0.5]], [[1.0]])
    assert len(model.grad_norms()) == 3
