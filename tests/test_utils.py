import random

import matplotlib.pyplot as plt
import numpy as np

from dlbook.data import XOR, XOR_LABELS
from dlbook.utils import plot_linear_boundary, plot_loss_curves, set_seed


def test_set_seed_makes_runs_reproducible():
    set_seed(42)
    a = (random.random(), float(np.random.randn()))
    set_seed(42)
    b = (random.random(), float(np.random.randn()))
    assert a == b


def test_plot_loss_curves_smoke():
    fig, ax = plot_loss_curves(
        {"train": [1.0, 0.5, 0.3], "val": [1.1, 0.6, 0.4]}
    )
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_linear_boundary_vertical_and_sloped():
    # 竖直边界（w[1]=0）与斜边界两条路径都能出图
    fig1, _ = plot_linear_boundary([-1.0, 0.0], 1.0, XOR, XOR_LABELS)
    plt.close(fig1)
    fig2, _ = plot_linear_boundary([1.0, 1.0], 0.0, XOR, XOR_LABELS)
    plt.close(fig2)
