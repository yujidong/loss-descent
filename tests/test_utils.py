import random

import matplotlib.pyplot as plt
import numpy as np

from dlbook.utils import plot_loss_curves, set_seed


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
