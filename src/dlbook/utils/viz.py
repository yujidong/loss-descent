"""全书统一的实验图表。

标签默认用英文，避免 matplotlib 中文字体缺失时出现方块；
如需中文标签，请先配置系统字体后再覆盖。

注意：这里不强制选择后端——Jupyter 内联显示依赖 inline 后端，
而无头环境（CI/服务器）下 matplotlib 会自动回退到 Agg。
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_loss_curves(
    history: dict[str, list[float]],
    *,
    title: str = "Training progress",
    xlabel: str = "step",
    ylabel: str = "loss",
    logy: bool = True,
    save_to: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """绘制 loss 曲线。

    history: ``{label: [loss0, loss1, ...]}``，例如 ``{"train": [...], "val": [...]}``。
    返回 ``(fig, ax)``；``save_to`` 给定时另存为图片文件。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in history.items():
        ax.plot(values, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    return fig, ax


def plot_linear_boundary(w, b, X, y, *, title="Linear boundary"):
    """绘制二维线性分类器 sign(w·x + b) 的决策边界与数据点。

    w: 长度 2 的法向量；X: (n, 2)；y: ±1 标签。兼容 w[1]=0 的竖直边界。
    """
    import numpy as np

    w = np.asarray(w, dtype=float)
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c="tab:red", marker="o", label="+1")
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c="tab:blue", marker="s", label="-1")
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 50)
    if abs(w[1]) > 1e-12:
        ax.plot(x1, -(w[0] * x1 + b) / w[1], "k--", label="boundary")
    else:
        ax.axvline(-b / w[0], color="k", ls="--", label="boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
