"""dlbook：随《降 Loss 之路》逐章生长的深度学习迷你库。

每一章把新实现的模型/工具合入对应模块，后续章节直接 import 复用，
让代码库本身也重演一次历史。模块规划：

- dlbook.utils   种子、绘图等基础设施（Part 0）
- dlbook.autodiff  手写 autograd 引擎（1.3 反向传播）
- dlbook.nn      各时代的层与模型（Perceptron → MLP → CNN → RNN → Transformer）
- dlbook.data    随书实验数据集工具
- dlbook.train   训练循环（SGD → Adam → …）
"""
__version__ = "0.1.0"

from dlbook.utils import plot_loss_curves, set_seed

__all__ = ["set_seed", "plot_loss_curves", "__version__"]
