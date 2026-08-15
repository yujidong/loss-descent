"""测试环境统一使用无头 Agg 后端。

conftest 在测试模块导入前执行，先于 matplotlib 的首次导入生效。
（不能在 dlbook 包里强制后端——那会破坏 Quarto/Jupyter 的内联图形显示。）
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
