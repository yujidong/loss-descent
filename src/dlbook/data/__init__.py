"""dlbook.data：随书实验数据集工具。"""
from dlbook.data.toys import (
    XOR,
    XOR_LABELS,
    make_blobs,
    make_moons,
    make_slab,
    make_spiral,
)

__all__ = [
    "XOR",
    "XOR_LABELS",
    "make_blobs",
    "make_slab",
    "make_moons",
    "make_spiral",
]
