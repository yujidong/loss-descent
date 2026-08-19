#!/usr/bin/env python3
"""为各章的 ## 节末尾插入「缓一缓」复述段落（改进版）。

改进：正确处理 Quarto callout 语法（::: 开/闭），只在安全的
段落边界插入。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RECAPS = [
    "让我们把刚才的内容用一句话再说一遍。",
    "到这里停一停，把刚走过的路理一理。",
    "回顾一下这一节讲了什么。",
    "把刚才的推导或实验消化一下再继续。",
]

SKIP_SECTIONS = {"练习", "参考文献", "本章 Loss 账本", "本章问题"}


def find_insert_points(text: str) -> list[tuple[int, str]]:
    """找到每个 ## 节末尾的安全插入点。返回 (位置, 节标题) 列表。"""
    lines = text.split('\n')
    insert_points = []
    in_code = False
    callout_depth = 0
    current_heading = None
    last_safe_line = -1  # 最后一个可以插入的行号

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 代码块跟踪
        if stripped.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue

        # Callout 跟踪（::: 开，::: 关）
        if stripped == ':::':
            if callout_depth > 0:
                callout_depth -= 1
            else:
                callout_depth += 1
            continue
        if stripped.startswith('::: ') or stripped.startswith(':::{'):
            callout_depth += 1
            continue
        if callout_depth > 0:
            continue

        # 标题检测
        if stripped.startswith('## '):
            # 前一个节如果没有缓一缓，记录插入点
            if (current_heading and
                current_heading not in SKIP_SECTIONS and
                last_safe_line >= 0):
                insert_points.append((last_safe_line, current_heading))
            current_heading = stripped[3:].strip()  # 去掉 "## "
            last_safe_line = -1  # 重置
            continue

        # 非空、非 callout、非代码的行 = 安全插入点
        if stripped and not stripped.startswith('#'):
            last_safe_line = i

    # 最后一节
    if (current_heading and
        current_heading not in SKIP_SECTIONS and
        last_safe_line >= 0):
        insert_points.append((last_safe_line, current_heading))

    return insert_points


def add_recaps(path: Path) -> int:
    text = path.read_text(encoding="utf-8")

    # 检查是否已有缓一缓
    if '缓一缓' in text:
        return 0

    insert_points = find_insert_points(text)
    if not insert_points:
        return 0

    lines = text.split('\n')
    # 从后往前插入（避免行号偏移）
    added = 0
    for line_num, heading in reversed(insert_points):
        topic = re.sub(r'\d+\.\d+\s*', '', heading)
        template = RECAPS[added % len(RECAPS)]
        recap = f"\n*缓一缓。*{template}这一节的核心是：{topic}。\n"
        lines.insert(line_num + 1, recap)
        added += 1

    if added > 0:
        path.write_text('\n'.join(lines), encoding="utf-8")
    return added


def main():
    total = 0
    for vol in ["vol1", "vol2", "vol3"]:
        for qmd in sorted((ROOT / "chapters" / vol).glob("*.qmd")):
            stem = qmd.stem
            added = add_recaps(qmd)
            if added > 0:
                print(f"  {stem}: +{added}")
                total += added
    print(f"\n总计新增 {total} 个缓一缓段落")


if __name__ == "__main__":
    main()
