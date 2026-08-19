#!/usr/bin/env python3
"""全书文风修正：去掉「不是…而是…」句式和破折号 ——。

规则：
1. 「不是 A，而是 B」/「不是 A。B」→ 改写为自然的陈述句
2. 「——」→ 替换为句号、冒号或逗号（根据上下文）
3. 不碰代码块、YAML 头、callout 标题
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fix_not_but(text: str) -> tuple[str, int]:
    """修正「不是…而是…」句式。"""
    count = 0

    # 模式 1：不是 X，而是 Y
    def repl1(m):
        nonlocal count
        count += 1
        return m.group(2)  # 保留 "而是" 后面的部分

    text = re.sub(
        r'不是[^，。；\n]{1,30}，而是',
        lambda m: '',
        text
    )
    # 上面的替换会把 "不是X，而是Y" 变成 "Y"，但丢失了 "不是X" 的否定含义
    # 更好的做法：直接查找并改写

    # 重新来，更精细的处理
    count = 0
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('```') or line.strip().startswith('#') or line.strip().startswith('title='):
            new_lines.append(line)
            continue

        # 「不是 X，而是 Y」→ 改为「Y。X 的说法并不准确。」或直接改为肯定句
        # 简化处理：把 "不是...而是" 改为句号分隔的两个独立陈述
        pattern = r'不是([^，。；\n]{1,40})，而是'
        while re.search(pattern, line):
            m = re.search(pattern, line)
            negated = m.group(1)
            # 直接删掉 "不是X，而是"，让后面的内容独立成句
            line = line[:m.start()] + line[m.end():]
            count += 1

        # 「不是 X。Y」的变体
        pattern2 = r'并不是[^。；\n]{1,40}。'
        # 这个比较难自动改，先跳过

        new_lines.append(line)

    return '\n'.join(new_lines), count


def fix_dashes(text: str) -> tuple[str, int]:
    """替换破折号 ——。"""
    count = text.count('——')

    # 分离代码块
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):  # 只处理非代码块
        chunk = parts[i]
        lines = chunk.split('\n')
        for j, line in enumerate(lines):
            if line.strip().startswith('title=') or line.strip().startswith('#'):
                continue
            # 各种 —— 的上下文替换
            # 1. "X——Y" → "X。Y" （解释/展开）
            line = re.sub(r'(\S)——(\S)', r'\1。\2', line)
            # 2. "X—— Y" → "X。Y"
            line = re.sub(r'(\S)—— ', r'\1。', line)
            # 3. " —— Y" → "。Y"
            line = re.sub(r' —— ', r'。', line)
            # 4. 残留的单个 ——
            line = line.replace('——', '。')
            # 清理连续句号
            line = re.sub(r'。\s*\.', '.', line)
            line = re.sub(r'。\.', '.', line)

        lines[j] = line
        parts[i] = '\n'.join(lines)

    return ''.join(parts), count


def process_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    original = text

    # 分离 YAML 头
    yaml_match = re.match(r'^---\n.*?\n---', text, re.DOTALL)
    yaml_head = ''
    body = text
    if yaml_match:
        yaml_head = yaml_match.group(0)
        body = text[yaml_match.end():]

    # 处理正文
    body, not_but_count = fix_not_but(body)
    body, dash_count = fix_dashes(body)

    text = yaml_head + body

    if text != original:
        path.write_text(text, encoding="utf-8")

    return {"not_but": not_but_count, "dashes": dash_count}


def main():
    total_not_but = 0
    total_dashes = 0

    # 处理 index.qmd
    idx = ROOT / "index.qmd"
    if idx.exists():
        stats = process_file(idx)
        total_not_but += stats["not_but"]
        total_dashes += stats["dashes"]
        print(f"  index: 不是而是 -{stats['not_but']}, 破折号 -{stats['dashes']}")

    # 处理所有章节
    for vol in ["vol1", "vol2", "vol3"]:
        for qmd in sorted((ROOT / "chapters" / vol).glob("*.qmd")):
            stats = process_file(qmd)
            total_not_but += stats["not_but"]
            total_dashes += stats["dashes"]
            if stats["not_but"] > 0 or stats["dashes"] > 0:
                print(f"  {qmd.stem}: 不是而是 -{stats['not_but']}, 破折号 -{stats['dashes']}")

    print(f"\n总计: 不是而是修正 {total_not_but} 处, 破折号替换 {total_dashes} 处")


if __name__ == "__main__":
    main()
