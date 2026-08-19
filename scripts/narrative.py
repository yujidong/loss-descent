#!/usr/bin/env python3
"""叙事优化 pass：让文字更流畅、更有叙述性。

核心策略：
1. 在密集的事实句之间插入过渡语（"值得注意的是""换句话说"等）
2. 把「弯路一/二」的标签式段落改为自然段落
3. 展开过于压缩的表述
4. 在实验描述前后添加铺垫句和回顾句
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 过渡语库（随机选择避免重复）
TRANSITIONS = [
    "值得注意的是，",
    "换句话说，",
    "这意味着",
    "从这个角度看，",
    "这里需要解释一下。",
    "理解这一点的关键是：",
    "让我们放慢一点来看。",
    "这个细节很重要。",
]

# 标签式段落的转换
LABEL_PATTERNS = [
    (r'^\*\*弯路一[：:]\*\*\s*', "第一个弯路来自这样的尝试："),
    (r'^\*\*弯路二[：:]\*\*\s*', "另一个弯路是"),
    (r'^\*\*弯路三[：:]\*\*\s*', "还有一条弯路："),
    (r'^\*\*弯路[：:]\*\*\s*', "一条弯路是"),
    (r'^\*\*正解[：:]\*\*\s*', "真正的解法是这样的："),
    (r'^\*\*诊断[：:]\*\*\s*', "诊断的结果是"),
    (r'^\*\*药方[：:]\*\*\s*', "解决方案是"),
    (r'^\*\*后续的理解[：:]\*\*\s*', "后来人们逐渐理解到"),
    (r'^\*\*反讽的是[：:]\*\*\s*', "有讽刺意味的是"),
]

# 实验前后的铺垫/回顾
EXPERIMENT_LEADINS = [
    "在运行实验之前，先说清楚我们要观察什么。",
    "下面这个实验能帮我们看清这一点。",
    "让我们用代码来验证。",
    "来看具体的数字。",
]

EXPERIMENT_WRAPUPS = [
    "这个结果值得停下来想一想。",
    "把这些数字放在一起看，规律就浮现出来了。",
    "如果暂时没跟上，没关系，下一段会换个角度再讲。",
]


def add_transitions(text: str) -> tuple[str, int]:
    """在连续的短事实句之间添加过渡语。"""
    lines = text.split('\n')
    count = 0
    transition_idx = 0

    for i, line in enumerate(lines):
        s = line.strip()
        # 跳过代码、标题、callout、列表
        if (s.startswith('```') or s.startswith('#') or s.startswith(':::') or
            s.startswith('- ') or s.startswith('* ') or s.startswith('1.') or
            s.startswith('title=') or not s):
            continue

        # 检测连续短句模式（3 句以上连续的事实句，每句 < 80 字符）
        # 这种模式通常表示密集的信息投送
        # 我们在第二句和第三句之间插入过渡语

        # 简化策略：在每个 ## 节的第 2-3 个段落后添加过渡
        # 这需要更复杂的逻辑，先用轻量方式

        pass

    return text, count


def convert_labels(text: str) -> tuple[str, int]:
    """把标签式段落改为自然段落。"""
    count = 0
    for pattern, replacement in LABEL_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
            count += len(matches)
    return text, count


def soften_compact_prose(text: str) -> tuple[str, int]:
    """展开过于压缩的表述。"""
    count = 0

    # "它是X" → "它的本质是X" / "它可以理解为X"
    replacements = [
        # 简短定义句展开
        (r'^(\*\*[^*]+\*\* 是)', r'\1可以理解为'),
        # "这意味着X" 已经是过渡语，跳过
        # 压缩的因果关系展开
        (r'因此([^。]{5,40})。', r'因此，\1。'),
        (r'所以([^。]{5,40})。', r'所以，\1。'),
        (r'但([^。]{3,30})。', r'但是，\1。'),
    ]

    lines = text.split('\n')
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('```') or s.startswith('#'):
            continue
        original = line
        for old_pattern, new_pattern in replacements:
            new_line = re.sub(old_pattern, new_pattern, line)
            if new_line != line:
                count += 1
                line = new_line
        lines[i] = line

    return '\n'.join(lines), count


def add_breathing_room(text: str) -> tuple[str, int]:
    """在密集段落之间插入空行（分段）。"""
    count = 0
    paragraphs = text.split('\n\n')

    result = []
    for i, para in enumerate(paragraphs):
        s = para.strip()
        # 跳过代码块、标题、callout
        if s.startswith('```') or s.startswith('#') or s.startswith(':::'):
            result.append(para)
            continue

        # 如果段落超过 6 行，考虑在中间分段
        lines = para.split('\n')
        if len(lines) > 6:
            # 找到最合适的分段点（句子结束处）
            mid = len(lines) // 2
            for j in range(mid, len(lines)):
                if lines[j].rstrip().endswith(('。', '）', '」')):
                    # 在这里分段
                    part1 = '\n'.join(lines[:j+1])
                    part2 = '\n'.join(lines[j+1:])
                    if part2.strip():
                        result.append(part1)
                        result.append('')
                        result.append(part2)
                        count += 1
                    else:
                        result.append(para)
                    break
            else:
                result.append(para)
        else:
            result.append(para)

    return '\n\n'.join(result), count


def process_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    # 分离 YAML 头
    yaml_match = re.match(r'^(---\n.*?\n---)(.*)$', text, re.DOTALL)
    if yaml_match:
        yaml_head, body = yaml_match.group(1), yaml_match.group(2)
    else:
        yaml_head, body = '', text

    stats = {"labels": 0, "soften": 0, "breathing": 0}

    # 1. 标签式段落转自然段
    body, n = convert_labels(body)
    stats["labels"] = n

    # 2. 展开压缩表述
    body, n = soften_compact_prose(body)
    stats["soften"] = n

    # 3. 长段分段（增加呼吸感）
    body, n = add_breathing_room(body)
    stats["breathing"] = n

    text = yaml_head + body
    path.write_text(text, encoding="utf-8")
    return stats


def main():
    total = {"labels": 0, "soften": 0, "breathing": 0}

    # 处理 index
    idx = ROOT / "index.qmd"
    if idx.exists():
        stats = process_file(idx)
        for k in total:
            total[k] += stats[k]

    # 处理所有章节
    for vol in ["vol1", "vol2", "vol3"]:
        for qmd in sorted((ROOT / "chapters" / vol).glob("*.qmd")):
            stats = process_file(qmd)
            for k in total:
                total[k] += stats[k]
            if any(v > 0 for v in stats.values()):
                print(f"  {qmd.stem}: 标签→{stats['labels']}, 展开→{stats['soften']}, 分段→{stats['breathing']}")

    print(f"\n总计: 标签转换 {total['labels']}, 压缩展开 {total['soften']}, 增加分段 {total['breathing']}")


if __name__ == "__main__":
    main()
