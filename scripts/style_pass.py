#!/usr/bin/env python3
"""v1.1 文风修订的机械化 pass：对全部章节应用可自动化的样式修正。

不做叙事重写（那需要逐章手工），只做四件可靠的事：
1. 减加粗密度（每段至多保留一处）
2. 去掉正文中的全角惊叹号
3. 在每个 ## 节的末尾插入「缓一缓」段落（如果还没有）
4. 删除"这正是全书的""教科书级""神来之笔"等评价性元话语
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 节末小结模板（根据节标题关键词选择合适的复述句式）
RECAP_TEMPLATES = [
    "让我们把这一节的内容用一句话再说一遍：{topic}。",
    "回顾一下刚才的内容：{topic}。",
    "到这里停一停。这一节讲的是{topic}。",
    "把刚走过的路理一理：{topic}。",
]

# 需要删除或替换的评价性词语
EVALUATIVE_REPLACEMENTS = [
    ("这正是全书的题眼", "这一点贯穿全书"),
    ("这正是全书的", "这是"),
    ("教科书级的", ""),
    ("神来之笔", "巧妙的设计"),
    ("戏剧化的", "显著的"),
    ("完美复刻", "复现"),
    ("辉煌与失败", "兴衰"),
    ("神作", "重要工作"),
    ("堪称", "可算"),
    ("全场震惊", "引起广泛注意"),
]


def count_bold(text: str) -> int:
    return len(re.findall(r'\*\*[^*]+\*\*', text))


def reduce_bold(text: str) -> str:
    """每段至多保留一处加粗（第一个），其余去掉。"""
    paragraphs = text.split('\n\n')
    result = []
    for para in paragraphs:
        bolds = list(re.finditer(r'\*\*([^*]+)\*\*', para))
        if len(bolds) <= 1:
            result.append(para)
            continue
        # 保留第一个，去掉其余
        new_para = para
        for m in bolds[1:]:
            old = f"**{m.group(1)}**"
            new_para = new_para.replace(old, m.group(1), 1)
        result.append(new_para)
    return '\n\n'.join(result)


def remove_exclamations(text: str) -> str:
    """去掉正文惊叹号（保留代码块和 callout 中的）。"""
    # 分离代码块
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):  # 只处理非代码块部分
        # 保留 callout 标题行
        lines = parts[i].split('\n')
        for j, line in enumerate(lines):
            if line.strip().startswith(':::') or line.strip().startswith('title='):
                continue
            lines[j] = line.replace('！', '。').replace('!', '.')
        parts[i] = '\n'.join(lines)
    return ''.join(parts)


def clean_evaluative(text: str) -> str:
    for old, new in EVALUATIVE_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def extract_topic(heading: str) -> str:
    """从节标题提取主题短语。"""
    # 去掉 markdown 标记和编号
    topic = re.sub(r'^#+\s*', '', heading)
    topic = re.sub(r'\d+\.\d+\s*', '', topic)
    # 截短
    if len(topic) > 30:
        topic = topic[:28] + "…"
    return topic


def add_recaps(text: str) -> int:
    """在每个 ## 节的末尾（下一个 ## 之前）插入缓一缓段落。"""
    lines = text.split('\n')
    output = []
    current_section = None
    added = 0
    in_code = False
    in_callout = 0

    for i, line in enumerate(lines):
        # 跟踪代码块
        if line.strip().startswith('```'):
            in_code = not in_code

        # 跟踪 callout
        if line.strip().startswith(':::'):
            if not line.strip().startswith('::: '):  # 开始
                in_callout += 1
            else:  # 结束
                in_callout = max(0, in_callout - 1)

        # 检测新节
        if not in_code and in_callout == 0 and line.startswith('## '):
            # 如果前一个节还没有缓一缓，且不是特殊节，插入
            if (current_section and
                current_section not in ('练习', '参考文献', '本章 Loss 账本') and
                not any('缓一缓' in output[k] for k in range(max(0, len(output)-5), len(output)))):
                # 找最后一个非空行
                last_content = -1
                for k in range(len(output) - 1, max(0, len(output) - 10), -1):
                    if output[k].strip():
                        last_content = k
                        break
                if last_content >= 0:
                    topic = extract_topic(current_section)
                    template = RECAP_TEMPLATES[added % len(RECAP_TEMPLATES)]
                    recap = f"\n*缓一缓。*{template.format(topic=topic)}\n"
                    output.insert(last_content + 1, recap)
                    added += 1

            current_section = line.strip()

        output.append(line)

    # 最后一节
    if (current_section and
        current_section not in ('练习', '参考文献', '本章 Loss 账本') and
        not any('缓一缓' in output[k] for k in range(max(0, len(output)-5), len(output)))):
        topic = extract_topic(current_section)
        template = RECAP_TEMPLATES[added % len(RECAP_TEMPLATES)]
        recap = f"\n*缓一缓。*{template.format(topic=topic)}\n"
        output.append(recap)
        added += 1

    return added, '\n'.join(output)


def process_chapter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    original = text

    stats = {
        "bold_before": count_bold(text),
        "exclamations": text.count('！') + text.count('!'),
        "recaps_added": 0,
    }

    # 1. 清理评价性词语
    text = clean_evaluative(text)

    # 2. 去掉惊叹号（代码块外）
    text = remove_exclamations(text)

    # 3. 减加粗
    text = reduce_bold(text)

    # 4. 加缓一缓
    recaps, text = add_recaps(text)
    stats["recaps_added"] = recaps

    stats["bold_after"] = count_bold(text)

    if text != original:
        path.write_text(text, encoding="utf-8")

    return stats


def main():
    grand = {"chapters": 0, "bold_before": 0, "bold_after": 0, "excl": 0, "recaps": 0}

    for vol in ["vol1", "vol2", "vol3"]:
        for qmd in sorted((ROOT / "chapters" / vol).glob("*.qmd")):
            stem = qmd.stem
            stats = process_chapter(qmd)
            grand["chapters"] += 1
            grand["bold_before"] += stats["bold_before"]
            grand["bold_after"] += stats["bold_after"]
            grand["excl"] += stats["exclamations"]
            grand["recaps"] += stats["recaps_added"]

            if stats["bold_before"] != stats["bold_after"] or stats["recaps_added"] > 0:
                print(f"  {stem}: 加粗 {stats['bold_before']}→{stats['bold_after']}, "
                      f"缓一缓 +{stats['recaps_added']}")

    print(f"\n处理 {grand['chapters']} 章")
    print(f"加粗: {grand['bold_before']} → {grand['bold_after']} (减少 {grand['bold_before']-grand['bold_after']})")
    print(f"缓一缓段落: +{grand['recaps']}")


if __name__ == "__main__":
    main()
