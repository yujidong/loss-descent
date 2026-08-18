#!/usr/bin/env python3
"""为所有无链接的参考文献条目添加可点击的搜索链接。

三种链接策略（按优先级）：
1. 已有 DOI → 保留
2. 已知 arXiv ID → 添加 arXiv 链接
3. 其余 → 添加 Google Scholar 搜索链接

用法：python scripts/link_refs.py
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]

# 已知 arXiv ID 的论文映射
ARXIV_IDS = {
    "he2015resnet": "1512.03385",
    "kingma2014": "1412.6980",
    "vaswani2017": "1706.03762",
    "kaplan2020": "2001.08361",
    "hoffmann2022": "2203.15556",
    "touvron2023": "2302.13971",
    "hu2021lora": "2106.09685",
    "schaeffer2023": "2304.15004",
    "hestness2017": "1712.00409",
    "oord2016": "1609.03499",
    "wei2022emergent": None,  # TMLR，无 arXiv preprint 为正式版
    "radford2018": None,  # OpenAI 报告，无 arXiv
    "radford2019": None,  # OpenAI 报告，无 arXiv
}


def add_links_to_chapter(path: Path) -> tuple[int, int]:
    """给一章的参考文献添加链接。返回 (总数, 新增链接数)。"""
    text = path.read_text(encoding="utf-8")

    # 找参考文献节（在 ## 参考文献 和 ## 练习 之间）
    ref_match = re.search(r'(## 参考文献\n)(.*?)(\n## 练习)', text, re.DOTALL)
    if not ref_match:
        return 0, 0

    ref_body = ref_match.group(2)
    lines = ref_body.split("\n")
    total = 0
    linked = 0

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or not re.match(r'^\d+\.', stripped):
            new_lines.append(line)
            continue

        total += 1

        # 已有链接？
        if "](https://" in line:
            linked += 1
            new_lines.append(line)
            continue

        # 提取标题（*斜体* 中的第一段通常是要点）
        title_match = re.search(r'\*([^*]+)\*', stripped)
        title = title_match.group(1) if title_match else ""

        # 提取作者和年份
        year_match = re.search(r'\((\d{4})\)', stripped)
        year = year_match.group(1) if year_match else ""
        author_match = re.match(r'\d+\.\s+([A-Z][^,(]+)', stripped)
        author = author_match.group(1).strip() if author_match else ""

        # 构建搜索查询
        query_parts = []
        if author:
            # 取第一作者的姓
            surname = author.split(",")[0].split()[-1]
            query_parts.append(surname)
        if year:
            query_parts.append(year)
        if title:
            # 取标题的前几个词
            words = title.split()[:6]
            query_parts.append(" ".join(words))

        query = " ".join(query_parts)
        if query:
            scholar_url = f"https://scholar.google.com/scholar?q={quote(query)}"
            # 在行末添加链接
            line = line.rstrip()
            if line.endswith("."):
                line = line[:-1]  # 去掉末尾句号
            line += f". [↗ 搜索原文]({scholar_url})"
            linked += 1

        new_lines.append(line)

    new_ref_body = "\n".join(new_lines)
    new_text = text[:ref_match.start(2)] + new_ref_body + text[ref_match.end(2):]
    path.write_text(new_text, encoding="utf-8")

    return total, linked


def main():
    grand_total = 0
    grand_linked = 0
    grand_new = 0

    for vol in ["vol1", "vol2", "vol3"]:
        for qmd in sorted((ROOT / "chapters" / vol).glob("*.qmd")):
            stem = qmd.stem
            total, linked = add_links_to_chapter(qmd)
            new = linked  # 简化：链接过的都算
            grand_total += total
            grand_linked += linked
            if total > 0:
                print(f"  {stem}: {linked}/{total} 条有链接")

    print(f"\n全书: {grand_linked}/{grand_total} 条引用有可点击链接")


if __name__ == "__main__":
    main()
