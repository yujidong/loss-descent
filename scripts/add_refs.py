#!/usr/bin/env python3
"""为全部章节添加手动格式化的参考文献列表（带 DOI/arXiv 链接）。

从 references.bib 读取条目，为每章生成带链接的 markdown 引用列表，
插入到"## 练习"之前。不依赖 Quarto 的自动 bibliography 系统。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 手动维护每章引用的 key 列表（从论文时光机和正文中提取）
CHAPTER_REFS = {
    # Part 0
    "part0-two-threads": ["shannon1948", "sutton2019"],
    "part0-reading-history": ["kuhn1962", "hochreiter1991", "dauphin2014"],
    # Part 1
    "part1-perceptron": ["rosenblatt1958", "minsky1969", "widrow1960", "hebb1949", "mcculloch1943"],
    "part1-symbolic-vs-connectionist": ["hopfield1982", "fukushima1980", "rumelhart1986pdp", "minsky1969"],
    "part1-backprop": ["rumelhart1986", "werbos1974", "linnainmaa1970", "parker1985",
                       "minsky1969", "widrow1960", "rumelhart1986pdp", "cybenko1989", "hebb1949", "rosenblatt1958"],
    "part1-mlp-winter": ["cybenko1989", "hornik1991", "hochreiter1991", "bengio1994", "sutton2019"],
    # Part 2
    "part2-mlp-difficulty": ["glorot2010", "dauphin2014", "goodfellow2016", "sutton2019"],
    "part2-vanishing-gradient": ["hochreiter1991", "bengio1994", "glorot2010", "hornik1991"],
    "part2-toolbox": ["nair2010", "srivastava2014", "ioffe2015", "he2015resnet", "santurkar2018", "sutton2019"],
    "part2-alexnet": ["krizhevsky2012", "glorot2010", "nair2010", "srivastava2014", "sutton2019"],
    "part2-capstone-deep-mlp": ["kingma2014", "ioffe2015", "he2015resnet", "sutton2019"],
    # Part 3
    "part3-lenet": ["lecun1989", "lecun1998", "fukushima1980"],
    "part3-resnet": ["he2015resnet", "krizhevsky2012"],
    "part3-cnn-legacy": ["razavian2014", "zeiler2013", "lecun1998"],
    "part3-cnn-limits": ["oord2016", "sutton2019"],
    # Part 4
    "part4-ngram": ["shannon1951", "shannon1948", "jelinek1976"],
    "part4-rnn": ["elman1990", "bengio1994", "bengio2003", "hochreiter1997"],
    "part4-lstm": ["hochreiter1997", "gers2000", "cho2014", "bengio1994"],
    "part4-word2vec": ["mikolov2013", "levy2014", "pennington2014"],
    "part4-seq2seq": ["sutskever2014", "cho2014", "bahdanau2015"],
    # Part 5
    "part5-bahdanau": ["bahdanau2015", "cho2014"],
    "part5-transformer": ["vaswani2017", "kaplan2020"],
    "part5-why-transformer-won": ["bradbury2017", "vaswani2017"],
    "part5-gpt-vs-bert": ["radford2018", "devlin2019", "radford2019", "brown2020"],
    # Part 6
    "part6-scaling-laws": ["kaplan2020", "brown2020", "hestness2017"],
    "part6-chinchilla": ["hoffmann2022", "kaplan2020", "touvron2023"],
    "part6-emergence": ["wei2022emergent", "schaeffer2023"],
    "part6-convergence": ["fedus2022", "touvron2023", "sutton2019"],
    # Part 7
    "part7-base-model": ["brown2020", "olsson2022"],
    "part7-sft": ["ouyang2022", "wei2022flan", "sanh2022", "hu2021lora"],
    "part7-rlhf": ["rafailov2023", "ouyang2022", "christiano2017"],
    "part7-reasoning": ["zelikman2022", "ouyang2022"],
    # Part 8
    "part8-agent-origins": ["schick2023", "radford2019"],
    "part8-agent-training": ["yao2023", "zelikman2022"],
    "part8-model-vs-system": ["lewis2020", "radford2018"],
    # Part 9
    "part9-patterns": ["sutton2019", "kuhn1962"],
    "part9-open-problems": ["hestness2017", "kaplan2020", "schaeffer2023"],
    "part9-prediction": ["sutton2019", "kuhn1962", "shannon1948"],
}

# BibTeX 解析（简化版：提取 key -> {author, title, journal, year, volume, pages, doi, url})
def parse_bib(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    entries = {}
    for m in re.finditer(r'@(\w+)\{([^,]+),\s*\n(.*?)\n\}', text, re.DOTALL):
        entry_type, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}', body, re.DOTALL):
            fields[fm.group(1).lower()] = fm.group(2).strip().strip('{}')
        entries[key] = {"type": entry_type.lower(), **fields}
    return entries


def format_ref(key: str, entry: dict) -> str:
    """格式化单条引用为带链接的 markdown。"""
    authors = entry.get("author", "")
    year = entry.get("year", "n.d.")
    title = entry.get("title", entry.get("booktitle", ""))
    title = re.sub(r'[{}]', '', title)

    # 期刊/会议信息
    venue = entry.get("journal", entry.get("booktitle", entry.get("publisher", "")))
    venue = re.sub(r'[{}]', '', venue)
    vol = entry.get("volume", "")
    pages = entry.get("pages", "")

    # 构建引用文本
    parts = []
    if authors:
        # 截短过长的作者列表
        author_list = authors.split(" and ")
        if len(author_list) > 3:
            parts.append(f"{author_list[0].split(',')[0]} et al.")
        else:
            parts.append(re.sub(r'[{}]', '', authors))
    parts.append(f"({year}).")
    parts.append(f"*{title}*.")
    if venue:
        v = venue
        if vol:
            v += f", {vol}"
        if pages:
            v += f", {pages}"
        parts.append(f"*{v}*.")

    # 链接
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    if doi:
        parts.append(f"[doi:{doi}](https://doi.org/{doi})")
    elif url:
        parts.append(f"[link]({url})")
    elif "arxiv" in str(entry).lower():
        arxiv_id = re.search(r'arxiv[^"]*?(\d{4}\.\d{4,5})', str(entry))
        if arxiv_id:
            parts.append(f"[arXiv:{arxiv_id.group(1)}](https://arxiv.org/abs/{arxiv_id.group(1)})")

    return " ".join(parts)


def main():
    bib = parse_bib(ROOT / "references.bib")
    print(f"BibTeX 条目: {len(bib)}")

    total_chapters = 0
    total_refs = 0

    for vol in ["vol1", "vol2", "vol3"]:
        for qmd in sorted((ROOT / "chapters" / vol).glob("*.qmd")):
            stem = qmd.stem
            if stem not in CHAPTER_REFS:
                continue

            text = qmd.read_text(encoding="utf-8")

            # 跳过已有参考文献的（除 #refs 占位符外还有内容的）
            if "## 参考文献" in text and "#refs" not in text:
                continue

            # 生成引用列表
            refs = []
            for key in CHAPTER_REFS[stem]:
                if key in bib:
                    refs.append(format_ref(key, bib[key]))
                else:
                    print(f"  ⚠ {stem}: bib 中无 {key}")

            if not refs:
                continue

            # 构建参考文献节
            ref_section = "\n## 参考文献\n\n"
            for i, r in enumerate(refs, 1):
                ref_section += f"{i}. {r}\n"
            ref_section += "\n"

            # 替换旧的 #refs 占位符（如果有）
            text = re.sub(r'\n## 参考文献\n\n::: \{#refs\}\n:::\n', '\n', text)

            # 插入到 ## 练习 之前
            if "\n## 练习" in text:
                text = text.replace("\n## 练习", ref_section + "\n## 练习", 1)
            else:
                # 没有练习节就追加到文件末尾
                text = text.rstrip() + "\n" + ref_section

            qmd.write_text(text, encoding="utf-8")
            total_chapters += 1
            total_refs += len(refs)
            print(f"  ✓ {stem}: {len(refs)} 条")

    print(f"\n完成: {total_chapters} 章, {total_refs} 条引用")


if __name__ == "__main__":
    main()
