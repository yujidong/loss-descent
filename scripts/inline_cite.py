#!/usr/bin/env python3
"""为剩余 32 章添加内联 [@key] 引用标记。

策略：在正文中搜索关键作者名+年份的组合，在其后插入 [@key]。
已有引用的章节跳过。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 每章要添加的引用：(搜索模式, bib key)
CITE_MAPS = {
    # Part 2
    "part2-mlp-difficulty": [
        (r"Glorot.*Bengio.*2010", "glorot2010"),
        (r"Dauphin.*2014", "dauphin2014"),
        (r"Goodfellow.*2015", "dauphin2014"),  # landscape 论文
    ],
    "part2-vanishing-gradient": [
        (r"Hochreiter.*1991", "hochreiter1991"),
        (r"Bengio.*1994", "bengio1994"),
        (r"Glorot.*Bengio.*2010", "glorot2010"),
    ],
    "part2-toolbox": [
        (r"Nair.*Hinton.*2010", "nair2010"),
        (r"Hinton.*2012|Srivastava.*2014", "srivastava2014"),
        (r"Ioffe.*Szegedy.*2015", "ioffe2015"),
        (r"He.*2015", "he2015resnet"),
        (r"Santurkar.*2018", "santurkar2018"),
    ],
    "part2-alexnet": [
        (r"Krizhevsky.*2012|AlexNet", "krizhevsky2012"),
        (r"Glorot.*Bengio.*2010", "glorot2010"),
        (r"Sutton.*Bitter Lesson|Bitter Lesson", "sutton2019"),
    ],
    "part2-capstone-deep-mlp": [
        (r"Kingma.*Ba.*2014", "kingma2014"),
        (r"Ioffe.*Szegedy.*2015", "ioffe2015"),
    ],
    # Part 3
    "part3-lenet": [
        (r"LeCun.*1989", "lecun1989"),
        (r"LeCun.*1998|LeNet-5", "lecun1998"),
        (r"Fukushima.*1980|Neocognitron", "fukushima1980"),
    ],
    "part3-resnet": [
        (r"He.*2015|ResNet", "he2015resnet"),
        (r"Krizhevsky.*2012", "krizhevsky2012"),
    ],
    "part3-cnn-legacy": [
        (r"Zeiler.*Fergus.*2013", "zeiler2013"),
        (r"Razavian.*2014", "razavian2014"),
    ],
    "part3-cnn-limits": [
        (r"van den Oord.*2016|WaveNet", "oord2016"),
    ],
    # Part 4
    "part4-ngram": [
        (r"Shannon.*1948", "shannon1948"),
        (r"Shannon.*1951", "shannon1951"),
        (r"Jelinek", "jelinek1976"),
    ],
    "part4-rnn": [
        (r"Elman.*1990", "elman1990"),
        (r"Bengio.*1994", "bengio1994"),
        (r"Bengio.*2003", "bengio2003"),
    ],
    "part4-lstm": [
        (r"Hochreiter.*Schmidhuber.*1997", "hochreiter1997"),
        (r"Gers.*2000", "gers2000"),
        (r"Cho.*2014|GRU", "cho2014"),
    ],
    "part4-word2vec": [
        (r"Mikolov.*2013", "mikolov2013"),
        (r"Levy.*Goldberg.*2014", "levy2014"),
        (r"Pennington.*2014|GloVe", "pennington2014"),
    ],
    "part4-seq2seq": [
        (r"Sutskever.*2014", "sutskever2014"),
        (r"Cho.*2014", "cho2014"),
        (r"Bahdanau.*2015", "bahdanau2015"),
    ],
    # Part 5
    "part5-bahdanau": [
        (r"Bahdanau.*Cho.*Bengio|Bahdanau.*2015", "bahdanau2015"),
        (r"Cho.*2014", "cho2014"),
    ],
    "part5-transformer": [
        (r"Vaswani.*2017|Attention Is All You Need", "vaswani2017"),
    ],
    "part5-why-transformer-won": [
        (r"Bradbury.*2017|QRNN", "bradbury2017"),
        (r"Vaswani.*2017", "vaswani2017"),
    ],
    "part5-gpt-vs-bert": [
        (r"Radford.*2018|GPT-1", "radford2018"),
        (r"Devlin.*201[89]|BERT", "devlin2019"),
        (r"Radford.*2019|GPT-2", "radford2019"),
        (r"Brown.*2020|GPT-3", "brown2020"),
    ],
    # Part 6
    "part6-scaling-laws": [
        (r"Kaplan.*2020|Scaling Laws", "kaplan2020"),
        (r"Brown.*2020|GPT-3", "brown2020"),
    ],
    "part6-chinchilla": [
        (r"Hoffmann.*2022|Chinchilla", "hoffmann2022"),
        (r"Kaplan.*2020", "kaplan2020"),
        (r"Touvron.*2023|LLaMA", "touvron2023"),
    ],
    "part6-emergence": [
        (r"Wei.*2022|emergent", "wei2022emergent"),
        (r"Schaeffer.*2023|mirage", "schaeffer2023"),
    ],
    "part6-convergence": [
        (r"Fedus.*2022|Switch", "fedus2022"),
        (r"Touvron.*2023|LLaMA", "touvron2023"),
        (r"Sutton.*Bitter|Bitter Lesson", "sutton2019"),
    ],
    # Part 7
    "part7-base-model": [
        (r"Brown.*2020|GPT-3", "brown2020"),
        (r"Olsson.*2022|induction", "olsson2022"),
    ],
    "part7-sft": [
        (r"Ouyang.*2022|InstructGPT", "ouyang2022"),
        (r"Wei.*2022|FLAN", "wei2022flan"),
        (r"Hu.*2021|LoRA", "hu2021lora"),
    ],
    "part7-rlhf": [
        (r"Rafailov.*2023|DPO", "rafailov2023"),
        (r"Ouyang.*2022|InstructGPT", "ouyang2022"),
        (r"Christiano.*2017", "christiano2017"),
    ],
    "part7-reasoning": [
        (r"Zelikman.*2022|STaR", "zelikman2022"),
    ],
    # Part 8
    "part8-agent-origins": [
        (r"Schick.*2023|Toolformer", "schick2023"),
    ],
    "part8-agent-training": [
        (r"Yao.*2023|Tree of Thoughts", "yao2023"),
        (r"Zelikman.*2022|STaR", "zelikman2022"),
    ],
    "part8-model-vs-system": [
        (r"Lewis.*2020|RAG", "lewis2020"),
        (r"Radford.*2018|GPT", "radford2018"),
    ],
    # Part 9
    "part9-patterns": [
        (r"Sutton.*Bitter|Bitter Lesson", "sutton2019"),
        (r"Kuhn.*1962|范式", "kuhn1962"),
    ],
    "part9-open-problems": [
        (r"Kaplan.*2020|Scaling", "kaplan2020"),
        (r"Schaeffer.*2023", "schaeffer2023"),
    ],
    "part9-prediction": [
        (r"Sutton.*Bitter", "sutton2019"),
        (r"Shannon.*1948", "shannon1948"),
    ],
}


def add_inline_citations(path: Path, cite_list: list) -> int:
    """在正文中搜索关键模式并添加 [@key]。"""
    text = path.read_text(encoding="utf-8")
    added = 0

    # 参考文献区起点：其后的任何插入都是破坏（v1.2 事故教训）
    ref_start = text.find("## 参考文献")

    for pattern, key in cite_list:
        # 检查是否已有这个 key 的引用
        if f"[@{key}]" in text:
            continue

        # 搜索模式
        matches = list(re.finditer(pattern, text))
        if not matches:
            continue

        # 在第一个匹配后面插入 [@key]（如果它不在代码块或 callout 标题中）
        for m in matches:
            insert_pos = m.end()
            # 检查插入点是否安全（不在代码块、标题、callout 标记中）
            context_before = text[max(0, insert_pos - 200):insert_pos]
            context_after = text[insert_pos:insert_pos + 50]

            # 跳过参考文献区与其后的一切
            if ref_start != -1 and insert_pos > ref_start:
                continue
            # 跳过不安全位置
            if any(x in context_before[-100:] for x in ['```', ':::', 'title=', '####']):
                continue
            if context_after.startswith(']') or context_after.startswith(')'):
                continue
            # 跳过 URL 行（DOI / scholar 链接内插引用会弄坏链接）
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            line = text[line_start:line_end if line_end != -1 else len(text)]
            if "http" in line or "doi.org" in line:
                continue
            # 跳过英文词中间：插入点后紧跟 ASCII 字母/连字符（如 induction|heads、GPT|-3）
            # 或匹配起点本身就在一个 ASCII 词的中间
            if insert_pos < len(text) and (text[insert_pos].isascii() and (text[insert_pos].isalnum() or text[insert_pos] in "-_")):
                continue
            if m.start() > 0 and text[m.start() - 1].isascii() and text[m.start() - 1].isalnum():
                continue

            # 确保不在已有的引用标记内
            if f"[@{key}]" in text[max(0, insert_pos - 10):insert_pos + 10]:
                continue

            # 插入
            citation = f" [@{key}]"
            # 如果后面紧跟标点，在标点前插入
            if insert_pos < len(text) and text[insert_pos] in '，。；）':
                text = text[:insert_pos] + citation.rstrip() + text[insert_pos:]
            else:
                text = text[:insert_pos] + citation + text[insert_pos:]
            added += 1
            break  # 每个模式只加一次

    if added > 0:
        path.write_text(text, encoding="utf-8")
    return added


def main():
    total_added = 0
    chapters_done = 0

    for chapter, cite_list in CITE_MAPS.items():
        # 确定文件路径
        found = False
        for vol in ["vol1", "vol2", "vol3"]:
            path = ROOT / "chapters" / vol / f"{chapter}.qmd"
            if path.exists():
                added = add_inline_citations(path, cite_list)
                total_added += added
                if added > 0:
                    chapters_done += 1
                    print(f"  ✓ {chapter}: +{added}")
                found = True
                break
        if not found:
            print(f"  ✗ {chapter}: 文件未找到")

    print(f"\n完成: {chapters_done} 章新增 {total_added} 处内联引用")


if __name__ == "__main__":
    main()
