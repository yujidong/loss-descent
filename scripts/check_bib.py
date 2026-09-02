"""检查 references.bib 中被截断的作者名/机构名/标题(v1.2 审稿 P2-5)。"""
import re
import sys

s = open("references.bib", encoding="utf-8").read()

# LaTeX 缩音截断特征: 姓氏重音元音后缺剩余字母,或单词在引号内戛然而止
PATTERNS = [
    (r'J\\"u(?!r)', 'Schmidhuber, J\\"u → 应为 J\\"urgen'),
    (r'L\\"e(?!o)', 'Bottou, L\\"e → 应为 L\\"eon'),
    (r"R\\'e(?!j)", "Réjean 截断"),
    (r'A\\"a(?!r)', "Aäron 截断"),
    (r'Universit\\"a(?!t)', 'Technische Universität 截断'),
    (r'Merri\\"e(?!n)', 'van Merriënboer 截断'),
    (r'Dess\\`i(?! )', 'Dessì 后缺名'),
    (r'\bJ\\"u\b', '孤立 J\\"u'),
]

found = 0
for pat, desc in PATTERNS:
    for m in re.finditer(pat, s):
        ctx = s[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")
        print(f"[{desc}] ...{ctx}...")
        found += 1

# 标题在 bib 条目内异常短(如只剩 "*CNN*")
for m in re.finditer(r'title\s*=\s*[{"]([^}"]{1,12})[}"]', s):
    print(f"[超短标题] {m.group(1)!r}")

print(f"\n共 {found} 处截断可疑;bib 条目总数: {s.count('@')}")
sys.exit(0)
