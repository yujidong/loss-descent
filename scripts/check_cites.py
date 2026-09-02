"""检查每章的内联引用数——删除手动参考文献节后，零引用章节将没有 bibliography。"""
import glob
import re

for f in sorted(glob.glob("chapters/vol*/*.qmd")):
    s = open(f, encoding="utf-8").read()
    # 只统计正文区的引用（参考文献节之前）
    body = s.split("## 参考文献")[0]
    n = len(re.findall(r"\[@[a-zA-Z0-9]+\]", body))
    flag = "  ← 无引用!" if n == 0 else ""
    print(f"{n:3d}  {f}{flag}")
