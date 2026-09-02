"""删除各章手动的「## 参考文献」节——Quarto 会用 references.bib 自动渲染
每章的 bibliography（P1-2：此前两套列表重复且手动版有人名截断）。

用法: python scripts/strip_manual_refs.py [--dry-run]
"""
import glob
import re
import sys

dry = "--dry-run" in sys.argv
changed = 0
for f in sorted(glob.glob("chapters/vol*/*.qmd")):
    s = open(f, encoding="utf-8").read()
    # 手动参考文献节：从 "## 参考文献" 到下一个 "## " 或文件尾
    m = re.search(r"\n## 参考文献\n.*?(?=\n## |\Z)", s, flags=re.S)
    if not m:
        continue
    new = s[: m.start()] + s[m.end():]
    # 清理多余空行
    new = re.sub(r"\n{4,}", "\n\n\n", new)
    if new != s:
        changed += 1
        print(f"strip: {f}")
        if not dry:
            open(f, "w", encoding="utf-8").write(new)
print(f"\n{changed} 个文件处理{'（dry-run 未写入）' if dry else ''}")
