#!/usr/bin/env python3
"""合并 reviews/ 目录下的全部批注导出文件，生成 REVIEW-COMMENTS.md。

配合书内「批注模式」使用：在渲染出的网页上划选文字添加批注，
点「导出 JSON」，把下载的文件放进 reviews/ 目录，然后运行：
    python scripts/collect_review.py [目录，默认 reviews/]
输出按章节分组、带小节与原文引文，供针对性修订直接使用。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPE_ORDER = {"问题": 0, "事实核对": 1, "建议": 2, "文风": 3, "其他": 4}


def load_all(review_dir: Path) -> dict[str, list[dict]]:
    """读取目录下全部 *.json，按章节合并（后导出的覆盖同 id 批注）。"""
    chapters: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(review_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"跳过无法解析的文件 {f.name}: {e}")
            continue
        comments = data.get("comments", data if isinstance(data, list) else [])
        chapter = data.get("chapter") or f.stem
        for c in comments:
            if not isinstance(c, dict) or not c.get("note"):
                continue
            c.setdefault("id", f"{f.stem}:{len(chapters[chapter])}")
            c["_src"] = f.name
            chapters[chapter][c["id"]] = c  # 同 id 后写覆盖（重复导出时以最新为准）
    return {ch: sorted(items.values(), key=lambda c: c.get("createdAt", "")) for ch, items in chapters.items()}


def find_qmd(chapter: str) -> str | None:
    hits = list((ROOT / "chapters").rglob(f"{chapter}.qmd")) or list((ROOT / "chapters").rglob(f"{chapter}*.qmd"))
    if hits:
        try:
            return hits[0].relative_to(ROOT).as_posix()
        except ValueError:
            return None
    return None


def main() -> int:
    review_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reviews"
    if not review_dir.exists():
        print(f"目录不存在：{review_dir}")
        return 1

    chapters = load_all(review_dir)
    if not chapters:
        print("没有找到任何批注。")
        return 0

    lines = [
        "# 审阅批注汇总（生成于 " + datetime.now().strftime("%Y-%m-%d %H:%M") + "）",
        "",
        "> 由 scripts/collect_review.py 从 reviews/ 自动合并。每条含类型、原文引文（定位用）、批注、所在小节。",
        "> 修订时按「章节 → 引文」在 .qmd 里检索定位；引文取自渲染文本，与源文件的 markdown 记号可能有细微差异。",
        "",
    ]
    total = 0
    for ch in sorted(chapters):
        items = chapters[ch]
        qmd = find_qmd(ch)
        lines.append(f"## {ch}" + (f"（{qmd}）" if qmd else "（未找到对应 .qmd）") + f" — {len(items)} 条")
        lines.append("")
        for c in items:
            total += 1
            lines.append(f"- **[{c.get('type', '其他')}]** 「{c.get('quote', '（整段）')}」")
            lines.append(f"  - 批注：{c['note'].strip()}")
            if c.get("section"):
                lines.append(f"  - 小节：{c['section']}")
            lines.append(f"  - 来源：{c.get('_src', '')} · {c.get('createdAt', '')[:16]}")
        lines.append("")

    by_type: dict[str, int] = defaultdict(int)
    for items in chapters.values():
        for c in items:
            by_type[c.get("type", "其他")] += 1
    summary = "、".join(f"{t} {by_type[t]}" for t in sorted(by_type, key=TYPE_ORDER.get) if t in by_type)
    lines.insert(2, f"共 {total} 条（{summary}），涉及 {len(chapters)} 章。")

    out = ROOT / "REVIEW-COMMENTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {out.name}：共 {total} 条批注，涉及 {len(chapters)} 章。")
    for ch in sorted(chapters):
        print(f"  {ch}: {len(chapters[ch])} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
