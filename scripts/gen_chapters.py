#!/usr/bin/env python3
"""根据 outline.yml 生成章节骨架。

只创建缺失的 .qmd 文件，绝不覆盖已有内容——已存在的章节说明正在写作或已完成。
用法：python scripts/gen_chapters.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from string import Template

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "templates" / "chapter-skeleton.md"


def main() -> int:
    outline = yaml.safe_load((ROOT / "outline.yml").read_text(encoding="utf-8"))
    tpl = Template(SKELETON.read_text(encoding="utf-8"))

    created: list[str] = []
    skipped: list[str] = []
    for volume in outline["volumes"]:
        for part in volume["parts"]:
            for ch in part["chapters"]:
                path = ROOT / ch["file"]
                if path.exists():
                    skipped.append(ch["file"])
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    tpl.substitute(
                        title=ch["title"],
                        era=ch["era"],
                        summary=ch["summary"],
                        milestone=ch["milestone"],
                        status=ch.get("status", "stub"),
                    ),
                    encoding="utf-8",
                )
                created.append(ch["file"])

    print(f"created {len(created)} chapter stubs:")
    for f in created:
        print(f"  + {f}")
    if skipped:
        print(f"skipped {len(skipped)} existing files (never overwritten):")
        for f in skipped:
            print(f"  = {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
