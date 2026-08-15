"""章节 .qmd 文件的代码围栏语法体检。

曾两次因围栏写成 ```{python （缺右括号）导致该 cell 未被识别为
jupyter 单元、静默退化为行内文本——渲染"成功"但实验消失。
此测试让该类错误在 CI 阶段就爆炸。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _qmd_files():
    return list((ROOT / "chapters").rglob("*.qmd")) + [
        ROOT / "templates" / "chapter-template.qmd",
    ]


def test_no_malformed_fence_openers():
    bad = []
    for f in _qmd_files():
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("```{") and not s.endswith("}"):
                bad.append(f"{f.relative_to(ROOT)}:{lineno}: {s}")
    assert not bad, "发现未闭合的围栏开头：\n" + "\n".join(bad)


def test_code_fences_are_balanced():
    """状态机式检查：每打开一个围栏（``` 任意语言）必须被闭合。"""
    for f in _qmd_files():
        in_fence = False
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not in_fence and s.startswith("```"):
                in_fence = True
            elif in_fence and s == "```":
                in_fence = False
        assert not in_fence, f"{f.relative_to(ROOT)}: 有未闭合的代码围栏"
