# 降 Loss 之路（Loss Descent）

一本用实验重走深度学习研究史的交互书：**从感知机到智能体，以历史为骨架、以降 loss 为解剖刀**。历史主线负责「为什么会这样」，loss 主线负责「这到底买到了什么」，两者拧成一股贯穿 38 章。

进度与计划见 [ROADMAP.md](ROADMAP.md)，全书章节清单见 [outline.yml](outline.yml)，部署与 CI 方案（零服务器）见 [DEPLOY.md](DEPLOY.md)。

## 目录结构

```
├── _quarto.yml          # 书的目录与主题配置
├── index.qmd            # 封面与阅读指南
├── outline.yml          # 全书 38 章清单（唯一事实来源）
├── chapters/            # 正文：vol1 根基与黎明 / vol2 序列与规模 / vol3 LLM 与智能体
├── templates/           # 章节骨架（生成用）与排版样板（参照用）
├── scripts/
│   └── gen_chapters.py  # 按 outline.yml 生成章节骨架，只建不覆盖
├── src/dlbook/          # 随书生长的代码库：每章新模型合入，后续章节 import 复用
├── tests/               # dlbook 的 pytest
└── theme/book.scss      # 自定义栏目配色（时代快照/停一停/回声/论文时光机）
```

## 本地开发

前置条件：Python ≥ 3.10；渲染需要安装 [Quarto](https://quarto.org)（Windows：`winget install --id Quarto.Quarto`）。

```bash
pip install -e ".[dev]"        # 安装 dlbook（可编辑）+ 测试工具
python scripts/gen_chapters.py # 按 outline.yml 补齐缺失的章节骨架
pytest -q                      # 跑 dlbook 测试
quarto preview                 # 本地预览整本书
quarto render                  # 渲染到 _book/
```

## 写作一章的流程

1. `python scripts/gen_chapters.py` 生成骨架（已有文件不会被覆盖）；
2. 对照 `templates/chapter-template.qmd` 的排版约定填充内容；
3. 章内代码实现进 `src/dlbook/` 对应模块，notebook 只负责调用与展示；
4. 实验跑通后在 `tests/` 补对应测试，`pytest` 与 CI 全绿；
5. `outline.yml` 里把该章 `status` 从 `stub` 改为 `drafting` → `done`。

## 章节固定栏目

每章按固定顺序包含：算力需求（A/B/C 级）→ 时代快照 → 本章问题 → 历史中的尝试（包括弯路）→ **停一停，你来当研究者**（在揭示解决方案之前，先让读者自己解上一章的遗留问题）→ 核心思想 → 从零实现 → 消融实验 → **本章 Loss 账本**（四问表格）→ 回声（→ 现代 LLM 后代）→ 论文时光机（原始文献导读）→ 遗留问题 → 练习与折叠答案。

其中「Loss 账本」「停一停」「回声」是本书区别于其他教材的核心栏目，详见排版样板。
