# 部署与 CI：为什么不需要自己的服务器

**结论先行**：本书的全部 CI 与托管都可以零服务器、零成本运行——GitHub Actions 的构建跑在 GitHub 免费提供的虚拟机上（每次渲染约 2–3 分钟），产物是纯静态 HTML，由静态托管免费分发。你的笔记本只用来写作。

## 方案 A（推荐）：GitHub 一条龙

仓库里现成的 `.github/workflows/render-site.yml` 已经写好，推上去即可：

```bash
gh repo create loss-descent --public --source=. --push   # 或先建私有仓再公开
```

1. push 到 `main` 后，Actions 自动渲染整本书并推到 `gh-pages` 分支；
2. 仓库 Settings → Pages → Source 选 `gh-pages` 分支，网站即上线 `https://yujidong.github.io/loss-descent/`；
3. 每周二 UTC 20:00 自动重渲染（防止依赖升级导致 notebook 腐化）。

**费用**：public 仓库 Actions 完全免费且不限时长；GitHub Pages 免费托管。若想先用 private 仓打磨，每月有 2000 分钟免费额度，本书一次全量渲染约 3 分钟，绰绰有余。

**注意**：CI 里跑的只是"渲染 + 跑测试"，不涉及任何私密信息；`_freeze/` 已入库，渲染结果可复现。

## 方案 B：Cloudflare Pages（国内读者访问更稳）

GitHub Pages 在国内访问时快时慢。若主要读者在国内，Cloudflare Pages 免费额度（不限带宽）通常更稳：

1. 把仓库接入 Cloudflare Pages（连 GitHub 一键授权）；
2. 构建命令：安装 Quarto 后 `quarto render`，输出目录 `_book`：

```bash
curl -L -o quarto.tar.gz https://github.com/quarto-dev/quarto-cli/releases/download/v1.10.18/quarto-1.10.18-linux-amd64.tar.gz
tar -xzf quarto.tar.gz && export PATH="$PWD/quarto-1.10.18/bin:$PATH"
quarto render
```

（Netlify 同理，也有社区维护的 Quarto 构建插件。）

## 方案 C：零部署

`_freeze/` 已提交，`quarto render` 在任何装了 Quarto 的机器上都能**不重新执行代码**直接复现出 `_book/` 静态站——把 `_book` 目录拖到任意静态托管、对象存储甚至局域网共享即可阅读。本地阅读就是直接打开 `_book/index.html`。

## 与写作的关系

三条路线互不干扰写作流：你永远只在本地 `quarto preview` 写作，部署是 CI 的事。若暂时不想上云，方案 C 说明本书今天就是可读、可分发、可复现的。
