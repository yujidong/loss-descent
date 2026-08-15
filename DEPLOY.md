# 部署与 CI：通道、现实与方案

**结论先行**：不需要自己的服务器；CI 永远跑在托管方的免费机器上。但在大陆环境下，必须把**两条通道**分开讨论，因为它们面对的墙不一样：

1. **写作通道**（你 → 代码仓库）：只有你 push 新内容时才用到。github.com 在大陆不翻墙时通断不定，但这是低频操作——有代理时推一下即可，还可以给 git 单独配代理（见下）。**每周的 CI 自动重渲染跑在 GitHub 的服务器上，不需要你翻墙。**
2. **阅读通道**（读者 → 托管站点）：这才是选托管方的决定因素，与你的推送方式无关。

## 大陆阅读通道的现实（2025–2026 实测社区共识）

| 托管 | 大陆直连情况 |
|---|---|
| GitHub Pages（`*.github.io`） | 时好时坏，不稳定；自定义域名也不保证 |
| Cloudflare Pages 默认域名（`*.pages.dev`） | **基本不可用**（DNS 污染/屏蔽）——常见误区："换 Cloudflare 就稳了"只对了一半 |
| Cloudflare Pages + **自定义域名** | 通常可直连，速度中等；配合优选 IP（CloudflareSpeedTest）可明显加速 |
| 国内云对象存储/CDN（腾讯 COS、阿里 OSS） | 最快，但自定义域名需 ICP 备案；默认域名外链可直接用，无需备案 |

## 推荐路线（按书的成长阶段）

**现在（试读期）**：方案 A（GitHub 一条龙）。读者主要是你和试读的朋友，可访问性问题影响面小；CI 与防腐蚀机制先跑起来，发布管线尽早经过实战。

**卷一发布（面向公众）**：升级为方案 B——Cloudflare Pages **绑定自定义域名**（域名几十元/年）。构建仍由 Cloudflare 从 GitHub 拉取（它的构建机在墙外，不受影响；你只需偶尔用代理 push 源码）。

## 方案 A：GitHub 一条龙

```bash
gh repo create loss-descent --public --source=. --push
```

push 到 `main` 后 Actions 自动渲染并推到 `gh-pages` 分支；仓库 Settings → Pages → Source 选 `gh-pages` 即上线。public 仓库免费不限时长。给 git 单独配代理（不影响其他流量）：

```bash
git config --global http.https://github.com.proxy http://127.0.0.1:7890   # 端口换成你代理的
```

## 方案 B：Cloudflare Pages + 自定义域名

1. 买一个域名，接入 Cloudflare（免费套餐即可）；
2. Cloudflare Dashboard → Workers & Pages → 创建 Pages 项目 → 连接 GitHub 仓库；
3. 构建命令安装 Quarto 后渲染（构建机在墙外，直连 GitHub 无碍）：

```bash
curl -sL -o quarto.tar.gz https://github.com/quarto-dev/quarto-cli/releases/download/v1.10.18/quarto-1.10.18-linux-amd64.tar.gz
tar -xzf quarto.tar.gz && export PATH="$PWD/quarto-1.10.18/bin:$PATH"
quarto render
```

4. 输出目录 `_book`；在 Custom domains 绑定你的域名（**关键一步，否则大陆读者打不开**）；
5. 可选：用 CloudflareSpeedTest 优选 IP 提速。

**变体 B'（完全不碰 GitHub）**：本地渲染 + 直传，适合暂时不想开仓的情况：

```bash
npm install -g wrangler
wrangler pages deploy _book --project-name=loss-descent
```

代价：失去"push 即部署"与基于 Actions 的每周自动重渲染（防腐化改由本地/CI 脚本承担），自定义域名同样必须绑。

## 方案 C：零部署

`_freeze/` 已入库，任何装了 Quarto 的机器都能不重新执行代码、直接复现出 `_book/` 静态站——拖到任意静态托管或对象存储默认域名外链即可阅读。本地阅读就是打开 `_book/index.html`。

## 与写作的关系

三条路线互不干扰写作流：你永远只在本地 `quarto preview` 写作。托管方随时可换、可叠加（A + B 可同时挂着），换的只是阅读通道的入口。
