# 《降 Loss 之路》学术审稿报告

> 审稿人视角:深度学习学者 + 教育者双重视角。
> 范围:全书 38 章正文(三卷)、`dlbook` 代码包、首页承诺与全书框架一致性。
> 方法:逐章精读 + 跨章数值核对 + 外部文献核查(关键事实均已用原始论文/一手来源验证)。
> 本文档 = 下一阶段的完整工作目标清单。所有条目可执行:均含文件定位、问题说明、修改建议、优先级。

---

## 一、总体评价

**骨架一流,血肉未齐。**

强项(修订时不要破坏的东西):
- 「上一章遗留问题 = 下一章起点」的问题链兑现度高,卷二尤佳(3.30 → 1.95 → 2.23 → 1.79 的 loss 数字全卷对得拢);
- Loss 账本框架贯穿 37/38 章,四问格式稳定;
- 实验纪律出色:多处「先估算再运行」「预测再验证」设计,且有敢记零结果和反直觉读数的诚实(LSTM 输给 RNN、MoE 打平、外推失败 15%、诱导头复现失败);
- 练习体系完整:35+ 章有「停一停,你来当研究者」,附可折叠自检锚点;
- Part 9 的可证伪预测框架(每条预测带「会证伪我的观察」)是同类书少见的认识论自觉。

弱项(下一阶段的全部工作):
1. **约 161 处机械复述**污染全书(详见全局问题 G1)——这是当前最伤读者信任的一项;
2. **出版级 P0 硬伤 15+ 处**(年份、归属、算术、实验有效性),集中在历史事实与实验声明;
3. **文风 v1.1 修订未落地**:范改例所在的 part5-transformer 本身仍是「改前」版本;
4. 批量脚本(inline_cite.py / add_recaps.py)留下的系统性残留(误插引用、空壳复述、标点事故);
5. 首页承诺未兑现(C 级实验、linear probe)。

---

## 二、下一阶段路线图(按此顺序执行)

| 阶段 | 内容 | 规模 |
|---|---|---|
| **R1. P0 硬伤清零** | §三 全部 27 组(正文 22 + 代码 5,均含验证),逐处修改并重渲染 | 2-3 天 |
| **R2. 实验重跑校准** | 修完代码 P0 后**先重跑受影响实验再改正文读数**:SimpleRNN/LSTM/AttentionSeq2Seq 梯度相关(part4-rnn 剖面、part4-lstm、part4-seq2seq)、MiniGPT MLM(part5-gpt-vs-bert 全部)、卷三 G4 跨章读数、8.1 计算器实验、9.2 数据墙实验 | 3-4 天 |
| **R3. 缓一缓全量重写** | 161 处空壳 → 每处 2-3 句真实复述(可半自动:LLM 初稿 + 人工校订) | 3-5 天 |
| **R4. 文风 v1.1 落地** | WRITING-STYLE.md 改例逐条落回正文;清 15 处「?。」与断片句 | 2-3 天 |
| **R5. 教学补强** | RLHF 目标函数小节、tokenizer/BPE callout、ReAct 补入、练习去剧透、PPO 铺垫 | 3-5 天 |
| **R6. 工程收尾** | 参考文献去重、C 级承诺兑现或删除、仓库链接统一、图片 alt、引号 bug、代码卫生项(C-3~C-7) | 2 天 |
| **R7. 测试补盲** | 现有 pytest 36 项全过却漏了 5 个数学 bug:补 FD 测试覆盖 RNN 的 dWxh/dWhh/dh0、LSTM 非零初态分支、AttentionSeq2Seq 的 dhs_d/dhs_e、MiniGPT 的 MLM grad_tok、MLPNumpy d_out>1;并把本次审稿的冒烟测试固化为 tests/ | 1 天 |

---

## 三、P0:必须修复的事实/技术/渲染错误

> 以下每条均已对照原文或原始论文核实。修订后逐条打勾。

### 全局性(多章)

- [ ] **P0-1 `scripts/inline_cite.py` 误插引用(卷三 6 处,渲染必坏)**:引用键被插进英文词中间、论文标题中间、URL/DOI 中间。逐处:
  - part7-base-model:`induction [@olsson2022] heads` → 移句末;
  - part8-model-vs-system:`GPT [@radford2018]-3` → `GPT-3 [@radford2018]`;
  - part9-open-problems:Hestness 论文标题内嵌 `[@kaplan2020]`,且 Kaplan 在该章正文零引用 → 拆开;
  - part9-prediction:Sutton 条目 scholar URL 内嵌 `[@sutton2019]`;Shannon 条目 DOI 内嵌 `[@shannon1948]`(链接必坏);
  - part8-agent-training:scholar URL 内嵌 `[@yao2023]`,且 `[@yao2023]` 正文零引用。
  - **防再犯**:给 inline_cite.py 加排除规则(参考文献区、URL、英文词内部)。
- [ ] **P0-2 part9-prediction.qmd:98 引号 bug**:`collapse="true'`(双开单闭),Quarto 解析失败风险 → 改 `collapse="true"`。
- [ ] **P0-3 标点事故「?。」全书 15 处** + 句中「。」替代逗号造成的断片句(如「它在训练中期突然形成(相变。)」「然后做一件本书一直在铺垫的事。亲眼看着模型被…骗走」)。批量修 + 复查。

### 卷一

- [ ] **P0-14 part1-backprop.qmd:15 Ronald Williams 身份错误**:「一位研究生 Ronald Williams」——他 1979 年已获博士,1986 年是东北大学(Northeastern University)教授 → 改「和东北大学的 Ronald Williams」。这是全书旗舰章的人物硬伤。
- [ ] **P0-15 part1-mlp-winter.qmd 「四个数量级」梯度论断与自家实验矛盾**:「16 层网络的第一层收到的梯度天然比第二层小约四个数量级」——相邻两层只差一次连乘,比值 ~0.25,谈不上数量级;本章消融实际测到的是 16 层 vs 2 层网络首层差约 **7 个数量级**。且 2.2 章开头转述成「比末层小七个数量级」,与实验测的量(跨深度比较首层)也不符 → 三处统一为同一个真命题:「16 层网络首层梯度比浅网络小约七个数量级(见 1.4 消融二)」。
- [ ] **P0-16 part2-vanishing-gradient.qmd:5 LSTM 时间线错误**:「被冷落了大约十年……直到 2001 年前后才被重新发现,并直接催生了 LSTM」——LSTM 是 **1997** 年,正是 1991 论文的直接后续而非「重新发现」产物;Bengio 1994 也已引用它 → 改「它的直接后果是 1997 年的 LSTM;但作为对『深』的通用诊断,它要到 2010 年代深网复兴才被广泛听见」。
- [ ] **P0-17 part2-vanishing-gradient.qmd 反传递推公式维度错误**:$\delta_1 = \prod_l (W_l^\top \odot f'(z_l))\,\delta_L$ 中矩阵与向量的逐元素乘在连乘号内不成立(缺 diag)→ 改 $\delta_1 = \big(\prod_{l}\mathrm{diag}(f'(z_l))\,W_{l+1}^\top\big)\delta_L$。以本书「逐字可验证」标准必须修。
- [ ] **P0-18 part2-toolbox.qmd 「2015 年前最深的实用网络约十层」与本书 3.2 章自相矛盾**:2014 年 VGG 已 19 层、GoogLeNet 22 层(3.2 章自己写了)→ 改「2012 年前约十层;2014 年 VGG/GoogLeNet 推到 19-22 层,但再加即退化」。
- [ ] **P0-19 part3-resnet.qmd 「退化病」张冠李戴**:plain 深度 10 崩到 0.50 被解读为「退化病复刻」,但 2.3 章已确证这是无 BN relu 深网的「训练动态漂移」旧病;退化(加 BN 后更深者训练误差更高)在本玩具上根本未测出 → 二选一:(a) 扫描加 BN 真测退化,测不出就诚实说明;(b) 改叙事为「残差恒等路把漂移病也治了——残差是比 BN 更结实的骨骼」。现状标签是全卷最需修的因果错误。
- [ ] **P0-20 part2-alexnet.qmd 练习 1 答案除法错误**:「$3\times10^{23} \div 10^{12} ≈ 10^{11}$ 秒 ≈ 三千年」——$3\times10^{23}/10^{12}=3\times10^{11}$ 秒 ≈ **9500 年**;乘 30% 利用率则约 3.2 万年 → 以本书「账本」标准,练习答案的除法必须过硬。
- [ ] **P0-21 part2-capstone-deep-mlp.qmd:85 参数量错误(已实测)**:「八个隐藏层、约两万六千个参数」——实际运行 `MLPNumpy(2, [32]*8+[1], batchnorm=True)` 为 **7521 个参数**,是旧配置残留 → 改「约七千五百个参数」。
- [ ] **P0-22 part3-cnn-legacy.qmd:25 OverFeat 引用键错配**:「OverFeat(2014 [@razavian2014])」——OverFeat 是 **Sermanet et al.**(ICLR 2014),Razavian 是另一篇 "CNN features off-the-shelf" → 新建 `[@sermanet2014]` 条目并改正;渲染后即成归属错误。

### 卷二

- [ ] **P0-4 part6-scaling-laws.qmd:5 GPT-3 年份错字**:「2010 年 1 月,OpenAI 发布 GPT-3」→ **2020 年**(5 月)。
- [ ] **P0-5 part4-lstm.qmd:122 遗忘门数字与自家代码表矛盾**:「$b_f=1$ 在 60 步后只剩 0.3%」。σ(1)=0.731,0.731^60 ≈ 6.8×10⁻⁹,与紧邻代码将打出的 `6.8e-09` 直接冲突(0.3% 约对应 18 步)→ 改「只剩 10⁻⁸ 量级」。
- [ ] **P0-6 part5-transformer.qmd:26 归属错误**:「Long et al. 2016 的结构化自注意力」——"Long" 是被误读的论文标题词(Long Short-Term Memory-Networks);作者是 **Cheng, Dong & Lapata (2016)**。
- [ ] **P0-7 part6-convergence.qmd:28 「MoE 与 LSTM 同年出生」自相矛盾**:Jacobs et al. 1991,但 LSTM 是 1997(本书 part4-lstm 自己写明)→ 改「与 Hochreiter 1991 年诊断梯度消失的论文同年」。顺修同句「Jacobs et al。)」标点。
- [ ] **P0-8 part5-gpt-vs-bert.qmd:14 「33 亿词是两家的共同起点」不成立**:GPT-1 **只用 BooksCorpus**(约 7000 本书);BooksCorpus+Wikipedia 是 BERT 的配方 → 改写。
- [ ] **P0-9 part4-lstm.qmd:5 谷歌翻译年份错误且与另一章矛盾**:「2015 年谷歌翻译上线」——GNMT 是 2016 年(9 月首个语言对,11 月全面推出;part4-seq2seq 自己写的是 2016)→ 统一改「2016 年秋切换 GNMT」。

### 卷三

- [ ] **P0-10 part7-rlhf.qmd:5 DPO 年份错误**:「2022 年的 RLHF……同年稍晚的 DPO」——DPO 是 **2023 年 5 月**(arXiv 2305.18290)→ 改「次年(2023)的 DPO」;本章时代快照「2022」期改「2022–2023」。
- [ ] **P0-11 part7-sft.qmd:15/53/227 「175 倍」算术错误(3 处)**:175B/1.3B ≈ **135 倍**;论文原话是 "100× fewer parameters" → 统一改「约 135 倍」或引原文「百倍」。
- [ ] **P0-12 part8-agent-origins.qmd 计算器实验拦截时机 bug,典型读数不可信**:工具调用检查发生在「生成一个字符之后」,而 prompt 以 `=` 结尾,该 `=` 在首次检查前就被顶掉——工具永远无法对被测算式注入答案;评测取最左匹配,量到的或是模型裸猜、或是平凡保证。**修法**:把 `a+b=` 检查移到每步生成之前,且只对被测算式判分;修后重跑校准读数。
- [ ] **P0-13 卷三 G4「典型读数」跨章不一致(同管线同种子应同数)**:SFT 贪心命中率 7.2 表 `0.17` vs 8.1 `0.33→0.42` vs 8.3 `0.00/0.00/0.33`;seq2seq 5-epoch 贪心 7.4 `0.88` vs 8.2 `0.80`。重跑统一,或在后文注明差异来源。

### 代码库(影响章节实验读数,与 R2 联动)

- [ ] **P0-23 `src/dlbook/rnn/core.py:55,147` SimpleRNN BPTT 缺 Whh 转置**:回传写成了 `dh_next = self.Whh @ dz`,链式法则要求 `Whh.T @ dz`(FD + torch 双重验证;修后 FD 误差 4.93→1.3e-11)。波及 `rnn_grad_flow_norms`(part4-rnn/part4-lstm 的梯度消失剖面实验)——**修后必须重跑,章内「56 个数量级」等读数可能变化**。
- [ ] **P0-24 `core.py:111,121` LSTM backward 在 t=0 把初始状态当零**:非零初态(Seq2Seq/Attention 解码器、RNNLM.sample)时丢两条梯度路径(gWx vs FD 误差 0.18)→ forward 缓存初始 (h,c) 并在 t=0 使用;零初态用法不受影响。
- [ ] **P0-25 `rnn/seq2seq.py:176-177` 注意力反向遗漏 1/√H**:前向 scores 乘了 1/√hidden,反向没乘回(方向导数甚至符号翻转)→ 两处 `@` 各乘 `1/np.sqrt(self.hidden)`。顺带:该类实现的是缩放点积注意力而非 Bahdanau 加性注意力,5.1 章标题是 Bahdanau——在 docstring/正文明确「简化版」或补加性版本(与 P1-5 呼应)。
- [ ] **P0-26 `transformer/model.py:99-101` MLM 模式 token 嵌入从不训练**:`grad_tok` 只在 causal 分支散射累加——**5.4 章 GPT vs BERT 的全部 MLM 实验是在随机冻结嵌入上跑的**。修复:mlm 分支补 `np.add.at(self.grad_tok, ...)`;**修后重跑 5.4 全部实验并校准读数**(与 P0-13 联动)。
- [ ] **P0-27 `nn/mlp_numpy.py:126-127` 多输出时 loss 与梯度口径不一致**:loss 按 n·d_out 平均、梯度只按 n(d_out=2 时 FD 比值恰为 2.000)。书中全部 d_out=1 自洽,但需修:梯度除以 `len(X)*pred.shape[1]`。

---

## 四、P1:教学与结构问题

### 全局

- [ ] **P1-1 约 161 处「缓一缓」是零信息空壳**(184 处中仅 23 处有真实内容):「这一节的核心是:〈节标题〉」。WRITING-STYLE 第五条的本意是内容性复述。重写为每处 2-3 句真实复述;做不到就全删——现状比没有更糟。
- [ ] **P1-2 参考文献双重显示(全书)**:每个页面同时渲染手动「参考文献」列表和 Quarto 自动 bibliography(`id="ref-"`,且自动版条目更全)。删除手动列表,保留 Quarto 原生(顺带解决 bib 作者名截断,见 P2-8)。
- [ ] **P1-3 练习把答案写进题干(剧透)**:如 part4-seq2seq 练习 3 直接给出答案句;part4-rnn 练习 5 剧透 IRNN。「预测再运行」类给锚点是体例,「造轮子/消融」类不应剧透 → 全书排查。
- [ ] **P1-3b 悬空交叉引用「7.x / 8.x」约 9 处**:指向不存在的章节编号,且「8.x」被混用于三个不同主题:扩散模型(part1-symbolic-vs-connectionist:79,全书无扩散章)、长上下文(part3-cnn-limits:167、part2-vanishing-gradient:149、part4-rnn:151、part5-bahdanau:55)、上下文工程(part2-capstone:149)。逐处改指真实章节(长上下文主要落点应是 part9-open-problems)或删引用;「扩散模型」要么在 part9 补一段,要么删掉指向。

### 卷一

- [ ] **P1-24 死链「1.6」共 6 处**:part2-mlp-difficulty(2 处)、part2-alexnet(3 处)、part3-resnet(1 处)指向不存在的 1.6 章;向量引擎/过拟合实验/「存在 ≠ 可寻」实际都在 **1.4**(mlp-winter)→ 全局替换「1.6」→「1.4」,并建立交叉引用清单在 render 时校验(与 P1-3b 的「7.x/8.x」一起做)。
- [ ] **P1-25 「第 1 章」指代不明 4 处**(part1-backprop ×2、part3-cnn-legacy ×2):会被读成整个 Part 1 → 统一改「1.1 章」或「感知机一章」。
- [ ] **P1-26 「十年/十五年」三种说法打架**:mlp-winter「等十年才被听见」vs 同章「十五年的时差」;vanishing-gradient 章名「被忽视十五年」vs 正文「大约十年」→ 统一口径(建议按「1991→Bengio 1994 三年;1991→2010 年代深网复兴约二十年」写清楚「被谁在何时听见」)。
- [ ] **P1-27 part1-backprop.qmd 感知机规则与 ADALINE delta 规则混同**:「本章问题」节把 $\Delta w = \eta(y-\hat y)x$ 归给感知机,而下文历史节正确归给 ADALINE → 改为「感知机的误差只能取 ±2;真正的连续误差信号要等 ADALINE」。
- [ ] **P1-28 part1-backprop.qmd 记号冲突**:L 同时作层数与损失名($L = \lVert a_L - y\rVert^2$)→ 损失改 $\mathcal{L}$ 或 $J$。
- [ ] **P1-29 part2-alexnet.qmd:86 LaTeX 渲染破损**:`$10^6}$` 多半个花括号 → `$10^{6}$`。
- [ ] **P1-30 part2-alexnet.qmd GTX 580 数字自相矛盾**:快照「合计约 1.5 TFLOP/s」vs 正文「合计约 3 TFLOP/s」(单卡约 1.58,正文对)→ 快照改「单卡约 1.5 / 合计约 3」。
- [ ] **P1-31 part2-alexnet.qmd 消融节结构错误**:「缓一缓」在第 121 行收尾后第 124-135 行才出现代码块(孤儿块)→ 代码移回「缓一缓」之前。
- [ ] **P1-32 part2-toolbox.qmd Dropout 实验演示名不副实**:两次前向都在评估态,必然相同,标签却写「训练态两次前向相同吗」→ 给 `forward` 加 `training=True` 再对比,或改写正文承认只验证了评估态。
- [ ] **P1-33 part1-backprop.qmd 练习 1「裁判」没上场**:只定义 `numeric_grad`(eval: false)没有对拍调用 → 补两行示例调用。
- [ ] **P1-34 part1-perceptron.qmd 停一停缺 bias 更新**:给的规则只有 $w \leftarrow w + y\cdot x$,第 1 问却要求找 $(w_1,w_2,b)$ → 补 $b \leftarrow b + y$。
- [ ] **P1-35 part1-perceptron.qmd「噪声地板」存疑**:`make_slab` 若是无噪声平行板则无噪声地板可言 → 核实实现,没有就删那半句。
- [ ] **P1-36 账本「第一份/唯一」声明三处打架**(perceptron「唯一一份空白」vs symbolic 成对账本也空白 vs backprop「全书第一份正式账本」)→ 统一措辞。

### 卷二

- [ ] **P1-4 tokenizer/BPE 全书零覆盖**:卷二所有实验是字符级,但 part6 起满篇 "token"(含「20 token/参数」核心数字),读者无从知道 token 与字符之别、GPT-3 论文 loss 与本书 nat/字符不可比。修法:part6-scaling-laws 开头加一个 callout(BPE 一段 + nat/token 换算)。
- [ ] **P1-5 part5-bahdanau.qmd:32 把 2017 年的 √d 缩放安在 2014 年机制头上**:Bahdanau 原文是加性打分、无缩放(练习 2 自己写对了)→ 历史节改用原式,或注明「√d 是 2017 年补丁」。
- [ ] **P1-6 part5-bahdanau.qmd:14 WMT'14 语料量级错误**:「约 350 万句对」——Bahdanau 训练用全集约 **3.48 亿法语词**;「1200 万句对」是同组 Cho et al. 2014 的配套数字 → 改「约 3.5 亿法语词」。
- [ ] **P1-7 part5-why-transformer-won.qmd:49 「两者都在付 O(T) 的总计算」与技术事实冲突**:注意力是 O(T²),本章主题就是「平方税」→ 改「总计算都随 T 增长(RNN 线性、注意力平方),差别在支付方式:串行墙钟 vs 可并行批量」。
- [ ] **P1-8 part5-why-transformer-won.qmd:128-143 消融是空消融**:`orig_forward`/`banded_forward` 定义后从未调用(死代码),只对原模型计时,注释承诺的 4 倍降算没有兑现 → 真实现带状掩罩对照,或降级为讨论并指向练习 1。
- [ ] **P1-9 part6-scaling-laws 与 part6-chinchilla 对 Kaplan 协议描述互相矛盾**:6.1 说「训练到收敛的耦合」、6.2 说「大多没训到收敛」。事实:各点训到(近似)收敛,但固定学习率日程等绑定协议使等算力比较偏向大模型 → 6.1 改「三种资源在受控族上分别拟合」;6.2 精确为「固定学习率日程与批次策略使大模型系统性欠训练」;同时删掉 6.2 错列的「临界批次」(Kaplan 恰恰估计了它)。
- [ ] **P1-10 part7-rlhf.qmd RLHF 目标函数从未显式写出**:loss 为骨架的书,RLHF 章却直接从 KL 最优解的等价形式开讲。补一小节:先写 max E[r] − β·KL 目标 → 三行讲为什么需要策略梯度(期望在策略自己的采样上,不可直接求导)→ DPO 的消元才顺理成章。
- [ ] **P1-11 part7-rlhf.qmd 「同一类东西」论断不严谨**:SFT 期望在数据分布上,RL 期望在策略采样上,中间隔着不可微的采样 → 收紧为「DPO 的贡献是把 RL 目标拉回监督 loss 家族」(这反而让 DPO 更亮)。
- [ ] **P1-12 part7-sft.qmd 消融死代码**:`make_qa_corpus.__wrapped__(...)` 永远返回 (None,None) 且未用 → 删除。
- [ ] **P1-13 part8-agent-origins.qmd ReAct 完全缺席**:Agent 起源章没有 ReAct(Yao et al., ICLR 2023)→ 补入谱系:ReAct(行为语法)→ Toolformer(自标注)→ ToT(搜索化)。
- [ ] **P1-14 part8-agent-origins.qmd 「函数调用没有发布新模型」字面不实**:2023-06-13 发布了为函数调用微调的 0613 快照模型 → 改「没有新的能力代际,是同一代模型加格式对齐与解析层」(论点本身成立)。
- [ ] **P1-15 part8-model-vs-system.qmd few-shot 失效结论有混杂因素**:prompt 尾部截 95 字符可能把示例截掉,「few-shot 完全失效」的强结论必须先排除「示例不完整」→ 打印截断后 prompt 确认至少一个完整示例在窗内,或缩小示例规模;否则这个结论随时会被读者推翻。
- [ ] **P1-16 part9-open-problems.qmd 数据墙实验机制解释与代码对不上**:同 1200 步、窗口均匀采样下,复制两遍不增加唯一样本 exposure,「把重复文本背了下来」解释不通 → 要么复核读数,要么改实验设定(固定 epochs、步数×2 才是「背下来」的正确实验)。
- [ ] **P1-17 part9-prediction.qmd 预测时效**:示范预测「一年内推理预算成为 API 定价显式维度」在 2025-2026 已大半兑现(o1 计价、Gemini thinking_budget、Anthropic budget_tokens)→ 改写为「已兑现部分 + 下一个未决维度」,或加写作时点声明。预测三行加绝对日期锚(如「一年内(至 2027-09)」)以执行本章自定的「时间戳仪式」。
- [ ] **P1-18 part7-rlhf.qmd 练习 5(PPO 最小版)对目标读者零铺垫** → 先给 REINFORCE 五行版当台阶,或降级为带脚手架的引导实现。
- [ ] **P1-19 首页承诺兑现**:(a) C 级实验(checkpoint 加载)全书不存在 → 要么补一章 C 级实验(建议:加载 MiniGPT checkpoint 做 loss 曲线分析),要么删首页 C 级条目;(b) part1-perceptron 回声承诺「卷三做 linear probe」→ 卷三补一个 10 行的线性探测小实验(顺带呼应感知机),或改承诺。

### 卷三(其余)

- [ ] **P1-20 part7-sft.qmd FLAN 年份口径**:正文「2021 年的 FLAN [@wei2022flan]」与引用键 2022 打架 → 「2021 年发布(ICLR 2022)」。
- [ ] **P1-21 part8-agent-training.qmd 过程/结果奖励主文献缺席** → 论文时光机配对阅读处点名 Lightman et al. 2023(*Let's Verify Step by Step*, PRM 代表作)。
- [ ] **P1-22 part7-reasoning.qmd CoT 归属混写**:「加一句让我们一步步思考」是 Kojima et al. 2022(Zero-Shot CoT);Wei et al. 2022 是 few-shot 示例写推理链 → 拆开各给一句(两个都是好故事)。
- [ ] **P1-23 卷二→卷三 WRITING-STYLE 改例未落实**:part7-rlhf.qmd:59 与 part5-transformer.qmd:107 仍是规范文档里点名的「改前」原文 → 按 WRITING-STYLE.md 逐条落回。

---

## 五、P2:完善建议

### 全局

- [ ] **P2-1 术语统一**:「掩罩/掩码/遮罩」→ 统一「掩码」;省略号记号统一 $x_1,\dots,x_{t-1}$(修 part4-ngram:20、part4-rnn:49 的单句点笔误)。
- [ ] **P2-2 首页语病 3 处**:「读完你走过的……完整循环」缺谓语;「架构史因此,一部……」缺「是」;「你拥有的一座……图书馆」缺「了」。
- [ ] **P2-3 仓库链接自相矛盾**:_quarto.yml 注释是 `yujid/loss-descent`,index.qmd 是 `yujidong/loss-descent` → 确认真实地址后统一。
- [ ] **P2-4 图片无 alt 无题注**:全书插图(matplotlib PNG)无 alt、无 fig-cap → 补简短题注,利于无障碍与交叉引用。
- [ ] **P2-5 bib 作者名截断与缺失**:`Schmidhuber, J\"u`(缺 rgen)、`van Merri\"e`(缺 nboer)、`R\'e`(缺 je)、Gers 2000 缺第三作者 Cummins(而正文自己写了三作者);`Dess\`i` 后名字丢失;`[@elman1990]` 等键嵌进 scholar URL 字符串。
- [ ] **P2-6 引用位置与格式**:三个章节标题内嵌 `[@key]`(GPT-3 [@brown2020] 与 Scaling Laws 等)→ 移出标题;「2020 [@kaplan2020] 年」类插进年份中间 → 移句末。
- [ ] **P2-7 笔误清单**:「神经可力学」→「神经可塑性」(part1-perceptron:31);「枯结」→「枯竭」(part9-open-problems);「过旱承诺」→「过早承诺」(part5-transformer:212);「世界观的 选择」多空格(scaling-laws:145);「different 包衣」→「不同包衣」(agent-training);「linguistic 的老直觉」→「语言学」(word2vec:25);「实验few-shot」缺空格(base-model);「展示出Few-shot」大小写(gpt-vs-bert:30)。
- [ ] **P2-8 评价性形容词残留**(文风四):「有趣的镜像」「漂亮案例」「朴素得惊人」「博学得惊人」「戏剧性的消融」「惊呼」「贴地爬行(误用)」→ 按规范删除或中性化。
- [ ] **P2-8b 断句破损模式全书排查**(疑似批量编辑事故,卷一/卷三密度最高):三个模式——①缺系动词「是」(「这七十年的主线**一系列解题循环**」「需要的东西**一种计算顺序**」);②从句被句号截断(「当一篇新论文出现时。比如某个新的注意力变体。你依然没有判断力」);③假设从句无主句(「若它其实是个『万能记忆器』。」)。修 P0-3 时一并按这三个模式过滤。

### 卷二

- [ ] **P2-9 part4-ngram.qmd 压缩下界表述**:「按 q 编码恰好用这么多比特」可达而非下界;下界是 H(p) → 改「q 越接近真实分布越接近压缩下界」。
- [ ] **P2-10 part4-ngram.qmd:135 `sample_ngram` 的 n==1 分支会崩**(`tuple(标量)` TypeError)→ 修边界;读者做练习扫 n 即触发。
- [ ] **P2-11 part4-ngram.qmd:115 bit→nat 换算超界**:1.5 bit = 1.04 nat,「≈0.7–1.1 nat」→「≈0.7–1.0 nat」。
- [ ] **P2-12 part4-seq2seq.qmd 外推尾档样本过小**(长度 8 档约 33 条)→ 按指定长度精确生成或加大 n,并报告各档样本数;顺带在数据集说明处写明「另有 BOS/EOS 两记号」。
- [ ] **P2-13 part4-word2vec.qmd 语料数字与所引论文不符**:「10 亿词(词表 300 万)」——1301.3781 原文是「约 60 亿 token、词表限 100 万」;300 万词表出自 1310.4546 → 核对后按所引论文统一。
- [ ] **P2-14 part4-lstm.qmd GNMT「训练一次数周」**:论文报 96 块 K80 约 6 天 → 改「数天」。
- [ ] **P2-15 part5-why-transformer-won.qmd 练习 5 答案**:「KV 缓存只占固定几 MB」——KV 总量随上下文线性增长,固定的是每 token 占用 → 改措辞。
- [ ] **P2-16 part6-scaling-laws.qmd**:「跨越五个/六个数量级」口径统一;「Rosenfeld 2018」→ **2019**(arXiv 1909.12673, ICML 2020),并补 bib 条目;「全球最优超算数周」→「数千块 GPU 连训数周」。
- [ ] **P2-17 part6-convergence.qmd**:「Switch 每层几千专家」→ 上限 2048,改「两千级」;MoE 消融「保持总活跃算力近似」声明与代码不符(top-1 下活跃 FLOPs 减半,不变的是总参数)→ 改声明。
- [ ] **P2-18 part6-emergence.qmd**:char_acc 读数 0.44→0.42 与「趋平」叙述不符 → 如实写「尾段回落(argmax 在个别上下文翻转,loss 仍在改善)」,这反而是教学点;「25 倍跳变」实为 27 倍(0.054/0.002);char_acc 是离散量,「连续度量」改「更平滑的度量」;练习 5 提示的 p*=e^{-1/n} 补声明模型假设。
- [ ] **P2-19 part5-bahdanau.qmd 对齐热图重复训练**(同一实验已训好模型可复用)→ 复用,省约一分钟。

### 卷一

- [ ] **P2-28 术语/译名**:「非线形」→「非线性」(part1-backprop:194,221);「凸包」误用(convex hull 是计算几何专名,此处指 bump)→「鼓包」(mlp-winter);「迷题」vs「谜题」用字统一(capstone);「指数性衰减」→「指数式衰减」;「塞巴斯蒂安·霍克赖特」译名核对。
- [ ] **P2-29 part1-mlp-winter 「第二次寒冬」术语错位**:AI 史通行口径是 1974-1980 与 1987-1993 两次;本书把 1969 后称第一次 → 加脚注说明本书「两次寒冬」专指连接主义视角。「NeuralWare 上市」[待核实](上市的是 Nestor/HNC)。「宽度 3 卡在 ≈0.020(RMSE 约幅度七分之一)」数字自洽性重算(实际约 1/10)。
- [ ] **P2-30 part1-symbolic-vs-connectionist**:MYCIN「朴素贝叶斯味道」→ 改「certainty factors 启发了后来的不确定性推理」(学界公认 CF 概率语义有缺陷);「LISP 机器(……硬件。)」括号内多句号;「1986 年美国神经网络会议复苏」落实为具体会议名(Snowbird 1985/86 或 NIPS 1987)。
- [ ] **P2-31 part2-mlp-difficulty**:图题「million-dimension」与实际 65 参数网络不符 → 改「a 2-D slice of a 65-dim landscape」并说明真实网络才是百万维;练习 5 答案「每层方 fan-in」衍字;「错怪了局部极小值三十年(1986-2014)」→「近三十年」。
- [ ] **P2-32 part2-toolbox**:「He 初始化补了第一刀」归因拆开(He 治幅度,不治神经元死亡);删「惊呼」评价词;「工具箱买卖的从来『抵达旧目标的路况』」缺「是」;「但是， relu+BN」多空格。
- [ ] **P2-33 part2-alexnet**:ImageNet 成绩「2012-09-30 公布」模糊为「2012 年 9 月底」;dropout「2012 年初」→「2012 年年中」(arXiv 7 月);「训练一遍约 10^18 FLOPs」→「约 10^17-10^18(口径:只算前向/含反向)」并给学生看算式。
- [ ] **P2-34 part3-lenet**:「SUN-4 上训练数天」→ 1989 论文通行记载是 SUN-3 工作站约 3 天,且原句不通;账本 MSE 处补一句历史注(LeNet-5 原文也用 MSE,交叉熵是后来);练习 3 答案「Inception,3.2」改「3.2 开头提到的 GoogLeNet」。
- [ ] **P2-35 part3-resnet**:论文时光机图号核对(残差块是图 2 非「图 4」);Highway Networks 三作者补 Greff;「loss 贴地爬行」用反了(loss 卡高位)→「贴着初值爬行」;参考文献仅 2 条,补 Veit 2016 与 Highway;「但它在 6.1 里还会遇见」的「但」不通。
- [ ] **P2-36 part3-cnn-legacy**:Zeiler & Fergus 年份统一为「2013(arXiv)/ECCV 2014」;回声栏「SPA」未定义(疑 SAE 笔误)→ 核对统一。
- [ ] **P2-37 part3-cnn-limits**:WaveNet 时光机末句不通且方向可疑(「一年后 Transformer 出局它成为默认项」——实际 WaveNet 王座还坐了约两年)→ 改写;「$k=10$:呢?」断句;两处「?。」。
- [ ] **P2-38 part1-backprop**:1985 夏圣地亚哥工作坊年份 [待核实](PDP 序言);Cybenko 条目未在正文引用 → 补 [@cybenko1989] 或删。
- [ ] **P2-39 part1-perceptron**:Mark I「400 个可调电位器」[待核实](常见记载 400 光电元件/512 关联单元);「账本第一份且唯一」见 P1-36。
- [ ] **P2-40 part0-two-threads**:「人类偏好奖励的最大化(变形的 loss)」→「优化人类偏好的代理奖励模型(一个替身 loss)」;「练习 4 的常见发现是。」→「是:」;「这七十年的主线……」缺「是」。part0-reading-history:练习 3 答案「算力+数据+三件套」五项并列矛盾 →「算力、数据、方法三件套」。

### 卷三

- [ ] **P2-20 part7-base-model.qmd**:「GPT-3 千万美元级」→ 常见估算约 460 万美元(最终一次训练,Lambda 口径),改「数百万美元级」或注明口径;论文时光机图号指针(Figure 1.3/3.14)可疑 → 对照 arXiv 2005.14165 改节级指针;「(相变。)」修括号。
- [ ] **P2-21 part7-rlhf.qmd**:RM「一个回归头」→「输出标量分、在成对比较上以 Bradley-Terry 损失训练」;$P(w \succ r)$ 记号 r 双义 → 改 $P(y_w \succ y_\ell) = \sigma(r(x,y_w) - r(x,y_\ell))$;`dpo_step` 的 z 用总和、梯度按长度平均、win/lose 各走一步——是教学近似,加注释声明;历史链补 Ziegler et al. 2019;回声补 Constitutional AI/RLAIF 一条;「附录记录长度偏好」给出具体图表编号或删除「附录」字样。
- [ ] **P2-22 part7-sft.qmd**:LoRA 记号 ΔW=AB 与原文 BA 不同 → 加半句注;「微调的本质就是低秩」→「低本征维(Aghajanyan et al. 2020)」;`seq_logp`/`margin` 两个评测函数给前补一句 teacher-forcing 铺垫。
- [ ] **P2-23 part8-agent-training.qmd**:「蒸馏式闭环」与经典知识蒸馏区分一句;「WebArena ~14%」保留(准确)。
- [ ] **P2-24 part8-model-vs-system.qmd**:Liu et al. *Pre-train, prompt, predict* 引用年份改 2023(ACM CSUR 正式发表)。
- [ ] **P2-25 part9-patterns.qmd**:模式三时差表「~15 年」指代不明 →「到被主流采用约 15-20 年」;模式四「LSTM 细胞状态 → ResNet」加「后见之明的谱系重构」标注;标题「Bitter Lesson [@sutton2019]。规模化迟到但从不缺席」改排版。
- [ ] **P2-26 part9-open-problems.qmd**:补「幻觉与校准」为第五问(与 7.2 似然-解码分离伏笔收束),或至少点名。
- [ ] **P2-27 part9-prediction.qmd**:「工程 Discipline」→「工程学科」;Shannon 条目正文零引用 → 补引或删。

---

## 六、分章详录

### 卷一要点(完整意见见 §三/§四/§五 对应条目)

值得保留的亮点(修订时勿动):part1-backprop 的历史线(Linnainmaa 1970/Werbos 1974/Parker 1985 优先权处理准确且克制)+ 手算例 + 种子扫描,是全书范本级章节;part3-cnn-limits 的概率地板论证(k=3 得 0.51、k=10 得 1.00、出窗反例)「全卷概念最干净」;part3-cnn-legacy「感知机复活」回声闭环(「不是它变强了,是特征变好了」);part2-capstone「零件交互」(朴素初始化被 BN 救活)是到此最好的实证教学;part3-resnet 练习 3 关于 Pre-Act 的答案与「诚实脚注」(玩具引擎 20 层发散)都是范本段落。

卷一特有的系统性问题(前三卷共同问题之外):死链「1.6」×6(P1-24)、「第 1 章」歧义 ×4(P1-25)、「十年/十五年」口径 ×3(P1-26)、bib 系统性损坏最重(人名截断 Léon/J\"urgen/Aäron、标题截断只剩 *CNN*/*WaveNet*、scholar 链接缺第一作者搜不到原文、正文引用与文献表双向不齐——见 G5/§五 P2-5)。

### 卷二要点(完整意见见 §三/§四/§五 对应条目)

值得保留的亮点(修订时勿动):RNN 梯度剖面实验(56 个数量级的震撼读数,兑现「先被折磨再懂 All You Need」);LSTM 2.8 输给 RNN 2.23 的诚实记录;「两个 loss 不可直接比大小,尺子就不同」的方法论提醒;scaling 外推游戏对纯幂律失败 15% 的正面处理;Chinchilla 章「玩具配比≈600 vs 真实 20」的换算校准;涌现章「同一组模型、三种度量、三个故事」。

### 卷三要点(完整意见见 §三/§四/§五 对应条目)

值得保留的亮点:Goodhart 双塌法实验;诱导头复现失败的负结果教学;Part 9 可证伪预测框架;「STaR 是结果奖励的祖先」的诚实自我指认;模型 vs 系统章的三问框架。

### 代码库详录(冒烟测试 41/52 通过;11 项失败对应 5 个 P0 根因)

验证方法:52 项数值冒烟测试(有限差分 + torch 2.10 CPU 对照仲裁),已覆盖 autodiff.Value 全算子、MLP、Perceptron(Rosenblatt 逐步对拍)、MLPNumpy(BN+residual)、Conv2D/Conv1D(vs torch,误差 0/1.3e-16)、SimpleConvNet、SimpleRNN/LSTM(前向 vs torch 2e-7)、RNNLM/Seq2Seq、AttentionSeq2Seq、MiniGPT(causal mask「未来 token 对过去 logits 影响=0」验证、概率归一化 2e-16)、LoRA、MoE、SkipGram、Trainer。**现有 pytest 36 项全部通过但漏检了上述 P0**(原因:test_rnn.py 只对输出层 Why 做 FD,它在 cell.backward 之前计算;零初态回归测试掩盖了 t=0 分支;causal 模式不触 MLM 分支)。

章节 API 一致性抽查 12 章:import 路径、构造参数、方法调用与包实际 API 全部一致,无失效引用。

除 P0 外需要处理的可疑/建议项:

- [ ] **C-1(P1) LoRA「只训适配器」名不副实**(`model.py:38-54 vs 132-134`):`step()` 实际同时更新全部 LayerNorm g/b 和输出 head(declared=1152 vs 实际可训练=1403)→ 要么 LoRA 模式冻结 LN/head,要么 `n_trainable_params()` 如实计数并在 docstring 说明;7.2 章正文措辞对齐。
- [ ] **C-2(P1) 5.4 章 MLM 实验读数需在修 P0-26 后重新校准**并入 R2。
- [ ] **C-3(P2) `n_params()` 在 MoE 替换后崩溃**(`model.py:56-63`):`blk.mlp.fc1` 对 MoEMLP 抛 AttributeError(part6-convergence 恰好做此替换,未调用才没炸)→ 加分支。
- [ ] **C-4(P2) LoRA 无 α/r 缩放**(`layers.py:30-46`):ΔW=A@B,无标准 α/r;B 零初始化设计可训练性没问题,但 7.2 章若讲 α/r 会与代码不一致 → docstring 注明或补 α/r。
- [ ] **C-5(P2) `train.py` 只有动量 SGD**:`__init__.py` docstring 承诺「SGD → Adam → …」→ 改 docstring 或补 Adam。
- [ ] **C-6(P2) `set_seed` 管不到 `np.random.default_rng`**(legacy API)→ docstring 说明各模块靠显式 seed 参数。
- [ ] **C-7(P2) 杂项**:`sinusoidal_pos` 奇数 D 广播崩溃(加 assert);MoEMLP 无负载均衡 aux loss 是练习 3 有意留白(docstring 注明);`layers.py:138` 两分支相同冗余;`__init__.py` 未导出 MoEMLP;word2vec 负采样均匀且不过滤碰撞(标准是 unigram^0.75,注释说明);scalar.py 递归 DFS 深图触递归上限(注释适用范围);Conv 未实现 dX(已有注释,可接受)。

值得保留:命名/类型标注/docstring 总体优秀(中文教学注释、复杂度说明、历史注脚);perceptron/conv/word2vec 的 docstring 与实现一致;causal mask、门序(i/f/g/o)、遗忘门偏置默认值等关键细节全部正确。

---

## 七、执行提醒

1. **顺序约束**:R1 的代码类 P0(23-27)必须在 R2 实验重跑**之前**完成——先修梯度 bug,再重跑读数,最后改正文;
2. **每次批量修改后重渲染 + 跑 `python scripts/check_layout.py`**,确认没有引入新的渲染问题;
3. 修实验类问题(R2)时,**以重跑输出为准更新正文读数**,不要反向凑数;
4. 缓一缓重写(R3)建议按卷推进、每卷人工过一遍,这是唯一无法纯自动化的部分;
5. 全部 P0 修完后,在本文档勾选,作为发布 v1.2 的门槛。

---

## 八、给作者的几句话

这本书最难得的东西——问题链的诚实、零结果的记录、可证伪的预测——恰恰是最难模仿的部分,它们已经在纸面上了。剩下的是脏活:把「逐字可验证」的标尺用在自己身上。审稿中印象最深的三个瞬间:part4-rnn 那张 56 个数量级的梯度剖面图(讽刺的是,画它的代码自己带着一个梯度 bug)、part3-cnn-limits 用概率地板把 CNN 边界讲成一道证明题、Part 9 敢给每条预测写「会证伪我的观察」。修完 R1-R7,这本书配得上它的野心。
