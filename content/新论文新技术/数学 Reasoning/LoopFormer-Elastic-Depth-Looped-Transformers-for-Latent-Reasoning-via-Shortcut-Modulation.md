---
title: LoopFormer：可变预算的 Looped Transformer 隐空间推理
summary: ICLR 2026 论文 LoopFormer 将 looped Transformer 的隐空间推理建模为可调步长轨迹，用时间/步长调制与 shortcut-consistency 训练实现 elastic depth，让同一模型能在不同推理预算下稳定退化并持续 refinement。
created: '2026-05-04 00:00:00+08:00'
updated: '2026-05-04 00:00:00+08:00'
order: 2
pinned: false
year: 2026
venue: ICLR 2026
keywords:
  - looped transformers
  - latent reasoning
  - elastic depth
  - shortcut consistency
  - adaptive computation
  - parameter sharing
url: https://arxiv.org/abs/2602.11451
digest: "LoopFormer 将循环 Transformer 的隐空间推理建模为可变长度轨迹，通过时间与步长条件化及 shortcut-consistency 训练，使同一模型能在不同推理预算下平滑退化与持续改进。"
---

# LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation

## 基本信息

- 论文标题：LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation
- 作者：Ahmadreza Jeddi, Marco Ciccone, Babak Taati
- 机构：University of Toronto; Vector Institute; University Health Network
- 会议：ICLR 2026
- 论文链接：https://arxiv.org/abs/2602.11451
- OpenReview：https://openreview.net/forum?id=RzYXb5YWBs
- 项目页：https://loopformer.github.io/
- 代码：https://github.com/armenjeddi/loopformer
- 模型：https://huggingface.co/collections/armenjeddi/loopformer-68a4850eb3d0259b48cc7584
- 研究主题：Looped Transformer、latent reasoning、预算条件推理、参数共享模型、动态计算。

这篇论文讨论的是一个非常实际的问题：如果 looped Transformer 的卖点是“用共享参数重复计算来获得更深的有效推理”，那么它是否真的能在部署时自由选择循环次数？已有方法通常在固定 loop 次数上训练，也在固定 loop 次数上推理；一旦提前退出或延长循环，模型就进入训练外分布，表现容易停滞、退化甚至 collapse。LoopFormer 的目标是把这种“可变推理深度”从口号变成训练时显式优化的能力。

## 一句话总结

LoopFormer 把 looped Transformer 的重复计算看成从 hidden state `h(0)` 走向 `h(1)` 的隐空间轨迹，用 normalized time `t` 和 step size `Δt` 调制每次循环，并用 shortcut-consistency 让短轨迹对齐长轨迹，从而支持用户按预算选择推理深度。

## 背景与问题动机

Looped Transformer 的基本想法是：不堆叠很多不同参数的层，而是把一小段 Transformer block 重复执行多次。论文采用 `(k⊗L)` 表示 `k` 个共享 blocks 被循环执行 `L` 次；它的有效计算深度约为 `kL`，但参数量接近 `k` 层模型。这类模型在算法推理、隐式多步计算和 latent reasoning 上有明显吸引力，因为“重复同一个更新算子”天然像一个迭代求解器。

但已有 looped model 有一个部署层面的缺口：它们多数只会在固定 `L` 上工作。训练时如果只见过 8 次循环，推理时直接跑 4 次或 12 次，都可能变成分布外使用。短路到较小预算时，中间 hidden state 未必已经可用于预测；继续拉长循环时，共享 block 可能反复把表示推向相似状态，出现 stagnation。这样一来，looped Transformer 虽然参数省了，却没有真正获得“按预算弹性推理”的能力。

这也是 early exit 直接套到 looped model 上会脆弱的原因。普通非共享 Transformer 的第 6 层、第 12 层、第 24 层是不同参数的阶段性特征；但 looped Transformer 的每一步是同一个变换反复作用。如果训练目标只监督最后一步，早期步骤不会被迫成为有用 endpoint；如果额外 loop 没有时间或步长信息，模型也不知道自己是在“第几步、离终点多远、这一步应该粗略跳跃还是精细修正”。最终很容易形成一种近似固定点：多跑几步只是重复相似表示，而不是继续推理。

LoopFormer 解决的是预算条件推理问题：部署时用户给定一个计算预算 `M≤L`，模型不需要重新训练，就能执行 `M` 次循环并给出尽可能好的输出；预算越大，表示应当继续 refinement，而不是停滞。

## 方法详解

### 1. 把循环推理写成 normalized-time trajectory

LoopFormer 不再把第 `i` 次 loop 仅仅看成整数层号，而是把 hidden state 的演化看成单位时间区间 `[0,1]` 上的一条轨迹：

- 初始 token 表示为 `h(0)`。
- 最终目标表示为 `h(1)`。
- 推理预算为 `M` 时，选择一组时间点 `0=t_0<t_1<...<t_M=1`。
- 第 `i` 步的步长为 `Δ_i=t_i-t_{i-1}`，并满足 `Σ_i Δ_i=1`。

最大轨迹是 `L` 个均匀小步，即每步 `Δ_i=1/L`。推理时用户可以选择任意 `M≤L`，通常默认使用均匀 schedule `Δ_i=1/M`，也可以使用非均匀 schedule。

这个视角很关键：短预算不再是“提前退出某一层”，而是“用更粗的 solver step 走完同一个从 `t=0` 到 `t=1` 的表示变换过程”。这和 diffusion / consistency model 的思想非常接近：长路径是细粒度求解，短路径是粗粒度近似；训练目标要让不同离散化路径最终对齐到同一个 endpoint。

### 2. `t` 与 `Δt` 分别提供什么信息

LoopFormer 每次循环都条件化在 `(t_{i-1}, Δ_i)` 上。两者不是重复信息：

- `t` 表示当前位置：模型知道当前 hidden state 处于轨迹早期、中期还是末期。早期可以做粗略语义组织，中期进行更强的上下文融合，末期为 LM head 准备可预测表示。
- `Δt` 表示这一步跳多远：当预算很小、`Δt` 较大时，当前 loop 需要承担更粗的更新；当预算较大、`Δt` 较小时，每步更像细粒度 refinement。

如果只有 loop index 或 normalized time，模型知道“我在哪”，但不知道“这一步应该走多大”。如果只有 step size，模型知道“这一步多粗”，但不知道“当前处于整条轨迹哪个阶段”。LoopFormer 同时使用二者，目标是在不同 `M` 和不同 schedule 下都能保持轨迹一致性。

### 3. 架构：用 AdaLN/RMSNorm 风格调制共享 block

LoopFormer 是 decoder-only looped Transformer。每一步先把 `t` 与 `Δt` 分别做 sine-cosine Fourier features，再通过小 MLP 得到两个 embedding：

`e_t = φ(t)`  
`e_Δ = φ(Δt)`  
`c_i = e_t + e_Δ`

这个 conditioning 向量 `c_i` 输入到一个调制器，生成两类参数：

- RMSNorm scale：`γ_msa, γ_mlp`
- residual gate：`α_msa, α_mlp`

它们分别作用在 MHSA 和 FFN 分支：

`x ← x + α_msa ⊙ MHSA(RMSNorm(x) ⊙ (1 + γ_msa))`

`x ← x + α_mlp ⊙ FFN(RMSNorm(x) ⊙ (1 + γ_mlp))`

直觉上，LoopFormer 不是改变 Transformer 主体结构，而是在每次重复使用共享 block 时告诉它：“你现在处于哪一段轨迹，以及这一步是粗跳还是细修。”这比普通 parameter sharing 更像一个受控迭代求解器。

### 4. 训练目标：长轨迹监督 + 短轨迹监督 + consistency

总损失为：

`L = L_L + λ_1 L_S + λ_2 L_cons`

其中：

- `L_L`：最长 `L` 步轨迹的 next-token cross entropy。
- `L_S`：随机 shortcut 轨迹的 next-token cross entropy。
- `L_cons`：让短轨迹的表示或输出对齐到长轨迹的 stop-gradient target。

论文正文有一处把 consistency 描述为 per-token logits 对齐，但算法中写的是 `||stopgrad(h^(L)) - h^(S)||^2`。因此更稳妥的理解是：它在表示或 logit 层面对短路径和长路径做自蒸馏式对齐，具体以算法中的 hidden-state L2 表述为准。实验中 `λ_1=λ_2=0.1`。

训练时每个 batch 会采样：

- 完整最大轨迹 `Δ_L`。
- 一个 shortcut 长度 `S ~ Uniform{1,...,L-1}`。
- 一个长度为 `S`、总和为 1 的 step schedule `Δ_S`。

这样模型既学会 full-depth endpoint，也学会不同预算下直接走到 endpoint。它不是简单地要求所有中间层都能预测，而是要求不同离散化路径尽量收敛到同一个 `t=1` 表示。

### 5. 为什么 shortcut-consistency 像 diffusion / consistency model

在 diffusion model 中，长采样路径可以看成细粒度数值积分；consistency / shortcut model 试图让少步甚至一步采样逼近多步采样结果。LoopFormer 借用的是同一类思想，但对象从图像生成轨迹换成了语言模型 hidden-state trajectory。

对应关系大致是：

- diffusion state：带噪 latent 或样本状态。
- LoopFormer state：Transformer hidden states。
- fine solver：`L` 次均匀小步 loop。
- coarse solver：`S` 次 shortcut loop。
- consistency target：长路径 endpoint 的 stop-gradient 表示。
- 用户预算：推理时选择 `M` 次循环。

这种类比的价值在于，它把“提前退出”改写为“低步数求解同一个隐空间终点”。这解释了为什么单纯 early exit 不够：early exit 只是把中间状态拿来用，并没有训练它成为同一 endpoint 的粗粒度解。

## 图文并茂的讲解

![LoopFormer architecture and elastic-depth inference](/static/images/uploads/数学 Reasoning/loopformer-architecture.png)

上图适合从两个层次理解 LoopFormer。左侧是架构：输入 token embedding 后，模型反复执行同一个 `K` 层共享 Transformer stack。每次循环前，模型读取当前 normalized time `t_{i-1}` 和 step size `Δ_i`，生成 conditioning `c_i`，再调制 RMSNorm scale 与 MHSA/FFN residual gate。也就是说，共享 block 并不是盲目重复，而是被告知当前 loop 的“阶段”和“步幅”。

右侧是预算条件推理：`M=1`、`M=2`、`M=L` 都要从 `t=0` 走到 `t=1`，只是离散化粗细不同。低预算时，模型用较大步长直接走完整条轨迹；高预算时，模型用更多小步持续 refinement。理想状态下，低预算输出已经可用，高预算输出更准确。

![Shortcut consistency training and representation dynamics](/static/images/uploads/数学 Reasoning/loopformer-shortcut-consistency.png)

这张图可以对应训练机制。full path 走 `L` 个小步，产生高质量 endpoint；shortcut path 走较少步数，必须通过 CE loss 和 consistency loss 学会逼近 full path 的 stop-gradient target。这里的 stop-gradient 很重要：长路径充当 teacher，短路径学习对齐它，但不会反过来把 full path 拉坏。

下半部分可以理解 representation dynamics：普通 early-exit looped baseline 往往很快进入相似状态，跨 step CKA 高、曲率和熵变化平坦，说明额外循环没有真正产生新计算。LoopFormer 则在中间深度表现出更强的表示演化，末端再逐渐收敛，这更符合“先探索/融合，再整理到可预测 endpoint”的推理轨迹。

## 实验与结果分析

### 实验设置

论文使用 GPT-style decoder / NanoGPT 配置，在 The Pile 去重子集上训练约 25B tokens。主实验比较：

- 约 24 层、约 1B 参数的非共享 Transformer base。
- 固定深度 looped baselines，例如 Base-Loop、TMLT。
- early-exit 风格 depth-elastic baselines。
- LoopFormer。

评估包括三类 perplexity 数据集：The Pile、FineWeb-Edu、OpenWebText；以及十个 zero-shot reasoning / language tasks：COPA、HellaSwag、LAMBADA、OpenBookQA、PIQA、RACE、SciQ、ARC、SocialIQA、WinoGrande。

### 主要结果：强在 looped baselines 内，但没有完全替代非共享深层模型

在 `3⊗8`、24x FLOPs 预算下，LoopFormer 的 The Pile PPL 为 10.28，优于 Base-Loop 的 10.91，也略优于或接近 TMLT 的 10.38。平均 zero-shot accuracy 为 44.81，接近 24 层非共享 Base 的 45.27。

但这不能被解读为“参数共享完全替代非共享深层模型”。非共享 Base 的 The Pile PPL 是 9.49，仍明显好于 LoopFormer。这个差距说明，对语言建模 perplexity 来说，参数量和非共享层的记忆/表达能力仍然重要。LoopFormer 的贡献不是证明 looped model 全面胜过 dense depth，而是证明 looped model 可以在参数共享框架内获得更好的预算弹性和更稳的 latent refinement。

在 12x 和 6x 预算下，LoopFormer 的性能下降但不崩溃。12x 时，它在 average accuracy 上达到 43.73，接近 12 层 Base 的 44.93；6x 时表现更弱，但仍展示了从高预算到低预算的可控退化。相比之下，一些 early-exit baseline 在低预算下退化明显，说明它们并没有真正学到“短轨迹 endpoint”。

### 表示分析：避免 stagnation 是论文最有说服力的证据之一

论文用 curvature、anisotropy、prompt entropy 和 CKA 分析 loop steps。结论大致是：

- early-exit looped baselines 的指标曲线较平坦。
- 跨 step CKA 高，说明不同循环步的表示高度相似。
- LoopFormer 的表示沿深度持续漂移，中间阶段活动增强，末端收敛。

这组证据支持论文的核心机制解释：LoopFormer 不是简单让早期 hidden state 也能预测，而是让共享 block 在不同时间和步长下承担不同功能。它把重复执行从“趋向固定点”推向“沿轨迹演化”。

不过这些分析仍主要是相关性证据。曲率、熵、CKA 的变化说明模型内部状态更活跃，但不能严格证明它真的执行了可解释的多步推理算法。尤其在自然语言建模中，perplexity 与 reasoning benchmark 的关系复杂，表示变化不一定等同于推理步骤。

### 轨迹选择：同样预算下 schedule 也很重要

论文还研究了固定 `M` 下不同 `Δ_M` schedule 的影响。在 `3⊗8, M=4` 中，perplexity spread 约 1.4，accuracy spread 约 1.3；在 `2⊗12, M=6` 中，perplexity spread 接近 3。这说明预算相同不代表计算效果相同，step schedule 本身是一个重要控制变量。

较好的 schedule 倾向于“早期大步、后期小步”。这符合 coarse-to-fine 直觉：早期先快速推进到较成熟的语义状态，后期用小步精修预测分布。但论文也指出，perplexity 最优 schedule 与 reasoning accuracy 最优 schedule 不完全一致。这暗示未来可能需要按任务、输入甚至 token 动态选择 trajectory。

### 成本与可复现风险

LoopFormer 训练需要同时跑 full trajectory 和 shortcut trajectory，理论 FLOPs 约为固定训练的 1.5x，实测 wall-clock slowdown 约 1.3x。推理时成本与选择的 `M` 成正比，这正是它的部署价值所在。

但实验规模仍然有限：约 1B 参数、25B tokens，并不能直接推出在 frontier-scale LLM 上也会同样成立。更大规模下，KV-cache、生成式长上下文、多 token decoding、batching 和 serving latency 都可能改变实际收益。论文也尚未解决自动预算分配问题：目前预算是 global sequence-level，而不是 instance-level 或 token-level adaptive。

## 优点、局限与个人评价

这篇论文最有价值的地方，是把 looped Transformer 的“可变推理深度”形式化为 hidden-state trajectory consistency，而不是只做一个 early-exit 工程技巧。它清楚地区分了三件事：参数共享、重复计算、预算弹性。很多 looped model 只有前两者，LoopFormer 试图补上第三者。

第二个亮点是 `t + Δt` conditioning。TMLT 已经说明 timestep conditioning 有用，但 LoopFormer 进一步指出：如果希望同一个模型支持不同长度轨迹，光知道“当前时间”还不够，还要知道“当前步长”。这使模型更像一个可变步长的神经求解器。

第三个亮点是诊断做得比较贴近问题。论文没有只报 benchmark，而是用表示几何指标说明 naive early exit 为什么会 stagnate，LoopFormer 为什么更像持续 refinement。这对理解 looped model 是否真的使用 test-time compute 很重要。

局限也很明确：

- 它没有完全缩小与非共享深层 Transformer 的 perplexity 差距。
- 预算和 schedule 仍由用户或枚举策略指定，不是模型自动学习。
- consistency 训练增加训练成本，且更大规模下的工程开销还不清楚。
- 表示分析主要是相关性，缺少因果干预。
- 当前 elastic depth 是 sequence-level global budget，不是 token-wise adaptive compute。
- 对数学推理而言，zero-shot 平均准确率只能说明倾向，不能等价于强数学 reasoning 能力；论文没有在更专门的数学数据集上建立充分证据。

我的判断是：LoopFormer 不应被看作“替代普通深层 LLM 的新架构”，而应被看作 looped LLM 走向实用化的一块关键训练拼图。它最适合的定位是：当参数共享循环模型已经被用于 latent reasoning 时，如何让这个模型在不同 test-time compute 下可控、稳定、可诊断。

## 发散性研究思考

### 方法改进 Agent 视角

下一步最自然的改进是把 schedule 从人工选择变成 learned policy。LoopFormer 已经证明不同 `Δ_M` schedule 在固定预算下差异很大，那么统一使用 uniform schedule 只是一个保守 baseline。可以训练一个轻量 policy，根据输入难度、prompt entropy、早期 hidden-state 变化或 uncertainty 来选择 `M` 和 `Δ_M`。

另一个方向是把 global budget 改为 token-level 或 span-level budget。数学题、代码推理和长文问答中，并不是所有 token 都需要相同循环次数。可以将 LoopFormer 的 trajectory conditioning 与 Mixture-of-Recursions 式 token routing 结合：全局仍保持轨迹时间，但不同 token 选择不同递归深度。

### 实验验证 Agent 视角

论文需要更强的数学 reasoning 实验。当前十个 zero-shot benchmark 覆盖 commonsense、阅读、语言预测和科学问答，但不能充分证明“latent reasoning”在数学上的收益。后续应加入 GSM8K、MATH、AIME 风格任务、program synthesis、algorithmic length extrapolation，并区分有无 CoT prompting 的设置。

还需要更系统的 ablation：只用 `t`、只用 `Δt`、不用 consistency、只用 shortcut CE、不做 stop-gradient、不同 `λ`、不同 trajectory sampling distribution。论文已有部分消融，但如果目标是证明机制，最好能展示每个组件对应解决哪种 failure mode。

### 应用落地 Agent 视角

LoopFormer 的应用价值主要在 serving 侧：同一个模型可以根据延迟预算运行不同 loop 次数。例如移动端、在线问答、批量离线推理可以共享同一模型权重，只改变 `M`。这比维护多个不同深度模型更简单，也比 early exit 更符合 looped model 的结构。

但落地还要回答几个工程问题：生成式解码时每个新 token 都要重复 loop，KV-cache 如何跨 loop 复用？不同用户预算混在同一个 batch 中是否会造成吞吐下降？如果 schedule 不同，服务端如何高效 batch？这些问题会决定 LoopFormer 的实际性价比。

### 理论分析 Agent 视角

LoopFormer 很适合被解释为可变步长动力系统或神经 ODE 的离散求解器。`t` 与 `Δt` conditioning 让共享 block 近似一个非自治动力系统更新函数：

`h(t+Δt) = h(t) + F(h(t), t, Δt)`

理论上可以研究 shortcut-consistency 是否约束了不同离散化路径的 endpoint invariance，以及这种约束是否降低了 recurrent block 收敛到坏固定点的概率。还可以分析什么时候 looped Transformer 比 CoT 更有效：latent loop 适合并行内部计算，而 explicit CoT 更适合需要采样、计数和可验证中间步骤的任务。

### 研究趋势 Agent 视角

最近一年 looped Transformer 研究正在从“证明循环有推理归纳偏置”转向三个更具体的问题：

- 如何规模化预训练 looped LM。
- 如何做动态递归深度分配。
- 如何解释每一步循环到底学到了什么。

LoopFormer 位于第二和第三个问题之间：它没有像 Ouro 那样主攻大规模 looped LM，也没有像 MoR 那样做 token-level routing，而是聚焦在 trajectory robustness。它的意义在于提供了一个中间抽象：loop 不只是层数，而是一条可被调制、对齐和诊断的隐空间轨迹。

综合来看，LoopFormer 后续最值得探索的路线是：把 shortcut-consistency、learned schedule、token-level recursion routing、post-training trajectory reward 结合起来，形成真正可部署的 adaptive latent reasoning LLM。

## 相关论文推荐

下面是最近一年内最适合作为 LoopFormer 后续阅读的 looped transformer / latent recursion 论文阅读清单。建议阅读顺序是：先读 Ouro 理解规模化 looped LM，再读 MoR 理解 token-level adaptive recursion，然后读 RLTT 看 post-training 如何奖励 latent trajectory，最后读机制分析与泛化论文补足解释框架。

### 1. Scaling Latent Reasoning via Looped Language Models

- 链接：https://arxiv.org/abs/2510.25741
- 项目页：http://ouro-llm.github.io
- 时间：2025-10-29 提交，2025-11-17 修订

这篇论文围绕 Ouro / LoopLM family，关注大规模预训练中的 latent-space iterative computation，并引入 entropy-regularized learned depth allocation。它与 LoopFormer 的关系非常直接：LoopFormer 解决的是 shortcut schedule consistency 与 elastic depth；Ouro 更关注 looped LM 能否扩展到 7.7T tokens 级别并在较大模型上接近更大 dense model。

推荐先读它，因为它回答了 LoopFormer 没有完全覆盖的问题：looped language model 在更大预训练规模下是否仍然有价值。LoopFormer 的 1B / 25B tokens 实验更像机制验证；Ouro 更像规模化路线图。

### 2. Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation

- 链接：https://arxiv.org/abs/2507.10524
- 时间：2025-07-14 提交，2025-10-25 修订

MoR 把 parameter sharing 与 adaptive computation 结合，在 Recursive Transformer 中用轻量 router 为不同 token 分配不同 recursion depths，并考虑 attention / KV 的计算和内存效率。它与 LoopFormer 的核心差异是：MoR 是 token-level adaptive recursive depth，LoopFormer 是用户指定 global budget 和 trajectory schedule。

这篇论文适合用来思考 LoopFormer 的下一步。LoopFormer 证明了不同预算下轨迹可以对齐，但预算仍是全局指定；MoR 则提供了输入和 token 级别动态分配的方向。如果两者结合，可能得到既有 trajectory consistency 又有 token-level routing 的 looped LM。

### 3. Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models

- 链接：https://arxiv.org/abs/2602.10520
- 时间：2026-02-11 提交，2026-02-12 修订

这篇论文关注 LoopLM 的 post-training / RL 阶段，批评只奖励 final latent state 的 GRPO 式训练无法给整条 latent trajectory 分配足够信用，因此提出 RLTT，对 latent thought trajectory 做更密集的 reward / credit assignment。

它是 LoopFormer 的自然后续：LoopFormer 的 consistency loss 是预训练阶段的 trajectory alignment，目标是让短路径逼近长路径；RLTT 则是在强化学习阶段奖励整条隐式思考过程。读完 LoopFormer 后再读 RLTT，可以把“轨迹一致性”和“轨迹信用分配”连成一条线。

### 4. A Mechanistic Analysis of Looped Reasoning Language Models

- 链接：https://arxiv.org/abs/2604.11791
- 时间：2026-04-13 提交

这篇论文从 mechanistic analysis 角度研究 looped reasoning language models，讨论循环块如何收敛到 distinct fixed points，attention head 行为如何随 recurrence 稳定，以及 recurrent block size、input injection、normalization 对 cyclic fixed points 的影响。

LoopFormer 使用 curvature、anisotropy、prompt entropy 和 CKA 证明自己的表示没有像 early-exit baseline 那样停滞；这篇机制分析可以进一步解释“停滞、固定点、循环稳定性”到底来自哪些结构因素。它适合作为 LoopFormer 表示分析部分的深入补充。

### 5. Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers

- 链接：https://arxiv.org/abs/2604.07822
- 时间：2026-04-09 提交

这篇论文研究 recurrent-depth transformer 的 implicit multi-hop reasoning、systematic generalization 和 depth extrapolation，特别关注增加 inference-time recurrence 是否能解锁更深推理，以及过度循环时是否会出现 overthinking。

它与 LoopFormer 的关系在于：LoopFormer 解决 elastic-depth 训练和 shortcut schedule；这篇论文更关注循环深度如何影响系统泛化和隐式多跳推理。读它可以帮助判断 LoopFormer 的 elastic depth 是否真的可能带来更强 reasoning，而不仅仅是更平滑的 perplexity scaling。

补充背景：Reasoning with Latent Thoughts: On the Power of Looped Transformers（https://arxiv.org/abs/2502.17416）是 ICLR 2025 的核心背景论文，略早于“最近一年”窗口，但非常值得在读 LoopFormer 前后回看。它解释了为什么 looped Transformer 被认为具有 latent thought / reasoning 归纳偏置，LoopFormer 很大程度上是在这条线上进一步解决可变预算推理问题。

## 思维导图

```mermaid
mindmap
  root((LoopFormer))
    问题动机
      Looped Transformer 有 latent reasoning 归纳偏置
      传统方法固定训练与推理 loop 次数
      短预算 early exit 容易退化
      长预算重复循环可能 stagnation
      部署需要 elastic depth
    核心思想
      隐空间轨迹
        h(0) 到 h(1)
        normalized time t
        step size Δt
        用户选择预算 M
      Shortcut modulation
        t 表示当前位置
        Δt 表示单步跨度
        Fourier features
        MLP conditioning
      Block 调制
        RMSNorm scale
        MHSA residual gate
        FFN residual gate
        共享 K 层 block 重复执行
    训练目标
      Full trajectory CE
      Shortcut trajectory CE
      Shortcut consistency
        stop-gradient full path target
        短路径对齐长路径
        类似 diffusion consistency
      随机采样 shortcut length
      采样 step schedule
    实验结论
      Looped baselines 中最好
      24x 下 accuracy 接近 non-looped base
      perplexity 仍落后 non-shared deep model
      12x 和 6x 预算下平滑退化
      schedule 选择影响明显
    表示分析
      Early exit 表示平坦
      CKA 高说明跨步相似
      LoopFormer 中间深度持续演化
      末端逐渐收敛
      支持避免 stagnation 的解释
    局限
      训练开销约 1.3x wall-clock
      global budget 非 token adaptive
      表示分析偏相关性
      大规模 LLM serving 成本未充分验证
      schedule policy 尚未学习
    后续阅读
      Ouro 扩展 looped LM 规模
      MoR 做 token-level recursion routing
      RLTT 奖励 latent thought trajectory
      Mechanistic analysis 解释固定点
      Recurrent-depth generalization 研究 overthinking
```
