---
created: '2026-04-23 02:48:29.577390+00:00'
order: 0
pinned: false
summary: 用 7B 的小模型，通过"深度思考"（System 2 式推理），在数学基准上追平甚至超过 OpenAI o1，并且完全不依赖从大模型蒸馏数据。
title: rStar-Math
updated: '2026-04-23 02:48:29.577435+00:00'
---

## 一、论文的定位：它在解决什么问题？

理解一篇 paper 的第一步，永远是搞清楚它在 research landscape 的哪个缺口上。rStar-Math 处在三条线索的交汇处：

**线索 1：System 2 推理范式兴起。** OpenAI o1 展示了"测试时计算（test-time compute）"这个新 scaling 维度——不是把模型做大，而是让模型在推理时多想一会。这需要两个核心组件：一个能生成候选推理步骤的 **policy model**，一个能准确评估这些步骤的 **reward model**。

**线索 2：训练数据的瓶颈。** 两个模型都依赖高质量数据，但：
- Policy 的训练数据：传统做法是从 GPT-4 蒸馏 CoT，但这有两个致命问题——(a) 上限被 teacher 锁死，(b) 即便最终答案对，中间步骤也可能是错的（论文里引用了 Lanham et al. 2023 的发现），这种"对答案但错过程"的数据会污染训练。
- Reward model 的训练数据：PRM 需要 step-level 标注，人工标注（如 PRM800k）昂贵且不可规模化；自动标注（Math-Shepherd、AlphaMath 等）噪声大，训练出的 PRM 效果有限。

**线索 3：小模型是否只能做陪跑？** 在主流认知里，SLM（小语言模型）做数学推理必须靠大模型蒸馏。rStar-Math 要挑战的就是这个假设。

**这篇论文的核心主张（thesis）**：**不依赖更强模型做数据蒸馏，通过 MCTS + 过程奖励模型的自演化（self-evolution），SLM 完全可以自举到前沿水平。**

---

## 二、三大核心创新（把 packaging 剥开）

作者自己概括了三点贡献。我把它翻译成更精准的技术语言：

1. **Code-Augmented CoT 数据合成**——用 Python 代码执行作为 intermediate step 的"硬验证器"，配合大量 MCTS rollout 给每一步打 Q-value，产出"step-by-step verified trajectories"。

2. **Process Preference Model (PPM)**——不去直接预测每步的"绝对分数"（那不靠谱），而是用 Q-value 构造**步级偏好对**，用 Bradley-Terry 排序损失训练。这是对现有 PRM 训练方法的一次范式调整。

3. **四轮自演化 recipe**——policy SLM 和 PPM 从零开始，相互 bootstrap，每轮都比上一轮更强，产出更好的数据，训出更好的下一代模型。

接下来我按方法细节的顺序逐个拆。

---

## 三、方法 Part 1：MCTS 驱动的推理树

### 3.1 基本框架

给定问题 $x$ 和 policy model $M$，构造一棵搜索树：
- 根节点 = 问题 $x$
- 子节点 = 中间推理步骤 $s_i$（由 $M$ 生成）
- 叶节点 = 终止步（包含最终答案）
- 根到叶的路径 = 一个完整轨迹 $t = x \oplus s_1 \oplus s_2 \oplus ... \oplus s_d$
- 每一步 $s_i$ 都有一个 Q-value $Q(s_i)$

每次 MCTS 迭代四个阶段：**Selection → Expansion → Simulation → Backpropagation**（这是标准 MCTS，AlphaGo 同款）。

### 3.2 节点选择用的 UCT 公式

$$
\mathrm{UCT}(s) = Q(s) + c \cdot \sqrt{\frac{\ln N_{\text{parent}}(s)}{N(s)}}, \quad Q(s) = \frac{q(s)}{N(s)}
$$

- $Q(s)$：利用项（exploitation），这一步平均有多好。
- 平方根那一项：探索项（exploration），访问次数越少的节点"性价比"越高。
- $c$：平衡参数。论文里**设为 2**，相对激进地鼓励探索。

> **直觉**：MCTS 其实就是"聪明版的 best-of-N"。Best-of-N 是在完整轨迹级别上盲目采样；MCTS 是在每一步决策时，用历史统计（Q-value）+ 不确定性（探索项）做加权选择，把计算预算花在有前途的分支上。

### 3.3 关键创新：Code-Augmented CoT（Figure 2 的核心）

这是**第一大贡献**的具体做法。在之前的 MCTS-for-math 工作里（rStar、MCTS-self-refine），每一步是自然语言 CoT。问题：LLM 会"侥幸对"——中间步骤有幻觉/错误，但最终答案蒙对。这种数据没法自动筛除。

rStar-Math 的做法：**每一步的输出必须是"NL CoT（作为 Python 注释）+ 可执行的 Python 代码"**。比如求对角线距离：

```python
# Step 1: Calculate the total distance walked south
total_south = 1/2 + 1/2
```

**验证机制**：在第 $i$ 步，把当前步的代码 **拼接上 1 到 i-1 步所有历史代码**，然后真的跑 Python。**只有代码执行成功的候选节点才保留**。

这等于给每一步塞了一个"廉价的形式化验证器"——不能保证数学逻辑对，但能**保证符号操作、计算、代码逻辑没炸**。这是把 tool use 和 process verification 融合的巧妙做法。

### 3.4 Q-value 的两种标注方法

每步的 Q-value 怎么来？这是 MCTS 能否 work 的关键。论文给了两种机制，分别对应自演化的不同阶段：

**方法 A：Terminal-guided 标注（用于 Round 1-2）**

没有可靠 PPM 时，靠"答案对不对"反向传播。更新公式：

$$
q(s_i)_k = q(s_i)_{k-1} + q(s_d)_k
$$

- $q(s_i)_k$：第 $k$ 次 rollout 后，步骤 $s_i$ 的累积 q 值
- $q(s_d)$：终止节点 reward，**对 = +1，错 = -1**
- 初始 $q(s_i)_0 = 0$

**直觉**：如果某一步在多次 rollout 中常常通往正确答案，Q-value 就累积得高；反之为负。这就是 AlphaGo 式的蒙特卡洛评估。**缺点**：需要大量 rollout 才能收敛，且对"一次侥幸对"不够鲁棒。

**方法 B：PPM-augmented 标注（用于 Round 3-4）**

有了可靠 PPM 后，直接用它给出初始 q 值：

$$
q(s_i)_0 = \mathrm{PPM}(x \oplus s_1 \oplus ... \oplus s_i)
$$

然后再按上面的公式做反向传播更新。好处：**一开始就有 informative 的先验**，不需要大量 rollout 就能得到好的 Q-value 估计。

> 💡 **这里体现了自演化的精髓**：PPM 一开始训不好（没好数据），所以先用 terminal-guided；等 PPM 变强了，又能反过来让 MCTS 数据质量更高。这是一个 bootstrap 循环。

---

## 四、方法 Part 2：Process Preference Model (PPM)

这是**第二大贡献**，也是论文技术上最有辨识度的一点。

### 4.1 动机：为什么不直接拿 Q-value 训 PRM？

之前的主流做法（Math-Shepherd、AlphaMath、ReST-MCTS*）：把 MCTS 得到的 Q-value 作为"真值分数"，用 **MSE loss** 或 pointwise loss 训 PRM，让它学会"预测每一步的分数"。

rStar-Math 作者的 insight（我觉得是全文最重要的洞察之一）：

> **Q-value 作为"分类信号"是可靠的（区分好坏步骤），但作为"回归目标"是不可靠的（给不出精确的绝对分数）。**

论文原话（paraphrase）：即便跑了大量 rollout，想精准区分"最好""次好""中等"的正确步骤几乎不可能，人类专家标注都做不到一致。那你用这些"含噪绝对分数"拿去 MSE 拟合，**模型就是在学噪声**。

### 4.2 解法：转成偏好学习（Preference Learning）

既然绝对值不可信，但相对高低可信，那就**造偏好对**来学。

**构造方法**（参见 Figure 1(b)）：
- 中间步骤 $s_i$：在同一个前缀下（共享 $s_1, ..., s_{i-1}$），从 MCTS 树里挑 Q-value 最高的 2 个作为 positive，最低的 2 个作为 negative。**且 positive 必须最终通向正确答案，negative 必须通向错误答案。**
- 最终答案步：因为相同前缀很难出不同最终答案，放宽约束——直接选 Q-value 平均最高的 2 条正确轨迹为 positive，最低的 2 条错误轨迹为 negative。

**训练损失**（标准 Bradley-Terry + pairwise ranking）：

$$
L_{\mathrm{ppm}}(\theta) = -\frac{1}{4}\mathbb{E}_{(x, y_i^{\mathrm{pos}}, y_i^{\mathrm{neg}})\in D}\left[\log \sigma\left(r_\theta(x, y_i^{\mathrm{pos}}) - r_\theta(x, y_i^{\mathrm{neg}})\right)\right]
$$

其中 $y_i^{\mathrm{pos}} = s_1 \oplus ... \oplus s_{i-1} \oplus s_i^{\mathrm{pos}}$，$r_\theta$ 是 PPM 给轨迹前缀的评分。

> **对做 RLHF 的读者**：这其实就是把 InstructGPT 那套 reward model 的 pairwise ranking loss 从"response 级"搬到"step 级"。概念上非常干净。

### 4.3 架构小细节

- PPM 从当轮 fine-tune 过的 policy model **初始化**
- 原来的 next-token prediction head 换成**标量 value head**（一层 linear + tanh，输出范围 [-1, 1]）
- 训练 1 epoch，batch size 512，lr 7e-6

---

## 五、方法 Part 3：四轮自演化 Recipe

这是**第三大贡献**——把前面两个组件组织成一个迭代提升的流程。数据起点是 747k 道数学题（NuminaMath 竞赛级 + MetaMath + GPT-4 合成的 AMC/AIME 风格题）。

Table 2 是核心证据，展示每轮的数据覆盖率：

| Round | Policy Model | PPM | GSM-level | MATH-level | Olympiad | 整体 |
|-------|-------------|-----|-----------|------------|----------|------|
| 1 | DeepSeek-Coder-V2-Instruct (236B) | — | 96.61% | 67.36% | 20.99% | 60.17% |
| 2 | SLM-r1 | — | 97.88% | 67.40% | 56.04% | 66.60% |
| 3 | SLM-r2, PPM-r2 | ✓ | 98.15% | 88.69% | 62.16% | 77.86% |
| 4 | SLM-r3, PPM-r3 | ✓ | 98.15% | 94.53% | 80.58% | 90.25% |

每轮的角色分工：

**Round 1（Bootstrap）**：没有 SLM 也没有 PPM。直接用 236B 的 DeepSeek-Coder-V2-Instruct 跑 MCTS（8 rollouts，terminal-guided），挑 top-2 正确轨迹做 SFT，得到 **SLM-r1**。这一步是在解"冷启动"——第一代 SLM 的老师是开源大模型，但**不是蒸馏 teacher 的 CoT 文本，而是蒸馏 teacher 参与的 MCTS 搜索树里被验证过的轨迹**。

**Round 2（第一个 PPM）**：policy 换成 SLM-r1（只有 7B，但已经有 code-augmented CoT 能力）。做 **16 rollouts/题**（多了一倍），Q-value 更可靠，首次训出 **PPM-r2**。

**Round 3（PPM-augmented MCTS）**：用 PPM-r2 指导 MCTS，数据质量大幅跃升。关键证据：Olympiad 级别的题覆盖率从 56% → 62%，MATH 级别从 67% → 88%——**PPM 确实大幅提升了难题的求解能力**。产出 SLM-r3 和 PPM-r3。

**Round 4（攻坚）**：对最难的题，把 rollout 数提到 64 甚至 128，再用不同随机种子跑多棵树。Olympiad 覆盖率到 80.58%。**剩下 ~10% 的未解题经人工抽检 20 道，19 道是 GPT-4 合成时 ground truth 就错了**——作者据此判断已经摸到了数据集的噪声天花板，停止演化。

---

## 六、实验讲解（重点看几个关键结论）

### 6.1 主结果（Table 3）

数字不用全记，关键 take-away 有三个：

- **Qwen2.5-Math-7B**（base 58.8%）→ rStar-Math **90.0%** on MATH（64 trajectories）。提升 31 个点。
- **AIME 2024**（高中奥赛级别）：rStar-Math 7B 做到 **53.3%**，超过 o1-preview 的 44.6%，逼近 o1-mini 的 56.7%。
- 连 1.5B 的小模型（Qwen2.5-Math-1.5B + 7B PPM）都能做到 MATH 88.6%、AIME 46.7%，已经**全面超过 72B 模型 + 72B ORM** 的组合。

### 6.2 Ablation 1：Step-by-step verified trajectories vs. 其他 SFT 数据（Table 5）

这是验证"**第一大贡献（code-augmented CoT）**"。在 Qwen2.5-Math-7B 上，SFT 数据对比：

| 数据来源 | MATH | AIME | Olympiad | College |
|---------|------|------|----------|---------|
| GPT-4 蒸馏的 NuminaMath-CoT | 69.6 | 10.0 | 37.2 | 43.4 |
| 自采样（无验证） | 72.4 | 10.0 | 41.0 | 48.0 |
| 拒绝采样（ORM 筛） | 73.4 | 13.3 | 44.7 | 50.8 |
| **Step-by-step verified（本文）** | **78.4** | **26.7** | **47.1** | **52.5** |

**解读**：
- 哪怕是**自采样**（policy 自己随机产）都已经超过 GPT-4 蒸馏——这很震撼，说明 SLM 经 4 轮自演化后，产生的 CoT 质量已经 >> GPT-4 的蒸馏数据。
- Rejection sampling（outcome-level 验证）再涨一点；**step-level 验证**再涨一大截（AIME 从 13.3 → 26.7）。**密集验证远优于稀疏验证**——这是用实验支撑了论文的 thesis。

### 6.3 Ablation 2：PPM vs. ORM vs. PQM（Table 6）

验证"**第二大贡献**"。同样的训练数据，不同的 RM 架构：

| RM | Inference | MATH | AIME | Olympiad | College |
|----|-----------|------|------|----------|---------|
| ORM（结果奖励） | Best-of-N | 82.6 | 26.7 | 55.1 | 55.5 |
| PQM（Q-value MSE） | MCTS | 88.2 | 46.7 | 62.9 | 57.6 |
| **PPM（偏好学习）** | **MCTS** | **89.4** | **50.0** | **65.3** | **59.0** |

**解读**：
- ORM → PQM 的跃升（尤其 AIME 26.7 → 46.7）说明**密集的 process 级信号确实关键**。
- PQM → PPM 的再次提升（尤其在 MATH/Olympiad 等难题上）说明**把"回归绝对值"换成"学偏好排序"确实更鲁棒**——这验证了作者关于 Q-value 噪声的那个 insight。

### 6.4 Ablation 3：Test-time scaling（Figure 3）

越多 MCTS 轨迹 → 越高准确率。但这里有个重要观察：**在 MATH、AIME、Olympiad 上，rStar-Math 只用 4 条轨迹就超过了 Qwen 72B Best-of-N 的 64 条**。说明效率远高于纯 Best-of-N。

### 6.5 意外发现（Findings，Appendix A.1）

这部分对**研究直觉**非常有启发：

**(1) 自反思（self-reflection）自发涌现**。Figure 4 是个典型例子：SLM 开始用 SymPy 形式化求解一道正整数方程，前三步 PPM 打分都是负的（-0.08, -0.219, -0.348）；结果模型在第四步**自己主动换路子**——直接暴力搜索正整数解，PPM 打分立刻转正（0.620），最终答对。**训练数据和 prompt 里都没有任何 self-reflection 的信号**。这是 emergent 现象，和 o1 的宣传描述高度相似。

**(2) PPM 决定推理上限**。Figure 5 显示：不同 size 的 policy model，pass@1 差异很大；但配上同一个 PPM 做 System 2 推理，最终精度**收敛**。这对你做方向选择有很强的启示：**在 System 2 范式下，可能"把 RM 做好"比"把 policy 做大"更重要**。

**(3) PPM 能识别"定理应用步"**。论文观察到 PPM 在关键定理（费马小定理、韦达定理、AM-GM 不等式、毕达哥拉斯定理、鞋带定理）应用的步骤上打分异常高，等于隐式学会了"什么是数学中的关键 insight 步骤"。

---

## 七、批判性解读（这部分你做科研必须有）

我必须给你指出这篇论文的一些局限和应该警惕的地方——读论文不是只看它宣传的优点。

**1. 计算成本被淡化。** Round 1 用 236B 的 DeepSeek-Coder-V2 + 10 nodes × 8× H100 跑了**两周**才把第一轮数据产出来；后续每轮也要 15 nodes × 4× A100 × 3 天；Round 4 是 1 周。**总训练计算成本不小**，自演化"从小模型自举"的叙事有一点 oversell——Round 1 的 bootstrap 本质上还是靠 236B 老师模型。

**2. "超过 o1" 的比较需要小心看。** rStar-Math 用 64 条 MCTS 轨迹 + PPM 重排；o1-preview 的测试时计算预算是 OpenAI 的黑盒默认值。这不是在"相同 test-time budget"下的对比。

**3. Code-augmented CoT 的领域适用性。** 它严重依赖"每一步都能转成可执行 Python"。对数学 word problem 和计算题非常适合，但**对纯定理证明、几何推理、组合计数里的高度符号/直觉步骤**，代码验证能验的东西很有限。作者在结尾也承认这是 future work。

**4. PPM 训练数据构造的潜在偏置。** 正负样本都来自同一个 policy model 的 MCTS 树，policy 的分布偏置会直接传到 PPM。**PPM 实际上在学"当前 policy 分布下什么样的步骤更有潜力"，不一定等于"客观上什么样的步骤更数学正确"。** 这可能解释了为什么不同 size 的 policy 在同一个 PPM 下收敛——PPM 某种程度上在对齐一个共同的"推理风格"。

**5. 终止奖励完全靠 ground truth。** 现实中的开放数学问题（尤其研究级别）没有 ground truth。这方法的一个隐含 assumption 是"所有训练题都有可验证的答案"。这也是为什么他们必须过滤 GPT-4 合成的劣质题。

**6. 数据集对"基础功"题的偏见**。作者提到"grade-school-level 题不能显著提升复杂推理"，只保留竞赛级。这个 claim 没做消融，可能对一些基础推理任务是有损失的。

---

## 八、对你研究的启发

结合你在做"数学推理能力提升"，我挑几个可以直接复用的点：

1. **"Step-level verification 密度 > Outcome-level"** 是 well-established 的实验结果。你如果做 PRM 相关，这篇论文是你必须引用的 baseline，Table 5 的 setup 也值得复现对比。

2. **Code-augmented CoT 作为 cheap verifier 是个很通用的技巧**——可以迁移到很多场景（比如把 tool call 输出当 step verifier）。

3. **"Q-value 适合做偏好信号，不适合做回归目标"** 这个 insight 值得进一步探索：能不能用更精细的信号源（比如多个 PRM 投票、形式化证明器信号）来取代 Q-value，进一步提升 preference data 的质量？这是一个潜在的研究课题。

4. **Policy / RM 的 bootstrap 自演化** 是一个 generalizable 的 pipeline 模板。你的工作如果要做数据自生成，可以参考这个四轮流程的设计思路，尤其是每轮"针对性解决一个瓶颈"的做法（round 1 解决冷启动、round 2 解决 RM、round 3 放大数据质量、round 4 攻坚难题）。

5. **如果你想挑战这篇 paper**：最有可能的攻击面是第三节的第 4 和第 6 点——PPM 的 policy 偏置问题、以及在非 word-problem 场景（定理证明、几何）的泛化性。这都是 open problem。