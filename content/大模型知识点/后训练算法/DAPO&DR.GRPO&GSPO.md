---
created: '2026-04-22 15:16:21.447976+00:00'
order: 3
pinned: false
summary: GRPO 改进
title: DAPO&DR.GRPO&GSPO
updated: '2026-04-22 15:16:21.448008+00:00'
---

> 本文详细解读三篇 GRPO 改进的代表性工作：
> - **Dr.GRPO**（2025.03）：纠正 GRPO 的优化偏差
> - **DAPO**（2025.03）：字节跳动开源的 32B 级 RL 系统
> - **GSPO**（2025.07）：Qwen 团队的序列级策略优化
>
> 这三篇论文分别从**数学正确性**、**工程可用性**、**算法根基**三个角度改进了 GRPO，是 2025 年大模型 RL 领域最重要的三项工作。

---

## 一、论文全景概览

| 算法 | 时间 | 机构 | 核心贡献 | 一句话总结 |
|------|------|------|----------|------------|
| **GRPO** | 2024.02 | DeepSeek | 组内归一化优势，去 critic | 起点 |
| **Dr.GRPO** | 2025.03 | Sail-SG (NUS) | 去掉 1/\|o\| 和 1/σ 归一化 | GRPO "Done Right" |
| **DAPO** | 2025.03 | 字节跳动 | Clip-Higher + 动态采样 + token 级 loss + 超长塑形 | 工业级四件套 |
| **GSPO** | 2025.07 | 阿里 Qwen | 序列级重要性比 + 序列级裁剪 | 从根上重构 IS |

**它们解决的问题各不相同：**

- **Dr.GRPO** 说：GRPO 的公式本身就有偏差，需要先"纠偏"；
- **DAPO** 说：GRPO 在大规模长 CoT 场景下训练不稳，需要一系列工程补丁；
- **GSPO** 说：GRPO 在 token 级做重要性采样是理论错误，需要在序列级重新定义。

---

## 二、Dr.GRPO：纠正 GRPO 的隐藏偏差

### 2.1 论文信息与核心贡献

- **论文标题**：*Understanding R1-Zero-Like Training: A Critical Perspective*
- **arXiv 编号**：2503.20783
- **发布时间**：2025 年 3 月 26 日（COLM 2025）
- **作者单位**：Sail-SG（新加坡国立大学 Sea AI Lab 团队）
- **代号**：**Dr.GRPO = GRPO Done Right**

**核心贡献：**

1. 批判性地分析 R1-Zero 式训练，发现 **GRPO 的优化目标存在系统性偏差**；
2. 提出修正版 **Dr.GRPO**，去掉两个有偏的归一化项；
3. 用 Qwen2.5-Math-7B + MATH 数据，**仅用 27 小时 8 张 A100** 就在 AIME 2024 上达到 43.3%，当时的 SOTA。

### 2.2 GRPO 的两个隐藏偏差

回顾 GRPO 的目标函数（简化记号）：

$$
\mathcal{J}_{\text{GRPO}} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min(r_{i,t} \hat{A}_{i,t},\ \text{clip}(\cdots) \hat{A}_{i,t}) \right], \quad \hat{A}_{i,t} = \frac{R_i - \text{mean}(R)}{\text{std}(R)}
$$

Dr.GRPO 指出这里藏着**两处偏差**：

#### 偏差 1：长度归一化偏差（Length Bias）—— $\dfrac{1}{|o_i|}$

对每个样本除以序列长度 $|o_i|$，会造成：

- **负优势时**（答案错误，$\hat{A}_{i,t} < 0$）：长度越长，单个 token 承担的惩罚**越小**；
- **正优势时**（答案正确，$\hat{A}_{i,t} > 0$）：长度越长，单个 token 获得的奖励也越小。

**后果**：当答案错误时，模型**更倾向于写得更长**来稀释惩罚！这正是 GRPO 训练中**错误响应越来越长**的根源。

一张图概括论文的实验现象：

```
GRPO 训练曲线：
  - 准确率上升 → 合理
  - 正确响应长度增长 → 合理（长 CoT）
  - 错误响应长度也持续增长 → 异常！此即长度偏差
```

#### 偏差 2：标准差归一化偏差（Difficulty Bias）—— $\dfrac{1}{\text{std}(R)}$

用组内标准差归一化优势，会造成：

- **低方差的组**（题目太简单全对，或太难全错，std 很小）：
  该组的梯度被放大
- **高方差的组**（有对有错，std 较大）：
  该组的梯度被缩小

**后果**：模型被引导去"刷"那些**要么全对、要么全错的极端难度题目**，而忽视真正有学习信号的中等难度题。这与我们希望的"在挑战性问题上学习"相反。

### 2.3 Dr.GRPO 的修正公式

Dr.GRPO 的修正非常简洁——**去掉这两个归一化项**：

$$
\boxed{
\mathcal{J}_{\text{Dr.GRPO}} = \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \left[ \min(r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t}) \right]
}
$$

其中优势也简化为 **不做 std 归一化**：

$$
\hat{A}_{i,t} = R_i - \text{mean}(R_1, \dots, R_G)
$$

实现上只需改两行代码：

```python
# GRPO (有偏)
advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
loss = (policy_loss / response_lengths).mean()

# Dr.GRPO (无偏)
advantage = rewards - rewards.mean()
loss = policy_loss.sum() / (batch_size * max_length_or_constant)
```

实际代码里通常用**一个常数**（如最大长度或固定 scale）除一下，保证梯度数值稳定性，但**不除可变的 $|o_i|$**。

### 2.4 实验现象与结论

论文做了一组非常干净的消融实验（Figure 8）：

| 变体 | 准确率 | 正确响应长度 | 错误响应长度 |
|------|--------|--------------|--------------|
| Vanilla GRPO | 高 | ↑ | **↑↑（异常增长）** |
| GRPO w/o length norm | 高 | ↑ | 平稳 |
| GRPO w/o std norm | 高 | ↑ | ↑↑ |
| **Dr.GRPO** | 高 | ↑ | **平稳** |

**结论**：去掉长度归一化是修复"错误响应无限变长"的关键；去掉 std 归一化则消除难度偏差。两者结合得到无偏 Dr.GRPO。

最终结果：Qwen2.5-Math-7B + Dr.GRPO + MATH level 3-5 题 + Qwen-Math 模板，**8×A100 仅训练 27 小时**，在 AIME 2024 达到 43.3% 的 SOTA。

### 2.5 关键启示

1. **不要盲从"看起来很合理"的归一化**：除以长度看似公平，实则引入方向性偏差；
2. **关注错误响应的行为**：训练中错误响应的长度演化是重要的诊断信号；
3. **简洁即美**：修正只需去掉两个除法，无需额外模块；
4. Dr.GRPO 在学术层面更严谨，但**在工业场景中 DAPO 的一揽子改进更常被采用**（见下节）。

---

## 三、DAPO：开源工业级 RL 系统

### 3.1 论文信息与背景

- **论文标题**：*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*
- **arXiv 编号**：2503.14476
- **发布时间**：2025 年 3 月
- **作者单位**：字节跳动 Seed + 清华 AIR
- **代号**：**DAPO = Decoupled Clip and Dynamic sAmpling Policy Optimization**

**背景动机**：DeepSeek-R1 虽然发表了技术报告，但许多关键训练细节未公开，社区复现一直困难。字节团队以 **Qwen2.5-32B 为底模，用 DAPO 在 AIME 2024 上达到 50 分**，用 **50% 的训练步数** 超越了 DeepSeek-R1-Zero-Qwen-32B 的 47 分。

**四大核心技术：**

1. **Clip-Higher**：解耦上下裁剪阈值，防止熵坍缩
2. **Dynamic Sampling**：动态过滤掉全对/全错的组
3. **Token-Level Policy Gradient Loss**：token 级损失
4. **Overlong Reward Shaping**：超长响应的软惩罚

此外 DAPO **彻底去掉了 KL 散度项**，认为长 CoT 任务中策略分布本就需要大幅偏离初始模型。

### 3.2 Clip-Higher：解耦裁剪上下界

#### 问题：熵坍缩（Entropy Collapse）

标准 PPO/GRPO 的裁剪是对称的：

$$
\text{clip}(r_t, 1 - \varepsilon, 1 + \varepsilon)
$$

例如 $\varepsilon = 0.2$ 时，一个概率为 0.9 的 token 最多被更新到 $0.9 \times 1.2 = 1.08$（裁到 1.0），而一个概率为 0.01 的探索性 token 最多被更新到 $0.01 \times 1.2 = 0.012$。

**结果**：高概率 token 得到充分更新，低概率（探索性）token 被严重抑制，策略熵迅速下降，模型变得死板、缺少多样性。

#### 解决：解耦 $\varepsilon_{\text{low}}$ 与 $\varepsilon_{\text{high}}$

DAPO 把裁剪的上下界拆开：

$$
\text{clip}(r_t, 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}})
$$

- **增大 $\varepsilon_{\text{high}}$**（例如 0.28）：给低概率 token 更大的"上升空间"，鼓励探索
- **保持 $\varepsilon_{\text{low}}$ 较小**（例如 0.20）：避免把某个 token 的概率压到 0（因为再压就是指数级损失）

论文的超参：$\varepsilon_{\text{low}} = 0.20,\ \varepsilon_{\text{high}} = 0.28$。

**效果**：熵保持在健康水平，避免了坍缩，模型探索能力显著增强。

```python
# Clip-Higher 的 PyTorch 实现
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio,
                                      1 - clip_ratio_low,     # 0.20
                                      1 + clip_ratio_high)    # 0.28
pg_loss = torch.maximum(pg_loss1, pg_loss2)
```

### 3.3 Dynamic Sampling：动态采样

#### 问题：零优势（Zero Advantage）

GRPO 用组内均值做 baseline，如果一组 $G$ 个样本的奖励**全都相同**（全对 $r_i = 1$ 或全错 $r_i = 0$）：

$$
\hat{A}_i = r_i - \bar{r} = 0, \quad \forall i
$$

**该组对梯度贡献为 0**，相当于"白采样"！

随着训练进行，简单题会越来越多被全对，导致有效梯度信号的样本比例持续下降。论文实验显示：训练后期，**超过 60% 的 prompt 属于"全对"或"全错"组**。

#### 解决：动态采样 + 过采样

DAPO 的策略：

1. **过采样**：每次采样的 prompt 数量大于 batch size（例如 $3\times$ 实际需要）；
2. **过滤**：剔除所有"全对"或"全错"的组；
3. **累积**：如果剩下的组不够一个 batch，再采样一批继续过滤，直到凑够；
4. 为避免无限采样，设置上限（例如 10 个 generation batch）。

```yaml
# DAPO 动态采样的典型配置
use_dynamic_sampling: true
num_prompts_per_step: 512                 # 目标 batch
num_generations_per_prompt: 16            # 每 prompt 采样数
batch_multiplier: 3                       # 数据加载器倍数
dynamic_sampling_max_gen_batches: 10      # 最大累积轮数
```

**效果**：每个 batch 都是"有信号"的样本，训练效率显著提升，收敛速度加快。

### 3.4 Token-Level Loss：Token 级损失

这一点与 Dr.GRPO 的发现**不谋而合**——都是针对长度归一化偏差。

#### 问题：序列级平均导致长样本权重偏低

原版 GRPO 的聚合方式是：

$$
\mathcal{L}_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G} \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \ell_{i,t} \right]
$$

**先在样本内求 token 平均，再在样本间求平均**。结果：
- 一个 1000 token 的样本里的每个 token 贡献 $\frac{1}{1000G}$
- 一个 100 token 的样本里的每个 token 贡献 $\frac{1}{100G}$

长样本里的 token 权重被严重稀释。长 CoT 场景下这意味着**关键的推理步骤（往往在长响应中）被低估**。

#### 解决：token 级平均

DAPO 改为：

$$
\mathcal{L}_{\text{DAPO}} = \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \ell_{i,t}
$$

所有 token **一视同仁**，长短样本对 loss 的贡献严格正比于其 token 数。

**效果**：
- 长响应中的低质量模式（重复、胡言乱语）能被充分惩罚；
- 长响应中的高质量推理能被充分奖励；
- 训练更稳定，熵和长度都不会异常爆炸。

### 3.5 Overlong Reward Shaping：超长奖励塑形

#### 问题：截断惩罚的噪声

训练时通常设定最大生成长度（如 16384），超过则截断。如果简单地把截断样本标记为"错误"（$r = -1$）：

- **问题 1**：一个推理正确但表达冗长的样本会被冤杀
- **问题 2**：引入奖励噪声，模型无法区分"方向错"和"说太多"

#### 解决：软超长惩罚（Soft Overlong Punishment）

设期望最大长度 $L_{\max}$（例如 16384）、缓冲长度 $L_{\text{cache}}$（例如 4096），则：

$$
r_{\text{length}}(|o|) = \begin{cases} 0, & |o| \leq L_{\max} - L_{\text{cache}} \\[4pt] \dfrac{(L_{\max} - L_{\text{cache}}) - |o|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |o| \leq L_{\max} \\[8pt] -1, & |o| > L_{\max} \end{cases}
$$

图示：

```
reward
   0 ┤━━━━━━━━━━━━━━━━━━━┓
     │                    ┃ 缓冲区（线性惩罚）
  -1 ┤                    ┗━━━━━━━━━━━━━
     └───────────────────┴──────────┴────── 长度
     0             L_max-L_cache  L_max
                  （12288）      （16384）
```

最终奖励 $= $ 答案正确性奖励 $+ r_{\text{length}}$。

**效果**：在缓冲区内给出平滑的长度压力，避免硬截断的奖励跳变；显著降低训练噪声，提升稳定性。

此外，论文还提到一种 **Overlong Filtering** 策略——对截断样本**直接 mask 掉 loss**。二者等效，默认实现用 Overlong Reward Shaping。

### 3.6 完整 DAPO 目标函数

把上述改动整合：

$$
\boxed{
\begin{aligned}
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{q,\ \{o_i\} \sim \pi_{\theta_{\text{old}}}} \Bigg[ & \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min\Big( r_{i,t}(\theta) \hat{A}_{i,t}, \\
& \text{clip}\big(r_{i,t}(\theta),\ 1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}}\big) \hat{A}_{i,t} \Big) \Bigg]
\end{aligned}
}
$$

配合 **动态采样** 的约束：

$$
0 < \big|\{i : \text{is\_equivalent}(a_i, \hat{a})\}\big| < G
$$

即强制每个组里既有对的也有错的。

**注意 DAPO 没有 KL 项**：对于长 CoT 推理任务，策略本就应当大幅偏离初始分布，KL 约束反而有害。

### 3.7 实验结果与工程细节

#### 主结果

| 方法 | AIME 2024 分数 | 训练步数 |
|------|----------------|----------|
| DeepSeek-R1-Zero-Qwen-32B | 47 | baseline |
| **DAPO-Qwen-32B** | **50** | **50% of baseline** |

#### 消融实验（从 Naive GRPO 逐步加入技术）

| 配置 | AIME 2024 |
|------|-----------|
| Naive GRPO baseline | 30 |
| + Clip-Higher | 38 |
| + Dynamic Sampling | 42 |
| + Token-Level Loss | 46 |
| + Overlong Reward Shaping | **50** |

每个组件都贡献了稳定的增益。

#### 关键超参

- 底模：Qwen2.5-32B base
- 优化器：AdamW，学习率 $1 \times 10^{-6}$，线性 warmup 20 步
- Rollout：prompt batch 512，每 prompt 采样 16 条
- Mini-batch：512（即每次 rollout 做 16 次梯度更新）
- 最大生成长度：16384 + 4096 缓冲 = 20480 硬上限
- Clip：$\varepsilon_{\text{low}} = 0.20,\ \varepsilon_{\text{high}} = 0.28$
- 硬件：128 张 H20
- 奖励：规则判定数学答案正确性（转换为整数便于匹配）
- 数据：DAPO-Math-17K（AoPS 爬取 + 人工标注）

---

## 四、GSPO：从 Token 级到序列级

### 4.1 论文信息与核心洞察

- **论文标题**：*Group Sequence Policy Optimization*
- **arXiv 编号**：2507.18071
- **发布时间**：2025 年 7 月 24 日
- **作者单位**：阿里 Qwen 团队（Chujie Zheng 等）
- **代号**：**GSPO = Group Sequence Policy Optimization**

**核心论断**（振聋发聩）：

> GRPO 的不稳定性**源于其重要性采样权重的根本性错用**。在 token 级应用重要性采样引入了高方差噪声，这些噪声会随响应长度累积，并被裁剪机制进一步放大，最终导致模型崩溃。

**核心创新**：**把重要性比从 token 级提升到序列级**。

GSPO 已成功应用于最新的 **Qwen3 系列模型**（Instruct / Coder / Thinking），是 MoE 模型 RL 训练的关键算法支撑。

### 4.2 GRPO 重要性采样的根本缺陷

#### 4.2.1 重要性采样的正确用法

重要性采样（IS）是一种用分布 $q$ 的样本估计分布 $p$ 下期望的技巧：

$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right] \approx \frac{1}{N} \sum_{i=1}^{N} \frac{p(x_i)}{q(x_i)} f(x_i)
$$

**关键前提**：需要**多个样本**来平均掉比值的随机性，否则单样本估计方差极高。

#### 4.2.2 GRPO 的错用

GRPO 的 token 级重要性比：

$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

在每个 $(q, o_{i,<t})$ 上下文下，**我们只有一个 token 样本 $o_{i,t}$**！这不满足"多样本平均"的前提，该 IS 权重几乎等同于**纯噪声**。

随着序列变长（长 CoT 场景下可能上万 token），这些噪声**逐 token 累积**，加上 PPO clip 机制的非线性放大，最终可能**不可逆地破坏模型**。

#### 4.2.3 在 MoE 模型上尤其致命

GSPO 团队发现，在 **Qwen3-30B-A3B-Base（48 层 MoE）** 上：

- 每次梯度更新后，同一 token 激活的专家**有约 10% 发生变化**
- 这让 token 级的 $\pi_\theta / \pi_{\theta_{\text{old}}}$ 剧烈波动
- GRPO 训练**无法稳定收敛**

此前 Qwen 团队尝试过 **Routing Replay**（缓存旧策略的专家路由，在新策略计算时重放）来绕过，但这种 hack 增加内存、复杂度且限制可扩展性。

### 4.3 GSPO 的序列级重要性比

GSPO 重新定义重要性比，**基于整个序列的似然**：

$$
\boxed{
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{\frac{1}{|y_i|}} = \exp\left(\frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}\right)
}
$$

**几个关键设计：**

1. **分子分母是整个序列的似然**：这与序列级奖励天然对齐（奖励就是对整句打的）
2. **$\frac{1}{|y_i|}$ 次方 = 长度归一化**：
   - 没有归一化时不同长度序列的 $s_i$ 数值范围差异巨大（短序列 $s_i$ 接近 1，长序列可能极小）
   - 归一化后所有序列的 $s_i$ 落在相近数值范围，裁剪阈值可以统一设置
3. **方差大幅降低**：序列级比值是 token 级 log-ratio 的平均，满足大数律

### 4.4 GSPO 目标函数

$$
\boxed{
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\ \{y_i\} \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{1}{G} \sum_{i=1}^{G} \min\!\left(s_i(\theta)\, \hat{A}_i,\ \text{clip}(s_i(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\, \hat{A}_i\right)\right]
}
$$

其中优势仍采用 GRPO 的组内归一化：

$$
\hat{A}_i = \frac{R(x, y_i) - \text{mean}(\{R(x, y_j)\}_{j=1}^G)}{\text{std}(\{R(x, y_j)\}_{j=1}^G)}
$$

**与 GRPO 的关键区别**：

| 维度 | GRPO | GSPO |
|------|------|------|
| 重要性比 | token 级 $r_{i,t}$ | 序列级 $s_i$ |
| 裁剪粒度 | token 级 | **序列级** |
| 优势粒度 | token 级（但值相同） | **序列级** |
| $\Sigma_t$ 对什么求和 | 逐 token 的 clip loss | 已无，外层直接是序列 |

**梯度理解**：GSPO 里，一个序列要么整体参与梯度更新（$s_i$ 在 $[1-\varepsilon, 1+\varepsilon]$ 内），要么整体被裁剪（不更新）。这与奖励本身的"序列级"性质**完全一致**。

### 4.5 为什么能解决 MoE 训练不稳定问题

GSPO 的序列级设计天然缓解 MoE 问题：

- 虽然 token 级路由会波动，但**序列似然是整个 token log-prob 的求和**，token 级波动在求和中**部分抵消**；
- 再加上 $\frac{1}{|y_i|}$ 次方的几何平均，**进一步平滑**单 token 的剧烈变化；
- 最终 $s_i$ 是一个稳定的、语义一致的量。

**因此可以完全丢弃 Routing Replay** —— 这是一个极大的工程简化。

### 4.6 GSPO-token 变体

论文还提出 GSPO-token 变体，允许在 token 级进行更细粒度的优势分配（例如来自过程奖励模型）：

$$
s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\text{sg}[\pi_\theta(y_{i,t} \mid x, y_{i,<t})]}
$$

其中 $\text{sg}[\cdot]$ 是 stop-gradient。

- **前向值**：$s_{i,t} = s_i$（序列级数值）
- **反向梯度**：只通过当前 token 的 log prob 回传
- **优点**：可以对每个 token 赋予不同的 $\hat{A}_{i,t}$，但数值上仍保持序列级稳定

这相当于"用序列级的比值尺度，做 token 级的梯度分配"。

### 4.7 实验现象：反直觉的裁剪比例

论文最引人注目的实验发现之一：

> **GSPO 裁剪掉的 token 比例比 GRPO 高两个数量级**，但训练效率反而更高！

这说明：
- GRPO 的 token 级梯度本身就充满噪声，"用了更多 token" 不等于"学得更多"；
- GSPO 虽然丢弃了大量被裁剪的 token，但**保留的信号质量更高**；
- 训练效率由**有效梯度质量**决定，而非"参与计算的 token 数"。

#### 实验结果概览

- 底模：Qwen3-30B-A3B-Base 的 cold-start 微调版
- 评测：AIME'24、LiveCodeBench、CodeForces
- 结果：
  - GSPO 收敛速度快于 GRPO
  - 同等计算量下 GSPO 准确率显著更高
  - 随训练 compute 增大，GSPO 继续提升，GRPO 趋于饱和

#### 基础设施友好性

论文额外提到一个工程优势：由于 GSPO 是序列级优化，**对推理与训练引擎的精度差异容忍度更高**。在大规模训练中，rollout 的推理引擎（vLLM/TensorRT）与训练引擎（Megatron）常有微小数值差异，token 级 IS 对此极其敏感，而序列级 IS 天然鲁棒，**可能去掉"重要性重计算"这类昂贵工程对齐**。

---

## 五、三种算法的对比与选型

### 5.1 目标函数并列对比

| 算法 | 核心公式 |
|------|----------|
| **GRPO** | $\dfrac{1}{G}\sum_i \dfrac{1}{\|o_i\|}\sum_t \min(r_{i,t}\hat{A}_i,\ \text{clip}(r_{i,t}, 1\pm\varepsilon)\hat{A}_i) - \beta \mathbb{D}_{\text{KL}}$ |
| **Dr.GRPO** | $\dfrac{1}{G}\sum_i \sum_t \min(r_{i,t}\hat{A}_i,\ \text{clip}(r_{i,t}, 1\pm\varepsilon)\hat{A}_i)$，$\hat{A}_i = R_i - \bar{R}$ |
| **DAPO** | $\dfrac{1}{\sum_i \|o_i\|} \sum_i \sum_t \min(\cdots,\ \text{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\cdots)$ + 动态采样 + 超长塑形 |
| **GSPO** | $\dfrac{1}{G}\sum_i \min(s_i \hat{A}_i,\ \text{clip}(s_i, 1\pm\varepsilon)\hat{A}_i)$，$s_i$ 为序列级长度归一化比 |

### 5.2 改动位置对比

| 改动维度 | Dr.GRPO | DAPO | GSPO |
|----------|:-------:|:----:|:----:|
| 去掉长度归一化 | ✅ | ✅ | — |
| 去掉 std 归一化 | ✅ | — | — |
| 解耦 clip 上下界 | — | ✅ | — |
| 动态采样过滤 | — | ✅ | — |
| 超长奖励塑形 | — | ✅ | — |
| 去掉 KL 惩罚 | ✅ ($\beta=0$) | ✅ | — |
| **改 IS 粒度** | — | — | **✅（根本性）** |

### 5.3 适用场景建议

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 学术研究、消融实验 | **Dr.GRPO** | 公式简洁、修正明确、便于分析 |
| 工业级稠密模型训练 | **DAPO** | 工程成熟、四件套 SOTA、开源完整 |
| MoE 模型训练 | **GSPO** | 唯一能稳定训练大 MoE 的方案 |
| 小模型快速实验 | **Dr.GRPO** | 27 小时就能出结果 |
| 长 CoT 推理（32B+） | **DAPO 或 GSPO** | 长度相关的偏差显著，必须修正 |
| 混合专家 + 长 CoT | **GSPO** | 两大痛点一起解决 |

### 5.4 可以组合吗？

**可以，而且推荐组合！** 这三项工作并非互斥：

- **Dr.GRPO 的 token 级聚合** ≈ **DAPO 的 Token-Level Loss**（二者目标相同）
- **DAPO 的 Clip-Higher** 可与 **GSPO 的序列级 clip** 结合，解耦序列级上下界
- **DAPO 的动态采样** 对 **Dr.GRPO、GSPO** 同样有效

实际生产里，常见的组合是：

```
GSPO 的序列级 IS   +   DAPO 的 Clip-Higher
                  +   DAPO 的动态采样
                  +   DAPO 的超长塑形
                  +   β = 0（无 KL）
```

这几乎是目前 LLM RL 的"最佳实践配方"。

---

## 六、统一视角：GRPO 的演化谱系

```
                           ┌─────────────┐
                           │    GRPO     │  2024.02  DeepSeek
                           │  (原版)      │
                           └──────┬──────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
   ┌─────────────┐         ┌─────────────┐        ┌─────────────┐
   │  Dr.GRPO    │         │    DAPO     │        │    GSPO     │
   │ (2025.03)   │         │ (2025.03)   │        │ (2025.07)   │
   │             │         │             │        │             │
   │ 去有偏归一化 │         │ 工程四件套   │        │ 序列级 IS    │
   │ (Done Right)│         │ (大规模稳定) │        │ (MoE 友好)  │
   └─────────────┘         └─────────────┘        └─────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  LLM RL 最佳实践   │
                        │  (组合式配方)     │
                        └──────────────────┘
```

**三篇论文分别回答了三个不同层次的问题：**

1. **Dr.GRPO**：GRPO 的数学公式对吗？→ **答：有偏差，需要修正**
2. **DAPO**：GRPO 在大规模工程里怎么用？→ **答：需要一系列补丁**
3. **GSPO**：GRPO 的理论基础对吗？→ **答：IS 的粒度根本就错了**

这三个问题由浅入深，恰好代表了算法改进的三个层次：**数学层 → 工程层 → 理论层**。

---

## 七、参考文献

### 核心论文

1. **Dr.GRPO**：Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., & Lin, M. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective*. arXiv:2503.20783. COLM 2025.
   - 代码：[github.com/sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)

2. **DAPO**：ByteDance Seed & Tsinghua AIR. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476.
   - 主页：[dapo-sia.github.io](https://dapo-sia.github.io/)
   - 代码：[github.com/BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)

3. **GSPO**：Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., & Lin, J. (2025). *Group Sequence Policy Optimization*. arXiv:2507.18071.
   - 博客：[qwenlm.github.io/blog/gspo](https://qwenlm.github.io/blog/gspo/)

### 相关基础

4. **GRPO**：Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
5. **DeepSeek-R1**：DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
6. **PPO**：Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

### 相关后续工作

7. **GTPO / GRPO-S**：基于策略熵的 token/序列级奖励塑形（arXiv:2508.04349）
8. **P-GSPO**：参数化 GSPO，在长度敏感性和稳定性之间权衡（2025.10）
9. **verl 框架**：字节开源的 LLM RL 基础设施，内置 DAPO 实现
10. **NeMo-RL**：NVIDIA 的 RL 框架，也支持 DAPO

---

## 附录：一页纸速查

### Dr.GRPO（3 秒理解）

> 把 GRPO 的 $\dfrac{1}{\|o_i\|}$ 和 $\dfrac{1}{\sigma}$ 都去掉。

### DAPO（四件套口诀）

> **"解耦裁剪、动态采样、token 级 loss、超长软罚"**

超参：$\varepsilon_{\text{low}}=0.20,\ \varepsilon_{\text{high}}=0.28$；batch 512、每 prompt 采 16 条；max len 16384 + 4096 缓冲；无 KL。

### GSPO（核心公式）

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{\frac{1}{|y_i|}}
$$

"**序列级 IS，几何平均，整句裁剪**"。MoE 训练必用。