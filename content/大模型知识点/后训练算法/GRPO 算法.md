---
created: '2026-04-22 14:11:34.081734+00:00'
order: 2
pinned: false
summary: deepseek 宇宙
title: GRPO 算法
updated: '2026-04-30 22:20:02+08:00'
---

> **Group Relative Policy Optimization（群组相对策略优化）**
> 由 DeepSeek 团队在 DeepSeekMath（2024）论文中提出，并在 DeepSeek-R1 中被广泛使用，成为当前大模型强化学习训练的主流算法之一。

![GRPO 训练流程](/static/images/uploads/后训练算法/grpo-training-flow.png)

图中展示了 GRPO 一个训练 step 的完整数据流：同一 prompt 先由旧策略采样出 $G$ 条回答，再用奖励模型或规则验证器打分，组内计算均值 $\mu$ 和标准差 $\sigma$ 得到相对优势 $A_i$，随后把样本级优势广播到 token 级，结合 clip loss 与参考模型 KL 项更新当前策略。左下角被划掉的 critic 是 GRPO 相比 PPO 最核心的工程收益。

---

## 1. 背景与动机

### 1.1 RLHF 的基本范式

大语言模型（LLM）的训练通常分为三个阶段：

1. **预训练（Pretraining）**：在海量无监督文本上学习语言规律
2. **监督微调（SFT）**：在高质量指令数据上对齐基础行为
3. **强化学习微调（RLHF / RLAIF）**：根据奖励信号进一步优化策略

在第三阶段，最经典的算法是 **PPO（Proximal Policy Optimization）**。但 PPO 在 LLM 场景下存在若干痛点，GRPO 正是针对这些痛点而生。

### 1.2 PPO 在 LLM 上的主要痛点

- **需要额外的 Value Model（价值网络）**：通常与策略模型规模相当，内存开销翻倍；
- **价值估计困难**：LLM 的奖励通常只在序列末尾给出（outcome reward），训练 token 级别的价值函数既困难又不准；
- **训练不稳定**：价值网络的偏差会传导到 Advantage 估计中，引入噪声；
- **实现复杂**：需要维护 actor、critic、reward model、reference model 四个模型。

### 1.3 GRPO 的核心诉求

> **去掉 Value Model，用"群组内相对比较"替代价值估计。**

这正是 GRPO 的核心贡献：通过对同一 prompt 采样多个输出，利用群组内部的奖励均值和标准差来估计优势函数，从而完全省去 critic 网络。

---

## 2. 前置知识：PPO 回顾

为了理解 GRPO 的改动，必须先梳理 PPO 的关键公式。

### 2.1 PPO 的目标函数

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{q, o \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t \right) \right]
$$

其中：

- $ r_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{\lt t})}{\pi_{\theta_{old}}(o_t \mid q, o_{\lt t})} $：新旧策略的概率比
- $\hat{A}_t$：时刻 $t$ 的优势估计，通常由 GAE 计算
- $\varepsilon$：裁剪阈值，常取 0.1 或 0.2

### 2.2 GAE 与 Value Model

PPO 使用 GAE（Generalized Advantage Estimation）计算优势：

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中 $V(s)$ 由价值网络给出。**这正是 PPO 需要额外模型的根源。**

### 2.3 KL 惩罚的位置

在 PPO 做 RLHF 时，KL 散度通常被加到奖励中（reward shaping）：

$$
r_t = r_\phi(q, o) - \beta \log \frac{\pi_\theta(o_t \mid q, o_{\lt t})}{\pi_{\text{ref}}(o_t \mid q, o_{\lt t})}
$$

这会让价值函数被迫建模 KL 项，进一步增大了 critic 的学习难度。

---

## 3. GRPO 核心思想

GRPO 的关键创新可以用三句话概括：

1. **对同一问题采样一组输出**，而非单条轨迹；
2. **用组内奖励的均值与标准差归一化**，得到相对优势，彻底去掉 critic；
3. **把 KL 散度直接放到 loss 里**，而不是奖励里，使用无偏估计量。

图示对比（概念）：

```
PPO：  prompt ──► 1 个输出 ──► reward ──► Advantage（需 Value）
                                   │
                                   ▼
                              Critic 网络（大！）

GRPO： prompt ──► G 个输出 ──► G 个 reward ──► 组内归一化 ──► Advantage
                                                                       （无需 Critic）
```

![PPO、GRPO 与 GRPO 改进对比](/static/images/uploads/后训练算法/grpo-ppo-variants-comparison.png)

这张对比图把 PPO、原版 GRPO 和后续 GRPO 改进放在同一坐标里：PPO 的优势估计依赖 critic，显存压力更高；GRPO 用同一 prompt 的组内奖励替代 critic，但采样成本上升；Dr.GRPO、DAPO、GSPO 等改进继续处理长度偏差、无效组过滤和序列级比值等问题，因此更适合长 CoT 与大规模推理 RL。

---

## 4. 数学公式详解

### 4.1 完整目标函数

$$
\boxed{
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O \mid q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( r_{i,t}(\theta)\, \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\, \hat{A}_{i,t} \right) - \beta\, \mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\Vert\, \pi_{\text{ref}}\right] \right\} \right]
}
$$

### 4.2 各符号含义

| 符号 | 含义 |
|------|------|
| $q$ | 输入 prompt（来自分布 $P(Q)$） |
| $G$ | 每个 prompt 采样的输出数量（组大小，常取 8、16、32、64） |
| $o_i$ | 第 $i$ 条采样输出（token 序列） |
| $o_{i,t}$ | 输出 $o_i$ 的第 $t$ 个 token |
| $\pi_\theta$ | 当前待更新的策略 |
| $\pi_{\theta_{\text{old}}}$ | 采样用的旧策略（一个 mini-batch 内冻结） |
| $\pi_{\text{ref}}$ | 参考策略（通常是 SFT 模型） |
| $r_{i,t}(\theta)$ | 概率比 $\pi_\theta(o_{i,t} \mid q, o_{i,\lt t}) / \pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})$ |
| $\hat{A}_{i,t}$ | token 级别优势（详见第 5 节） |
| $\varepsilon$ | 裁剪阈值 |
| $\beta$ | KL 惩罚系数 |

### 4.3 关键区别再强调

- **期望外层对"群组"求平均**：$\frac{1}{G}\sum_{i=1}^G$
- **KL 项在目标函数内**，不在奖励内
- **没有 $V(s)$**，也没有 GAE

---

## 5. 优势函数的计算方式

GRPO 支持两种监督粒度：**结果监督（Outcome Supervision）** 与 **过程监督（Process Supervision）**。

### 5.1 结果监督（Outcome Supervision）

最常见、最简洁的设定：奖励只在序列末尾给出（如答案是否正确）。

**步骤：**

1. 对 prompt $q$ 采样 $G$ 条输出 $\{o_1, o_2, \dots, o_G\}$；
2. 对每条输出打分得到 $\{r_1, r_2, \dots, r_G\}$；
3. 计算组内统计量：
   $$
   \mu = \frac{1}{G}\sum_{i=1}^G r_i, \quad \sigma = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu)^2}
   $$
4. 归一化得到**样本级优势**：
   $$
   \tilde{A}_i = \frac{r_i - \mu}{\sigma}
   $$
5. 把样本优势广播到该样本的每个 token：
   $$
   \hat{A}_{i,t} = \tilde{A}_i, \quad \forall t = 1, \dots, |o_i|
   $$

**直觉**：同一个问题有 $G$ 个答案，比组内平均好的答案获得正向梯度、差的获得负向梯度，形成"兄弟之间互相对比"的学习信号。

### 5.2 过程监督（Process Supervision）

当奖励模型可以对推理过程中的关键步骤打分时（例如 PRM，Process Reward Model）：

1. 对输出 $o_i$ 的每一步 $k$ 得到奖励 $r_i^{(k)}$；
2. 将所有样本所有步骤的奖励一起归一化：
   $$
   \tilde{r}_i^{(k)} = \frac{r_i^{(k)} - \mu}{\sigma}
   $$
3. token $t$ 的优势等于**其后所有步骤归一化奖励之和**：
   $$
   \hat{A}_{i,t} = \sum_{\text{step } k\ \text{s.t.\ step ends at or after }t} \tilde{r}_i^{(k)}
   $$

**直觉**：让模型能感知到"从哪一步开始走偏了"，而不只是终点好坏。

### 5.3 为什么不需要 Critic？

因为 $\mu$（组均值）本身就在扮演"baseline"的角色。在策略梯度理论中，任何只依赖状态的 baseline 都不会引入偏差，只会降低方差。GRPO 用"同一 prompt 下其他样本的平均奖励"作为 baseline，**既无偏又方差低**，天然满足 baseline 的要求。

---

## 6. KL 散度的处理

### 6.1 放到 Loss 里而不是 Reward 里

PPO：$r_t = r_\phi - \beta \log\frac{\pi_\theta}{\pi_{\text{ref}}}$（KL 混入奖励）

GRPO：直接在目标函数末尾减去 $\beta \mathbb{D}_{\text{KL}}[\pi_\theta \Vert \pi_{\text{ref}}]$

**好处：**

- 不会污染 Advantage 估计
- 使 KL 约束更显式、可调
- 数值稳定性更好

### 6.2 无偏低方差的 KL 估计量

GRPO 采用 John Schulman 在博客中提出的 **k3 估计量**：

$$
\mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\Vert\, \pi_{\text{ref}}\right]_{i,t} = \frac{\pi_{\text{ref}}(o_{i,t}\mid q,o_{i,\lt t})}{\pi_\theta(o_{i,t}\mid q,o_{i,\lt t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}\mid q,o_{i,\lt t})}{\pi_\theta(o_{i,t}\mid q,o_{i,\lt t})} - 1
$$

令 $u = \dfrac{\pi_{\text{ref}}}{\pi_\theta}$，则估计量为：

$$
\hat{D}_{\text{KL}} = u - \log u - 1
$$

**性质：**

- $\hat{D}_{\text{KL}} \geq 0$ 恒成立（因为 $x - \log x - 1 \geq 0$ 对 $x > 0$ 总成立）
- 无偏：$\mathbb{E}_{\pi_\theta}[\hat{D}_{\text{KL}}] = D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$
- 方差远小于 $-\log u$ 这种朴素估计

---

## 7. 完整算法流程

### 7.1 训练一个 step

```
输入：当前策略 π_θ，参考策略 π_ref，奖励模型 R，prompt batch B

1. 冻结 π_θ_old ← π_θ（本 step 采样使用）

2. for 每个 prompt q ∈ B:
     a. 从 π_θ_old 采样 G 条输出 {o_1, ..., o_G}
     b. 用奖励模型打分 {r_1, ..., r_G}
     c. 计算组均值 μ 和标准差 σ
     d. 对每个 o_i：A_i = (r_i - μ) / σ
     e. 对 o_i 中每个 token t：Â_{i,t} = A_i（结果监督）

3. 用 {(o_i, Â_{i,t})} 组成训练数据

4. for k 次 PPO-style 内部迭代:
     a. 计算概率比 r_{i,t}(θ) = π_θ / π_θ_old
     b. 计算 clip loss
     c. 计算 KL 惩罚（对 π_ref 的 k3 估计量）
     d. 梯度上升更新 θ

5. （可选）每隔若干 step 更新 π_ref ← π_θ（例如 DeepSeekMath 中每个 iteration 更新）
```

### 7.2 数据流示意图

```
      ┌────────────────┐
      │ prompt batch   │
      └───────┬────────┘
              │
              ▼
   ┌──────────────────────┐
   │ π_θ_old 采样 G 条输出 │ ── 对每个 prompt
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ 奖励模型 R 打分       │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ 组内归一化 → Â       │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ GRPO 目标 + KL 惩罚   │
   └──────────┬───────────┘
              │
              ▼
         更新 π_θ
```

---

## 8. GRPO 与 PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 是否需要 Value Model | ✅ 需要 | ❌ 不需要 |
| 显存占用 | 高（4 个模型） | 中（3 个模型） |
| 每个 prompt 的采样数 | 1（或少量） | G（通常 8~64） |
| Advantage 估计 | GAE（依赖 Value） | 组内奖励归一化 |
| KL 惩罚位置 | 奖励中 | Loss 中 |
| KL 估计量 | $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | $u - \log u - 1$（k3） |
| 适用场景 | 通用 RL / RLHF | 具有可验证奖励的任务（数学、代码、推理） |
| 训练稳定性 | 中等，调参复杂 | 相对稳定 |
| 实现复杂度 | 较高 | 较低 |

### 显存占用直观感受

以 70B 模型为例：

- **PPO**：Actor（70B）+ Ref（70B）+ Critic（70B）+ RM（70B）= 280B 参数的显存占用
- **GRPO**：Actor + Ref + RM = 210B 参数，省了 1/4 显存

但 GRPO 采样数 $G$ 较大，**计算量（推理 FLOPs）反而可能更高**。这是一个用"多采样"换取"去 critic"的 trade-off。

---

## 9. 实现细节与伪代码

### 9.1 PyTorch 风格伪代码（核心部分）

```python
def grpo_step(prompts, policy, ref_policy, reward_model,
              G=8, epsilon=0.2, beta=0.04, inner_iters=1):
    # 1. 采样 G 条输出（使用旧策略）
    old_policy = deepcopy(policy).eval()
    all_outputs, all_old_logprobs = [], []
    for q in prompts:
        outputs, old_lp = old_policy.generate(q, num_return_sequences=G)
        all_outputs.append(outputs)       # [G, seq_len]
        all_old_logprobs.append(old_lp)   # [G, seq_len]

    # 2. 奖励打分 + 组内归一化
    advantages = []
    for outputs in all_outputs:
        rewards = reward_model.score(outputs)         # [G]
        mu, sigma = rewards.mean(), rewards.std() + 1e-8
        A = (rewards - mu) / sigma                    # [G]
        # 广播到 token 级
        A_token = A.unsqueeze(1).expand_as(outputs)   # [G, seq_len]
        advantages.append(A_token)

    # 3. 多次内部优化
    for _ in range(inner_iters):
        for q, outputs, old_lp, A in zip(prompts, all_outputs,
                                         all_old_logprobs, advantages):
            new_lp = policy.log_prob(q, outputs)      # [G, seq_len]
            ref_lp = ref_policy.log_prob(q, outputs)  # [G, seq_len]

            # 概率比
            ratio = torch.exp(new_lp - old_lp)        # [G, seq_len]

            # PPO clip
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A
            policy_loss = -torch.min(surr1, surr2)

            # KL 的 k3 无偏估计
            log_u = ref_lp - new_lp
            u = torch.exp(log_u)
            kl = u - log_u - 1                        # [G, seq_len]

            # 注意：GRPO 的样本平均是 "先样本内求均值再样本间求均值"
            loss_per_sample = (policy_loss + beta * kl).mean(dim=-1)  # 对 token 求均值
            loss = loss_per_sample.mean()                             # 对样本求均值

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 9.2 几个易踩的坑

1. **归一化时 $\sigma$ 为 0 的情况**：一组全部回答正确或全部错误时 $\sigma = 0$，要加 $\varepsilon$ 防止除零；这组样本的梯度信号几乎为 0，也可以直接跳过该 prompt。

2. **Token 级 vs 序列级归一化**：
   - DeepSeekMath 原版：先对 token 求均值，再对样本求均值 → 短序列权重更大
   - DAPO、Dr.GRPO 等变体：改为直接对所有 token 求均值 → 消除长度偏差

3. **采样温度**：为了组内多样性，采样 temperature 一般设为 0.7~1.0，否则 G 个样本高度相似，$\sigma$ 很小，梯度爆炸。

4. **旧策略缓存**：采样时必须保存 $\log \pi_{\theta_{\text{old}}}$，否则多次内部迭代无法重用数据。

5. **KL 系数 $\beta$**：常见取值 0.001 ~ 0.05。过大会让模型不敢更新；DeepSeek-R1 甚至直接设 $\beta = 0$（认为 reference model 约束不必要）。

---

## 10. 典型应用：DeepSeek-R1

### 10.1 DeepSeekMath（首次提出）

- 任务：数学推理（GSM8K、MATH）
- 奖励：规则判定答案是否正确（binary reward）
- 效果：7B 模型在 MATH 上超越了当时绝大多数开源模型

### 10.2 DeepSeek-R1-Zero（完全 RL，不用 SFT）

- 直接从 base 模型用 GRPO 做 RL
- 奖励组成：
  - **准确率奖励**：答案正确性（规则判定 / 代码执行）
  - **格式奖励**：思维链写在 `<think></think>` 之间
- 神奇现象：**模型自发地涌现出长链推理、自我反思（"Aha moment"）**
- 证明了大规模 RL 可以独立于 SFT 产生高质量推理行为

### 10.3 DeepSeek-R1（SFT + RL 混合）

- 先用少量 cold-start SFT 数据初始化
- 再用 GRPO 做多阶段 RL
- 最终性能达到 o1 级别

---

## 11. 优缺点分析

### 11.1 优点

✅ **内存友好**：省去 critic，大模型场景下显存省一大截
✅ **实现简单**：去掉 GAE、value 训练等逻辑，代码量减少
✅ **训练稳定**：组内归一化是自然的 baseline，方差低
✅ **适合可验证奖励**：数学、代码等任务的规则奖励信号天然适合
✅ **与任务结构契合**：LLM 天然是序列生成，采样 G 条本就常见（如 best-of-N）

### 11.2 缺点 / 局限

❌ **采样成本高**：每个 prompt 要生成 G 条，推理 FLOPs 是 PPO 的 G 倍
❌ **长度偏差**：原版按 token 平均会让短输出占优势（变体已修正）
❌ **奖励质量依赖高**：组内相对比较要求奖励有区分度，同分情况下信号为 0
❌ **不适合稀疏/连续奖励**：对通用 RL 场景（如游戏）不如 PPO 成熟
❌ **对 G 值敏感**：G 太小方差大；G 太大成本高，需要调参
❌ **难以处理多轮交互**：更适合单轮生成任务

---

## 12. 常见变体与改进

以下是 2024~2025 年基于 GRPO 的重要改进：

### 12.1 DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）

- **Clip-Higher**：上下裁剪阈值解耦（$\varepsilon_{\text{low}} < \varepsilon_{\text{high}}$），鼓励低概率 token 的探索
- **动态采样**：过滤掉全对或全错的 prompt，避免无效梯度
- **Token-level loss**：对所有 token 一起平均，消除长度偏差
- **Overlong shaping**：对超长截断输出做平滑惩罚

### 12.2 Dr.GRPO（Done Right）

- 发现原 GRPO 会系统性偏好**更长的错误答案**
- 去掉对 $|o_i|$ 的除法以及对 $\sigma$ 的除法
- 公式从 $\frac{1}{G}\sum\frac{1}{|o_i|}\sum$ 改为 $\frac{1}{\sum|o_i|}\sum\sum$

### 12.3 RLOO（REINFORCE Leave-One-Out）

- 比 GRPO 更早提出的类似思想
- baseline 用"除自己外其他样本的平均奖励"，更严格的无偏估计
- 在 LLM 场景下性能与 GRPO 相近甚至更好

### 12.4 GSPO（Group Sequence Policy Optimization，Qwen 团队）

- 将 token 级重要性比改为序列级
- 进一步降低方差，提升大模型训练稳定性

### 12.5 无 KL 版本

许多实际落地工作直接设 $\beta = 0$，完全移除 KL 约束，依赖 clip 来限制更新幅度。经验表明在数学/代码任务上效果反而更好。

---

## 13. 常见问题 FAQ

**Q1：GRPO 一定需要奖励模型吗？**
不一定。对于数学/代码这类任务，可以用"答案是否正确"、"代码能否通过测试"等**规则奖励**，比训练奖励模型更稳定、也更便宜。

**Q2：G 多大合适？**
经验上 $G \in [8, 64]$。DeepSeekMath 用 64，DeepSeek-R1 也用较大值。小模型可用 8~16 以节省成本。

**Q3：为什么 GRPO 在推理任务上效果特别好？**
因为推理任务（数学、代码）具备三个关键特性：
1. 答案**可自动验证**（奖励准确）
2. 同一问题的不同解法**奖励差异大**（组内对比信号强）
3. 长链推理受益于**探索-利用** trade-off

**Q4：GRPO 能用在通用对话 / 偏好对齐上吗？**
可以，但需要奖励模型提供有区分度的评分。如果奖励模型对组内 $G$ 条回答打分差异不大，$\sigma$ 很小会让梯度信号失真。

**Q5：KL 惩罚到底要不要？**
取决于任务：
- 需要保持与 SFT 分布接近（如对话） → 要
- 只关心最终答案正确性（如数学） → 可以去掉
- DeepSeek-R1-Zero 直接不要 reference model

**Q6：GRPO 和 REINFORCE 是什么关系？**
GRPO 本质上是 **REINFORCE + 组内 baseline + PPO 式 clip + KL 约束**。可以看作 REINFORCE 的"工业级强化版"。

---

## 14. 参考文献

1. **DeepSeekMath**: Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300
2. **DeepSeek-R1**: DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948
3. **PPO**: Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
4. **GAE**: Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438
5. **KL 估计量**: Schulman, J. (2020). *Approximating KL Divergence*. Blog: [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)
6. **DAPO**: ByteDance Seed. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476
7. **Dr.GRPO**: Liu, Z., et al. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective*. arXiv:2503.20783
8. **RLOO**: Ahmadian, A., et al. (2024). *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*. arXiv:2402.14740

---

## 附录 A：一页纸速查

```
【GRPO 核心公式】

J(θ) = E_{q, {o_i}~π_old} [
  (1/G) Σ_i (1/|o_i|) Σ_t {
    min(r_{i,t} Â_{i,t},  clip(r_{i,t}, 1-ε, 1+ε) Â_{i,t})
    - β · KL[π_θ ‖ π_ref]
  }
]

其中:
  r_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})
  Â_{i,t} = (R_i - mean(R)) / std(R)          (outcome supervision)
  KL = u - log(u) - 1,   u = π_ref / π_θ       (k3 estimator)

【关键 hyperparameters】
  G      : 8 ~ 64
  ε      : 0.1 ~ 0.2   (可解耦为 ε_low, ε_high)
  β      : 0 ~ 0.05
  温度   : 0.7 ~ 1.0

【训练要点】
  1. 冻结 π_old 采样
  2. 用 reward model 或规则打分
  3. 组内归一化得 Â
  4. clip + KL loss 更新 π_θ
  5. 每若干 step 可更新 π_ref
```

---

*笔记完成于 2026 年 4 月。如有错漏欢迎指正。*
