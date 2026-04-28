---
created: '2026-04-22 15:42:12.039450+00:00'
order: 2
pinned: false
summary: GRPO 代码
title: GRPO 🍣
updated: '2026-04-22 16:02:26.984739+00:00'
---

GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 使用的核心 RL 算法，相比 PPO **去掉了 Critic 网络**，改用**组内相对奖励**作为 baseline。

---

### 核心思路（面试先说清楚）

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{old}} \hat{A}_i,\ \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_i\right)\right] + \beta \cdot \mathbb{KL}(\pi_\theta \| \pi_{ref})$$

**Advantage 计算（组内归一化）：**

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1,...,r_G\})}{\text{std}(\{r_1,...,r_G\})}$$

---

### 完整实现

```python
def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> torch.Tensor:
  """
  计算广义策略优化 (GRPO) 损失。GRPO 结合了 PPO 的裁剪机制和 KL 散度惩罚。

  Args:
    log_probs: 当前策略的对数概率。形状为 (batch_size, sequence_length)。
    old_log_probs: 旧策略（行为策略）的对数概率。形状与 log_probs 相同。
    ref_log_probs: 参考策略的对数概率，用于 KL 散度惩罚。形状与 log_probs 相同。
    rewards: 从环境中获得的奖励。形状为 (batch_size, sequence_length)。
    attention_mask: 注意力掩码，用于忽略填充部分。形状与 log_probs 相同。
    group_size: 组大小，此处未直接使用，但可能用于其他上下文。
    clip_eps: PPO 裁剪参数，用于限制策略更新的幅度。
    kl_coef: KL 散度项的系数，用于控制正则化强度。

  Returns:
    计算出的 GRPO 损失。
  """

  # 1. 优势函数 (Advantage) 计算
  # 对奖励进行归一化处理，使其均值为0，标准差为1。这有助于稳定训练。
  mean_r = rewards.mean(dim=-1, keepdim=True) # 计算每行的平均奖励
  std_r = rewards.std(dim=-1, keepdim = True).clamp(min=1e-8) # 计算每行的标准差，并防止除零
  advantages = (rewards - mean_r) / std_r # 计算标准化后的优势

  # 将优势张量展平为一维，以便后续计算。
  advantages = advantages.view(-1)

  # 2. 序列对数概率 (Sequence Log Probabilities) 计算
  # 将对数概率与注意力掩码相乘，然后沿序列维度求和，得到每个序列的总对数概率。
  # 这样可以忽略填充部分对对数概率的贡献。
  seq_log_prob = (log_probs * attention_mask).sum(dim = -1) # 当前策略的序列对数概率
  seq_old_log_prob = (old_log_probs * attention_mask).sum(dim = -1) # 旧策略的序列对数概率
  seq_ref_log_prob = (ref_log_probs * attention_mask).sum(dim = -1) # 参考策略的序列对数概率

  # 3. 策略梯度 (Policy Gradient) 损失计算 (PPO 裁剪)
  # 计算当前策略与旧策略之间的对数概率比率。
  log_ratio = seq_log_prob - seq_old_log_prob
  # 将对数比率转换为实际的比率 exp(log_ratio)。
  ratio = log_ratio.exp()

  # PPO 损失项 1: 正常的策略梯度损失。
  pg_loss1 = ratio * advantages
  # PPO 损失项 2: 裁剪后的策略梯度损失。
  # ratio.clamp(1 - clip_eps, 1 + clip_eps) 将比率限制在 [1 - clip_eps, 1 + clip_eps] 之间，
  # 防止策略更新过大，提高训练稳定性。
  pg_loss2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
  # 最终的策略梯度损失取 pg_loss1 和 pg_loss2 中的最小值，然后取负平均值。
  # 因为我们通常希望最大化奖励，所以优化器会尝试最小化负的策略梯度损失。
  pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

  # 4. KL 散度 (KL Divergence) 损失计算
  # 计算当前策略与参考策略之间的 KL 散度。
  # 鼓励新策略不要偏离参考策略太远，作为一种正则化手段。
  kl = (seq_log_prob - seq_ref_log_prob).mean()
  # KL 损失乘以一个系数 kl_coef，以控制其对总损失的贡献。
  kl_loss = kl_coef * kl

  # 5. 总损失 (Total Loss)
  # 总损失是策略梯度损失和 KL 散度损失的和。
  loss = pg_loss + kl_loss

  return loss
```

---

### 与 PPO 的核心区别（面试必答）

| | PPO | GRPO |
|---|---|---|
| Baseline | Critic 网络（V 函数） | 组内奖励均值 |
| 显存开销 | 需额外 Critic（≈ 同等大小）| 无 Critic，**省约一半显存** |
| Advantage | GAE 估计 | $(r_i - \mu_G) / \sigma_G$ |
| 适用场景 | 通用 RL | LLM RLHF / 推理奖励 |

---

### 面试加分点

1. **为什么能去掉 Critic？** —— LLM 生成是 episode 级任务，每条输出有完整奖励，不需要对中间状态做 value estimation；组内归一化本身就是一个低方差的 baseline。

2. **group size G 怎么选？** —— 太小方差大，太大采样成本高，实践中 DeepSeek 用 G=8~16。

3. **KL 惩罚的作用？** —— 防止策略偏离 SFT 参考模型太远，避免 reward hacking 和模型退化。