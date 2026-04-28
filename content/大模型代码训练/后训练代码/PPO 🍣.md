---
created: '2026-04-22 16:33:25.837583+00:00'
order: 0
pinned: false
summary: 手撕 PPO
title: PPO 🍣
updated: '2026-04-22 16:34:49.092840+00:00'
---

### 核心公式先背熟

**GAE（Generalized Advantage Estimation）：**

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**PPO-Clip Loss：**

$$\mathcal{L}^{CLIP} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

$$\mathcal{L}^{total} = \mathcal{L}^{CLIP} + c_1 \mathcal{L}^{VF} - c_2 \mathcal{H}[\pi_\theta]$$

---

### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


# ══════════════════════════════════════════════════════════════
# 1. GAE 计算
# ══════════════════════════════════════════════════════════════

def compute_gae(
    rewards: torch.Tensor,      # [T]      每步奖励
    values: torch.Tensor,       # [T+1]    V(s_0)...V(s_T)，最后一个是 bootstrap
    dones: torch.Tensor,        # [T]      episode 终止标志
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
        advantages : [T]  GAE 优势估计
        returns    : [T]  TD(λ) returns，用于训练 Critic
    """
    T = len(rewards)
    advantages = torch.zeros(T)

    gae = 0.0
    for t in reversed(range(T)):
        # done=1 时下一状态价值清零（episode 结束）
        next_value = values[t + 1] * (1.0 - dones[t])

        # TD 残差 δ_t
        delta = rewards[t] + gamma * next_value - values[t]

        # 递推：A_t = δ_t + γλ * (1-done) * A_{t+1}
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    # returns = advantages + values（用于 Critic 回归目标）
    returns = advantages + values[:-1]
    return advantages, returns


# ══════════════════════════════════════════════════════════════
# 2. PPO Loss
# ══════════════════════════════════════════════════════════════

def compute_ppo_loss(
    new_log_probs: torch.Tensor,   # [B]  当前策略下 log π(a|s)
    old_log_probs: torch.Tensor,   # [B]  采样时旧策略 log π_old(a|s)
    advantages: torch.Tensor,      # [B]  GAE 优势（已归一化）
    returns: torch.Tensor,         # [B]  TD(λ) 目标
    values: torch.Tensor,          # [B]  当前 Critic 输出 V(s)
    entropy: torch.Tensor,         # [B]  策略熵 H[π(·|s)]
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    value_clip_eps: float = 0.2,   # Critic 也做 clip（OpenAI 实现）
    old_values: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:

    # ── 2.1 Policy Loss（Clipped Surrogate）────────────────────
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()                          # ρ_t = π/π_old

    # 数值稳定性检查（面试可提）
    # approx_kl = ((ratio - 1) - log_ratio).mean()

    pg_loss1 = ratio * advantages
    pg_loss2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

    # ── 2.2 Value Loss（MSE + 可选 Clip）───────────────────────
    if old_values is not None:
        # OpenAI 版本：对 value 也做 clip，防止更新幅度过大
        values_clipped = old_values + (values - old_values).clamp(
            -value_clip_eps, value_clip_eps
        )
        vf_loss1 = (values - returns).pow(2)
        vf_loss2 = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
    else:
        value_loss = 0.5 * (values - returns).pow(2).mean()

    # ── 2.3 Entropy Bonus（鼓励探索）────────────────────────────
    entropy_loss = -entropy.mean()

    # ── 2.4 总 Loss ──────────────────────────────────────────────
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":     -entropy_loss.item(),
        "approx_kl":   ((ratio - 1) - log_ratio).mean().item(),
        "clip_frac":   ((ratio - 1).abs() > clip_eps).float().mean().item(),
    }
    return total_loss, metrics
```

---

### 流程图总结

```
┌─────────────────────────────────────────────────────┐
│                   PPO 训练循环                        │
│                                                      │
│  ① Rollout 收集                                      │
│     env → sample action → store (s,a,r,done,logp,V) │
│                    ↓                                 │
│  ② GAE 计算                                          │
│     δ_t = r_t + γV(s_{t+1}) - V(s_t)               │
│     A_t = Σ (γλ)^l δ_{t+l}   （逆序递推）            │
│     归一化 A_t                                        │
│                    ↓                                 │
│  ③ K epochs × minibatch 更新                         │
│     Policy Loss  = -E[min(ρA, clip(ρ,1±ε)A)]        │
│     Value  Loss  = 0.5 * E[(V - R)²]                │
│     Entropy Bonus= +β * H[π]                         │
│     Total = L_clip + c1*L_vf - c2*H                 │
└─────────────────────────────────────────────────────┘
```

---

### 面试必答要点

| 问题 | 回答 |
|------|------|
| **为什么要 clip ratio？** | 防止单步更新过大导致策略崩塌，IS ratio 偏差太大时梯度不可信 |
| **GAE 中 λ 的作用？** | λ=0 退化为 TD(0)（低方差高偏差），λ=1 退化为 MC（高方差低偏差），λ=0.95 取折中 |
| **为什么 advantage 要归一化？** | 控制梯度量级，相当于自适应学习率，防止不同 episode 奖励尺度差异过大 |
| **为什么打乱 minibatch？** | 破坏时序相关性，让每个 minibatch 的梯度估计更接近 i.i.d |
| **Value clip 的意义？** | 防止 Critic 更新幅度超出旧值太多，与 policy clip 逻辑对称 |
| **PPO vs GRPO？** | PPO 需要 Critic 网络（显存翻倍）；GRPO 用组内均值做 baseline，适合 LLM 无状态场景 |