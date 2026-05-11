---
title: KL 散度
summary: "手撕 KL 散度计算，理解分布距离度量在知识蒸馏和 RLHF 中的应用"
pinned: false
created: "2026-05-11 00:00"
updated: "2026-05-11 00:00"
order: 4
---

面试官让你手撕 **KL 散度（Kullback-Leibler Divergence）**，考察的是对分布距离度量的理解。KL 散度在知识蒸馏、RLHF（PPO 中的 KL 惩罚）、VAE 等场景中广泛使用。

---

### 1. 数学定义

离散 KL 散度衡量分布 $Q$ 相对于 $P$ 的信息损失：

$$D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} = \sum_{i} P(i) [\log P(i) - \log Q(i)]$$

性质：
- $D_{KL}(P \| Q) \geq 0$（Gibbs 不等式）
- $D_{KL}(P \| Q) = 0$ 当且仅当 $P = Q$
- **不对称**：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

---

### 2. 从零实现

```python
import torch
import torch.nn.functional as F

def kl_divergence_manual(p_logits, q_logits, temperature=1.0):
    """
    手动计算 KL(P || Q)
    p_logits: 目标分布的 logits (如 teacher)
    q_logits: 近似分布的 logits (如 student)
    temperature: 温度系数，用于软化分布
    """
    # 对 logits 除以温度再做 softmax
    p = F.softmax(p_logits / temperature, dim=-1)
    log_p = F.log_softmax(p_logits / temperature, dim=-1)
    log_q = F.log_softmax(q_logits / temperature, dim=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl


def kl_divergence_pytorch(p_logits, q_logits, temperature=1.0):
    """
    使用 PyTorch 内置函数计算 KL(P || Q)
    注意: F.kl_div 的输入是 log_q 和 p (不是 log_p)
    """
    p = F.softmax(p_logits / temperature, dim=-1)
    log_q = F.log_softmax(q_logits / temperature, dim=-1)

    # F.kl_div(input=log_Q, target=P) 计算 KL(P || Q)
    # reduction='batchmean' 对 batch 维度求均值
    kl = F.kl_div(log_q, p, reduction='batchmean')
    return kl


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟 teacher 和 student 的 logits
    batch_size, vocab_size = 4, 1000
    teacher_logits = torch.randn(batch_size, vocab_size)
    student_logits = torch.randn(batch_size, vocab_size)

    # 方法 1: 手动实现
    kl_manual = kl_divergence_manual(teacher_logits, student_logits)
    print(f"手动 KL (per sample): {kl_manual}")
    print(f"手动 KL (mean): {kl_manual.mean().item():.4f}")

    # 方法 2: PyTorch 内置
    kl_pytorch = kl_divergence_pytorch(teacher_logits, student_logits)
    print(f"PyTorch KL: {kl_pytorch.item():.4f}")

    # 验证一致性
    print(f"差异: {abs(kl_manual.mean().item() - kl_pytorch.item()):.8f}")

    # 温度的影响
    for T in [0.5, 1.0, 2.0, 5.0]:
        kl = kl_divergence_manual(teacher_logits, student_logits, temperature=T).mean()
        print(f"T={T}: KL={kl.item():.4f}")
```

---

### 3. 知识蒸馏中的 KL 散度

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """
    知识蒸馏损失 = alpha * 软标签损失 + (1-alpha) * 硬标签损失
    """
    # 软标签损失: KL(teacher_soft || student_soft)
    # 乘以 T^2 是因为软化后梯度缩小了 1/T^2
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # 硬标签损失: 标准交叉熵
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

### 4. RLHF 中的 KL 惩罚

```python
def rlhf_kl_penalty(policy_logits, ref_logits, beta=0.1):
    """
    PPO/GRPO 中的 KL 惩罚项
    防止策略模型偏离参考模型太远
    reward_adjusted = reward - beta * KL(policy || ref)
    """
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1)

    # 逐 token 的 KL 散度
    policy_probs = F.softmax(policy_logits, dim=-1)
    kl_per_token = (policy_probs * (policy_logprobs - ref_logprobs)).sum(dim=-1)

    return beta * kl_per_token
```

---

### 5. 关键要点

| 要点 | 说明 |
|------|------|
| **不对称性** | $D_{KL}(P \| Q)$ 和 $D_{KL}(Q \| P)$ 不同；蒸馏用前者，PPO 常用后者 |
| **`F.kl_div` 的坑** | 输入是 `log_q`（不是 `q`），target 是 `p`（不是 `log_p`） |
| **温度 $T$** | $T$ 越大分布越平滑，能传递更多 dark knowledge；乘 $T^2$ 补偿梯度缩放 |
| **数值稳定** | 永远用 `log_softmax` 而不是 `log(softmax())`，避免下溢 |

### 6. 面试常见追问

- **KL 散度和交叉熵的关系？** $H(P, Q) = H(P) + D_{KL}(P \| Q)$，当 $P$ 固定时，最小化交叉熵等价于最小化 KL 散度
- **为什么蒸馏用 KL 而不是 MSE？** KL 散度在概率分布上有信息论意义，且对 logits 的缩放不敏感
- **前向 KL vs 反向 KL？** 前向 $D_{KL}(P \| Q)$ 是 mean-seeking（Q 覆盖 P 的所有模式），反向 $D_{KL}(Q \| P)$ 是 mode-seeking（Q 集中在 P 的某个模式）
