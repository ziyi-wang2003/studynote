---
title: 手撕 DPO 损失
summary: "先写四个序列 log 概率的差，再补标签移位和回答 mask；不展开复杂推导。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 6
---

**记住：让策略相对参考模型，更偏好 chosen 而非 rejected。** 约 4 分钟。这里只写标准 sigmoid DPO。

## 最小损失：四个 log 概率

chosen 和 rejected 来自**同一个 prompt**。`pi_w`、`pi_l` 是训练模型对两个回答的序列 log 概率；`ref_w`、`ref_l` 来自冻结的参考模型，输入形状都是 `[B]`。

```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    policy_gap = pi_w - pi_l
    reference_gap = ref_w.detach() - ref_l.detach()
    logits = beta * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean()

pi_w = torch.tensor([-2.0], requires_grad=True)
pi_l = torch.tensor([-4.0], requires_grad=True)
ref_w = torch.tensor([-3.0])
ref_l = torch.tensor([-3.0])
loss = dpo_loss(pi_w, pi_l, ref_w, ref_l)
print(loss.item())  # 约 0.5981
loss.backward()
```

$$L=-\log\sigma\left(\beta\left[(\log\pi_w-\log\pi_l)-(\log\pi_{ref,w}-\log\pi_{ref,l})\right]\right)$$

## 追问：序列 log 概率怎么取

模型输出 `[B,T,Vocab]`，完整输入是 prompt + response。`response_mask` 与 `input_ids` 对齐，仅回答的有效 token 标 1；prompt、padding 标 0，通常把有效 EOS 也算入回答。

```python
def response_log_prob(logits, input_ids, response_mask):
    # 位置 t 的 logits 预测位置 t+1 的 token
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:]  # 真实词表 ID，padding ID 也必须合法
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = response_mask[:, 1:].to(token_logp.dtype)
    return (token_logp * mask).sum(dim=-1)  # [B]
```

分别对策略/参考、chosen/rejected 求这四个值，再调用 `dpo_loss`。参考模型应在 `torch.no_grad()` 中前向，策略模型保留梯度。

## 易错点与追问

- **为什么求和，不是求平均？** 标准 DPO 用回答序列的 log 概率，即各 token log 概率之和；长度归一化属于额外变体。
- **为什么不算 prompt？** 比较对象是同一 prompt 下的两个回答，mask 只保留回答 token。
- **mask 要不要移位？** 要和标签一起移位；回答第一项由它前一个位置的 logits 预测。
- **初始策略等于参考模型时损失多少？** 差值为 0，损失是 `log(2)`，约 0.6931；梯度仍然存在。
- **beta 是什么？** 源于 KL 正则的系数，也缩放这里的偏好 logit；不要把它简单说成学习率。
- **DPO 还要单独训练奖励模型吗？** 标准流程直接用偏好对训练策略，无需另一个显式奖励模型。

面试口述：“先分别求 chosen/rejected 的回答 log 概率，再减去参考模型的对应差值，乘 beta 后做负 log-sigmoid。参考不更新，标签和回答 mask 都与 logits 错开一个位置。”

参考：[DPO 论文](https://arxiv.org/abs/2305.18290)、[作者实现中的损失与序列概率](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py)。
