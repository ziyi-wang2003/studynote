---
title: 手撕稳定 Softmax 与交叉熵
summary: "减 max 防溢出，log-sum-exp 算交叉熵，顺手记住语言模型的标签移位。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 1
---

**记住：Softmax 先减最大值；交叉熵直接从 logits 算，不要先 softmax 再 log。** 约 4 分钟。

## 最小实现

下面只处理有限 logits、整数类别标签，不包含 class weight、label smoothing 和 padding。

```python
import torch

def softmax(x, dim=-1):
    x = x.float()
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(logits, targets):
    # logits: [N, C]；targets: [N]，dtype=torch.long
    x = logits.float()
    shifted = x - x.max(dim=-1, keepdim=True).values
    log_probs = shifted - shifted.exp().sum(-1, keepdim=True).log()
    target_logp = log_probs.gather(-1, targets[:, None]).squeeze(-1)
    return -target_logp.mean()

logits = torch.tensor([[1000., 1001., 1002.]])
print(softmax(logits))  # 约 [[0.0900, 0.2447, 0.6652]]
print(cross_entropy(logits, torch.tensor([2])))  # 约 0.4076
```

核心公式：

$$p_i = \frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j$$

$$L=-\log p_y=\log\sum_j e^{z_j-m}-(z_y-m)$$

减掉同一个常数不会改变概率；使用平移后的 logits 计算 log 概率，还能避免先算出极小概率再取 log。

## 大模型多一个标签移位

`logits[:, t]` 预测 `input_ids[:, t+1]`。下例假定至少有两个 token，没有 padding；真实训练还要屏蔽 padding 和不参与监督的 prompt。

```python
def next_token_loss(logits, input_ids):
    # logits: [B, T, Vocab]；input_ids: [B, T]
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    target = input_ids[:, 1:].reshape(-1)
    return cross_entropy(pred, target)
```

## 易错点与追问

- **在哪一维 softmax？** 分类在类别维，attention 在 key 的序列维，通常都是 `dim=-1`。
- **为什么不用 `log(softmax(x))`？** 极小概率可能变成 0，取 log 后是 `-inf`；用 log-sum-exp 或 `F.log_softmax`。
- **CE 梯度是什么？** 单个样本对 logits 的梯度是 `p - one_hot(y)`；batch 均值再除以样本数。
- **mask 能把整行都填成 `-inf` 吗？** 不能直接套这段 Softmax，会出现 NaN；attention 应保证有效 query 至少有一个可见 key。

面试口述：“我沿类别维减最大值，用指数和归一化；交叉熵直接算正确类别的负 log 概率。语言模型里预测与标签要错开一个位置。”

参考：[PyTorch CrossEntropyLoss 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
