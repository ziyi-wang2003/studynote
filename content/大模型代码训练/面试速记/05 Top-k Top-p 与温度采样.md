---
title: 手撕 Top-k、Top-p 与温度采样
summary: "Top-k 和 Top-p 分别手写：一个留固定数量，一个按累计概率保留候选。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 5
---

**记住：温度改概率分布，Top-k 留固定数量，Top-p 留累计概率达到 p 的最短前缀。** 约 4 分钟。

两个例子都只处理单个样本：输入 `[Vocab]` 的有限 logits，返回一个 token ID。温度为 0 时直接贪心；温度大于 0 时，先用 `logits / temperature` 调整分布。

## Top-k：只保留最大的 k 个

步骤：**缩放 logits → 取最大的 k 个 → 在这 k 个里 softmax → 采样 → 映射回 token ID**。

```python
import torch

def sample_top_k(logits, k=50, temperature=1.0):
    assert logits.ndim == 1 and logits.numel() > 0
    assert k >= 1 and temperature >= 0
    if temperature == 0:
        return logits.argmax().item()
    x = logits.float() / temperature
    values, indices = torch.topk(x, min(k, x.numel()))
    probs = torch.softmax(values, dim=-1)
    selected = torch.multinomial(probs, num_samples=1).item()
    return indices[selected].item()

logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
print(sample_top_k(logits, k=1))  # 0
print(sample_top_k(logits, k=2))  # 只可能是 token 0 或 1
```

**易错点：**采样得到的是候选列表中的位置，要通过 `indices` 映射回原词表 ID。按索引取 k 个，遇到并列 logits 也不会留下超过 k 个候选。

## Top-p：保留累计概率达到 p 的前缀

步骤：**缩放 logits → 从大到小排序 → 计算累计概率 → 删除前缀外的候选 → 重新 softmax → 采样**。

```python
import torch

def sample_top_p(logits, p=0.9, temperature=1.0):
    assert logits.ndim == 1 and logits.numel() > 0
    assert 0 < p <= 1 and temperature >= 0
    if temperature == 0:
        return logits.argmax().item()
    x = logits.float() / temperature
    sorted_x, indices = torch.sort(x, descending=True)
    probs = torch.softmax(sorted_x, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # 当前 token 之前的概率和已达到 p，才删除当前 token
    remove = torch.zeros_like(probs, dtype=torch.bool)
    if p < 1.0:
        remove[1:] = cumulative[:-1] >= p
    sorted_x = sorted_x.masked_fill(remove, float('-inf'))
    probs = torch.softmax(sorted_x, dim=-1)  # 过滤后重新归一化
    selected = torch.multinomial(probs, num_samples=1).item()
    return indices[selected].item()

logits = torch.tensor([0.15, 0.50, 0.05, 0.30]).log()
print(sample_top_p(logits, p=0.70))  # 只可能是 token 1 或 3
```

**易错点：**要保留第一个让累计概率达到阈值的 token。

排序后概率为 `[0.50, 0.30, 0.15, 0.05]`，`p=0.70`。应该保留前两个，因为累计到第二个才达到 0.80；不能用“当前累计概率大于 p 就删掉当前项”，那会误删第二个。

这里检查的是**当前项之前**的累计概率，第一项总是保留。如果累计值恰好等于 p，前缀已经够了，后面的 token 就可以删掉。浮点边界和并列概率在不同库中可能有细微约定差异，面试先说清楚即可。

## 易错点与追问

- **温度越大越怎样？** `logits / temperature` 差异变小，分布更平；温度小于 1 通常更尖锐。
- **Top-k 和 Top-p 的区别？** Top-k 固定候选数量；Top-p 的候选数量随概率分布变化。
- **过滤后为什么重新 softmax？** 剩余候选的概率需要重新归一化，总和为 1。
- **排序后怎么回到原词表？** 两个例子都用 `indices[selected]` 映射回原 token ID。
- **Top-p 和 beam search 一样吗？** Top-p 限制当前步候选并随机采样；beam search 维护多条候选序列。

面试口述：“Top-k 选概率最高的 k 个候选；Top-p 选累计概率达到 p 的最短前缀。两种方法都在保留的候选里重新归一化并采样。”

参考：[Hugging Face 生成参数文档](https://huggingface.co/docs/transformers/main_classes/text_generation)。
