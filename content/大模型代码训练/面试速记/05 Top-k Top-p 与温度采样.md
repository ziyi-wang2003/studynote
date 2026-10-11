---
title: 手撕 Top-k、Top-p 与温度采样
summary: "只处理下一 token 的一行 logits，记住过滤顺序和 Top-p 的阈值边界。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 5
---

**记住：温度改概率分布，Top-k 留固定数量，Top-p 留累计概率达到 p 的最短前缀。** 约 4 分钟。

## 最小实现

面试先写单个样本：输入 `[Vocab]` 的有限 logits，返回一个 token ID。这里规定温度为 0 时直接贪心，跳过随机采样。

```python
import torch

def sample_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    assert logits.ndim == 1 and logits.numel() > 0
    assert temperature >= 0 and top_k >= 0 and 0 < top_p <= 1
    if temperature == 0:
        return logits.argmax().item()
    x = logits.float().clone() / temperature

    if top_k > 0:
        k = min(top_k, x.numel())
        indices = torch.topk(x, k).indices
        keep = torch.zeros_like(x, dtype=torch.bool)
        keep[indices] = True
        x = x.masked_fill(~keep, float('-inf'))

    if top_p < 1.0:
        sorted_x, order = torch.sort(x, descending=True)
        probs = torch.softmax(sorted_x, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # 若当前 token 之前的概率和已达到 p，才删除当前 token
        remove = torch.zeros_like(probs, dtype=torch.bool)
        remove[1:] = cumulative[:-1] >= top_p
        sorted_x = sorted_x.masked_fill(remove, float('-inf'))
        x = torch.empty_like(x).scatter(0, order, sorted_x)

    probs = torch.softmax(x, dim=-1)  # 过滤后重新归一化
    return torch.multinomial(probs, num_samples=1).item()

logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
print(sample_token(logits, temperature=0))  # 0
print(sample_token(logits, top_k=1))       # 0
```

## Top-p 最容易错在哪里

排序后概率为 `[0.50, 0.30, 0.15, 0.05]`，`p=0.70`。应该保留前两个，因为累计到第二个才达到 0.80；不能用“当前累计概率大于 p 就删掉当前项”，那会误删第二个。

这里检查的是**当前项之前**的累计概率，第一项总是保留。如果累计值恰好等于 p，前缀已经够了，后面的 token 就可以删掉。浮点边界和并列概率在不同库中可能有细微约定差异，面试先说清楚即可。

## 易错点与追问

- **温度越大越怎样？** `logits / temperature` 差异变小，分布更平；温度小于 1 通常更尖锐。
- **先做哪个？** 本例采用温度 → Top-k → Top-p；Top-p 用 Top-k 过滤后重新归一化的概率，两者一起用时顺序影响候选集。
- **Top-k 为什么不用小于第 k 大的阈值？** 阈值法遇到并列值可能留下超过 k 个；这里按索引选恰好 k 个。
- **排序后怎么回到原词表？** 用 `scatter`，再按原 token ID 采样。
- **Top-p 和 beam search 一样吗？** Top-p 限制当前步候选并随机采样；beam search 维护多条候选序列。

面试口述：“我先缩放 logits，再删掉 Top-k 和 Top-p 之外的候选，最后重新 softmax 并采样。Top-p 要保留第一个让累计概率达到阈值的 token。”

参考：[Hugging Face 生成参数文档](https://huggingface.co/docs/transformers/main_classes/text_generation)。
