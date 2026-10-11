---
title: 手撕 KV Cache 与因果注意力
summary: "在分头后的注意力中追加 K/V，用历史长度修正因果 mask，同时支持 prefill 与单步 decode。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 4
---

**记住：新 Q 查全部 K/V；K/V 沿序列维追加；mask 必须考虑历史长度。** 约 5 分钟。

## 最小实现

假设 Q/K/V 已经完成线性投影和分头，Q/K 已按实际位置应用 RoPE。为突出 cache，不写投影、输出层和 padding mask；batch 内所有样本长度相同，每次输入至少一个新 token。

```python
import math
import torch

def cached_attention(q, k, v, cache=None):
    # q/k/v: [B, H, T_new, Dh]；cache: (past_k, past_v)
    past_len = 0 if cache is None else cache[0].size(-2)
    if cache is not None:
        k = torch.cat((cache[0], k), dim=-2)
        v = torch.cat((cache[1], v), dim=-2)
    # k/v: [B, H, T_total, Dh]
    tq, tk = q.size(-2), k.size(-2)
    q_pos = past_len + torch.arange(tq, device=q.device)
    k_pos = torch.arange(tk, device=q.device)
    visible = k_pos[None, :] <= q_pos[:, None]  # [T_new, T_total]

    scores = q.float() @ k.float().transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~visible, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = (probs @ v.float()).to(q.dtype)
    return out, (k, v)

q = torch.randn(1, 2, 4, 8)
k, v = torch.randn_like(q), torch.randn_like(q)
prefill, cache = cached_attention(q, k, v)  # 一次处理 4 个 prompt token
new_q = torch.randn(1, 2, 1, 8)
out, cache = cached_attention(new_q, torch.randn_like(new_q),
                              torch.randn_like(new_q), cache)
print(out.shape)       # [1, 2, 1, 8]
print(cache[0].shape)  # [1, 2, 5, 8]
```

## mask 为什么要带偏移

历史有 3 个 token，这次来了 2 个新 token。两个 query 的实际位置是 3、4，可见矩阵应该是：

```text
key 位置       0 1 2 3 4
query 位置 3   1 1 1 1 0
query 位置 4   1 1 1 1 1
```

直接对一个 `2 × 5` 矩阵用普通 `tril()` 会把历史 key 错误挡住。单步 decode 时只有当前 query，所有缓存 key 加当前 key 都是可见的。

## 易错点与追问

- **为什么缓存 K/V，不缓存 Q？** 当前输出只需要新 token 的 Q；历史 K/V 仍要作为被查询的上下文。
- **沿哪一维拼接？** 对 `[B,H,T,Dh]` 是 `dim=-2`，不是 head 维或特征维。
- **prefill 和 decode 的区别？** prefill 同时处理整个 prompt；decode 通常一次只输入新 token，每层各自维护缓存。
- **代价是什么？** 不再重算历史 K/V；单步 attention 仍要读所有历史 key，随上下文长度线性增长。`cat` 是教学实现，实际可预分配或分页存储。
- **KV Cache 显存怎么估？** `2 × 层数 × B × T × H_kv × Dh × 每元素字节数`；2 代表 K 和 V，GQA/MQA 用 KV 头数。

面试口述：“缓存复用历史 K/V，新 token 只算自己的 Q/K/V，再查询全部历史。缓存不会让 attention 的读取成本变成常数，长上下文会增加显存和带宽开销。”

参考：[Hugging Face 的 KV Cache 说明](https://huggingface.co/docs/transformers/cache_explanation)。
