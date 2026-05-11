---
title: MQA 多查询注意力
summary: "手撕 Multi-Query Attention，理解所有 Q 头共享单组 KV 的极致压缩方案"
pinned: false
created: "2026-05-11 00:00"
updated: "2026-05-11 00:00"
order: 3
---

面试官让你手撕 **MQA（Multi-Query Attention）**，考察的是对 KV 共享机制的理解。MQA 由 Google 在 2019 年提出，是 GQA 的极端情况，所有 Query 头共享同一组 Key 和 Value。

---

### 1. 核心思想

标准 MHA 中每个头都有独立的 $Q_i, K_i, V_i$，MQA 则让所有头共享同一组 $K, V$：

$$\text{head}_i = \text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V$$

最终拼接所有头并投影：$\text{MQA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$

---

### 2. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    所有 Query 头共享同一组 Key 和 Value
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q 仍然有 n_heads 组投影
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        # K, V 只有 1 组投影 (关键区别)
        self.W_k = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: (B, n_heads, S, head_dim)

        K = self.W_k(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        # K, V: (B, 1, S, head_dim)
        # 广播机制自动将 K, V 扩展到 (B, n_heads, S, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, n_heads, S, S) — 广播自动处理

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        # out: (B, n_heads, S, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)


if __name__ == "__main__":
    B, S, D = 2, 10, 512
    n_heads = 8
    x = torch.randn(B, S, D)

    mqa = MultiQueryAttention(d_model=D, n_heads=n_heads)
    out = mqa(x)
    print(f"MQA Output: {out.shape}")  # (2, 10, 512)

    # 参数量对比
    head_dim = D // n_heads
    mha_kv_params = 2 * D * D                 # MHA 的 K, V 参数
    mqa_kv_params = 2 * D * head_dim           # MQA 的 K, V 参数
    print(f"MHA KV 参数量: {mha_kv_params:,}")
    print(f"MQA KV 参数量: {mqa_kv_params:,}")
    print(f"KV 参数压缩比: {mqa_kv_params / mha_kv_params:.2%}")

    # KV Cache 对比 (推理时每个 token 的 KV 缓存)
    mha_cache = 2 * n_heads * head_dim   # MHA: 2 * h * d_k
    mqa_cache = 2 * head_dim             # MQA: 2 * d_k
    print(f"KV Cache 压缩比: {mqa_cache / mha_cache:.2%}")
```

---

### 3. MQA vs MHA 对比

```
MHA:  Q₁ → K₁, V₁    每个 Q 头有独立的 KV
      Q₂ → K₂, V₂
      Q₃ → K₃, V₃
      Q₄ → K₄, V₄

MQA:  Q₁ → K, V       所有 Q 头共享同一组 KV
      Q₂ → K, V
      Q₃ → K, V
      Q₄ → K, V
```

---

### 4. 关键要点

| 要点 | 说明 |
|------|------|
| **广播机制** | K, V 形状为 `(B, 1, S, d_k)`，PyTorch 自动广播到 `(B, n_heads, S, d_k)` |
| **KV Cache** | 推理时 KV Cache 减少 $h$ 倍，这是 MQA 的核心优势 |
| **质量损失** | 共享 KV 会有一定精度下降，因此实践中更多使用 GQA 作为折中 |
| **代表模型** | PaLM, Falcon, StarCoder 等采用 MQA |

### 5. 面试常见追问

- **MQA 为什么能加速推理？** KV Cache 缩小 $h$ 倍，减少内存带宽瓶颈，解码阶段速度提升显著
- **为什么训练速度没有明显提升？** 训练不需要 KV Cache，计算量主要在 QKV 矩阵乘法，Q 投影没有缩减
- **MQA 和 GQA 的关系？** MQA 是 GQA 在 `n_kv_heads=1` 时的特例
