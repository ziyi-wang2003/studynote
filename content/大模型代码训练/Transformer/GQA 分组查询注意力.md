---
title: GQA 分组查询注意力
summary: "手撕 Grouped-Query Attention，理解 KV 头分组共享机制"
pinned: false
created: "2026-05-11 00:00"
updated: "2026-05-11 00:00"
order: 2
---

面试官让你手撕 **GQA（Grouped-Query Attention）**，考察的是对 KV 头分组共享的理解。GQA 是 MHA 和 MQA 的折中方案，被 LLaMA 2/3、Mistral 等主流模型采用。

---

### 1. 核心思想

- **MHA**：每个 Query 头对应独立的 K、V 头，共 $h$ 组 KV
- **MQA**：所有 Query 头共享同一组 K、V
- **GQA**：将 $h$ 个 Query 头分成 $g$ 组，每组共享一组 K、V

$$\text{Attention}(Q_i, K_{g(i)}, V_{g(i)}) = \text{softmax}\left(\frac{Q_i K_{g(i)}^T}{\sqrt{d_k}}\right) V_{g(i)}$$

其中 $g(i) = \lfloor i \cdot g / h \rfloor$ 是第 $i$ 个 Query 头对应的 KV 组索引。

---

### 2. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    - n_heads: Query 头数
    - n_kv_heads: KV 头数 (n_heads 必须能整除 n_kv_heads)
    - 当 n_kv_heads == n_heads 时退化为 MHA
    - 当 n_kv_heads == 1 时退化为 MQA
    """
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads 必须能整除 n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # 每组 KV 被多少个 Q 头共享
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        # 投影
        Q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: (B, n_heads, S, head_dim)

        K = self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # K, V: (B, n_kv_heads, S, head_dim)

        # 关键步骤：扩展 KV 头以匹配 Q 头数
        # (B, n_kv_heads, S, head_dim) → (B, n_heads, S, head_dim)
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, n_heads, S, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        # out: (B, n_heads, S, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)

    def _repeat_kv(self, x):
        """将 KV 头重复 n_rep 次以匹配 Q 头数"""
        if self.n_rep == 1:
            return x
        B, n_kv, S, D = x.shape
        # (B, n_kv, S, D) → (B, n_kv, 1, S, D) → (B, n_kv, n_rep, S, D)
        x = x.unsqueeze(2).expand(B, n_kv, self.n_rep, S, D)
        # → (B, n_kv * n_rep, S, D) = (B, n_heads, S, D)
        return x.reshape(B, n_kv * self.n_rep, S, D)


if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)

    # GQA: 8 个 Q 头, 2 个 KV 头 (每 4 个 Q 共享 1 组 KV)
    gqa = GroupedQueryAttention(d_model=D, n_heads=8, n_kv_heads=2)
    out = gqa(x)
    print(f"GQA Output: {out.shape}")  # (2, 10, 512)

    # 参数量对比
    mha_params = 4 * D * D  # MHA: 4 个 (D, D) 的矩阵
    gqa_params = sum(p.numel() for p in gqa.parameters())
    print(f"MHA 参数量: {mha_params:,}")
    print(f"GQA 参数量: {gqa_params:,}")
    print(f"GQA / MHA = {gqa_params / mha_params:.2%}")
```

---

### 3. 关键要点

| 要点 | 说明 |
|------|------|
| **`_repeat_kv`** | GQA 的核心操作，通过 `unsqueeze + expand + reshape` 将 KV 头广播到与 Q 头相同数量 |
| **参数节省** | KV 投影参数从 $2 \times d \times d$ 降低到 $2 \times d \times (d \cdot g / h)$ |
| **KV Cache 节省** | 推理时 KV Cache 大小按 $g/h$ 比例缩小，这是 GQA 的主要收益 |
| **退化情况** | `n_kv_heads == n_heads` → MHA；`n_kv_heads == 1` → MQA |

### 4. 面试常见追问

- **为什么不直接用 MQA？** MQA 压缩太激烈，模型质量有损失；GQA 在速度和质量间取得平衡
- **repeat_kv 用 expand 而不是 repeat？** `expand` 不分配新内存，只改变 stride，更高效
- **实际部署中 GQA 的收益在哪？** 主要在推理阶段减少 KV Cache 的显存占用和内存带宽消耗
