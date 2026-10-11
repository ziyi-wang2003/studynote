---
title: 手撕 RoPE 旋转位置编码
summary: "把 Q/K 的相邻两个维度当作二维向量，按位置旋转；记住频率、广播和 cache 的位置偏移。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 3
---

**记住：RoPE 对 Q、K 做成对旋转，通常不改 V。** 约 4 分钟。这里只写基础 RoPE，不展开长上下文缩放。

## 最小实现：相邻维度配对

```python
import torch

def apply_rope(x, positions, base=10000.0):
    # x: [B, H, T, Dh]；positions: [T]，Q/K 各调用一次
    d = x.size(-1)
    assert d % 2 == 0
    assert positions.numel() == x.size(-2)
    inv_freq = base ** (-torch.arange(0, d, 2,
                                    device=x.device, dtype=torch.float32) / d)
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :]  # [1, 1, T, Dh/2]
    sin = angles.sin()[None, None, :, :]

    z = x.float()
    even, odd = z[..., 0::2], z[..., 1::2]
    y_even = even * cos - odd * sin
    y_odd = even * sin + odd * cos
    return torch.stack((y_even, y_odd), dim=-1).flatten(-2).to(x.dtype)

q = torch.randn(2, 4, 3, 8)
k = torch.randn(2, 4, 3, 8)
positions = torch.arange(3)
q_rot = apply_rope(q, positions)
k_rot = apply_rope(k, positions)
print(q_rot.shape)  # [2, 4, 3, 8]
```

## 只背一个二维旋转

第 `i` 对维度，位置 `m` 对应角度 `m × base^(-2i/Dh)`：

$$\begin{pmatrix}x'_{2i}\\x'_{2i+1}\end{pmatrix}
=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
\begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}$$

Q 在位置 m 旋转，K 在位置 n 旋转后，点积中的旋转关系只依赖 `n-m`，从而把相对位置带入 attention。

## 易错点与追问

- **频率的分母是哪个维度？** 这里用每个头的 `Dh`，不是整个隐藏维 `D`。
- **位置 0 应该怎样？** 角度为 0，输入保持不变；旋转还应保持每对维度的平方和。
- **单步解码怎么传位置？** 无 padding、无滑窗时，若缓存有 P 个 token，新 token 用 `positions = torch.tensor([P])`，不能每步都用 0。
- **K 缓存存哪一个？** 常规实现存已经应用 RoPE 的 K，读取历史 K 时不要再次旋转。
- **为什么别人的 `rotate_half` 不一样？** 有些实现配对前后两半，本例配对相邻维。数学旋转相同，但维度排列必须和模型权重约定一致，不能直接混用。

面试口述：“我先生成位置乘频率的角度，再对 Q/K 每两个维度做二维旋转。旋转后点积编码相对位置，缓存解码时要接着历史位置往后走。”

参考：[RoFormer 原论文](https://arxiv.org/abs/2104.09864)。
