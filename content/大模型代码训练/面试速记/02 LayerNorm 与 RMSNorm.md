---
title: 手撕 LayerNorm 与 RMSNorm
summary: "两个归一化模块各十来行：LN 减均值除标准差，RMSNorm 只除均方根。"
created: "2026-10-12 00:00"
updated: "2026-10-12 00:00"
order: 2
---

**记住：每个 token 沿最后的隐藏维独立归一化；LN 减均值，RMSNorm 不减。** 约 4 分钟。

## 最小实现

```python
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        z = x.float()  # 用 float32 算统计量
        mean = z.mean(dim=-1, keepdim=True)
        var = ((z - mean) ** 2).mean(dim=-1, keepdim=True)
        y = (z - mean) * torch.rsqrt(var + self.eps)
        return (y * self.weight + self.bias).to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        z = x.float()
        mean_square = (z ** 2).mean(dim=-1, keepdim=True)
        y = z * torch.rsqrt(mean_square + self.eps)
        return (y * self.weight).to(x.dtype)

x = torch.randn(2, 3, 8)  # [B, T, D]
print(LayerNorm(8)(x).shape)  # [2, 3, 8]
print(RMSNorm(8)(x).shape)    # [2, 3, 8]
```

## 两个公式就够了

$$\mathrm{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\mathrm{mean}((x-\mu)^2)+\epsilon}}+\beta$$

$$\mathrm{RMSNorm}(x)=\gamma\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}$$

| | LayerNorm | 这里的 RMSNorm |
|---|---|---|
| 减均值 | 是 | 否 |
| 缩放 | 标准差 | 均方根 |
| 学习参数 | weight、bias | weight |
| 沿哪一维 | 最后的 D 维 | 最后的 D 维 |

## 易错点与追问

- **方差要不要无偏修正？** 不要。上面手写 `mean`；若用 `torch.var`，设置 `unbiased=False`，与 LayerNorm 一致。
- **为什么 `keepdim=True`？** `[B,T,1]` 才能自然广播到 `[B,T,D]`。
- **RMSNorm 输出均值是不是 0？** 不一定，因为没有减均值。
- **和 BatchNorm 差在哪？** 这里不跨 batch 求统计量，不维护 running mean/variance，训练和推理都使用当前输入的统计量。
- **Pre-Norm 怎么写？** `x = x + attention(norm(x))`；Post-Norm 是 `x = norm(x + attention(x))`。

面试口述：“LN 处理中心和尺度，RMSNorm 只处理尺度。两者都是每个 token 对隐藏维归一化，再乘可学习权重。”

参考：[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)、[RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) 官方文档。
