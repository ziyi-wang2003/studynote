---
title: LoRA 实现
summary: "从零实现 LoRA (Low-Rank Adaptation) 线性层，理解低秩分解微调的核心原理"
pinned: false
created: "2026-05-03 00:00"
updated: "2026-05-03 00:00"
order: 1
---

# LoRA 实现

## 核心思想

LoRA (Low-Rank Adaptation) 通过在冻结的预训练权重旁添加低秩分解矩阵来实现高效微调：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d_{out} \times r}$，$A \in \mathbb{R}^{r \times d_{in}}$，$r \ll \min(d_{in}, d_{out})$。

## 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
  def __init__(self, in_features, out_features, rank = 8, alpha = 1.0, dropout = 0.0):
    super().__init__()
    self.weight = nn.Linear(in_features, out_features, bias = False)
    self.weight.weight.requires_grad = False

    self.lora_a = nn.Linear(in_features, rank, bias = False)
    self.lora_b = nn.Linear(rank, out_features, bias = False)

    self.alpha = alpha
    self.rank = rank

    self.scaling = self.alpha / self.rank

    self.dropout = nn.Dropout(dropout)

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.lora_a.weight, a = math.sqrt(5))
    nn.init.zeros_(self.lora_b.weight)

  def forward(self, x):
    with torch.no_grad():
      original_output = self.weight(x)
    
    lora_output = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

    return original_output + lora_output


if __name__ == "__main__":
  x = torch.randn(2,5,10)

  layer = LoRALinear(10,20,rank=4)
  out = layer(x)

  print(f"Output Shape: {out.shape}")

  diff = (out - layer.weight(x)).abs().sum()

  print(diff.item())
```

## 关键设计点

### 1. 冻结原始权重

```python
self.weight.weight.requires_grad = False
```

原始线性层的权重不参与梯度更新，前向传播时用 `torch.no_grad()` 包裹以节省显存。

### 2. 低秩分解

```python
self.lora_a = nn.Linear(in_features, rank, bias = False)   # 降维: d_in → r
self.lora_b = nn.Linear(rank, out_features, bias = False)   # 升维: r → d_out
```

参数量从 $d_{in} \times d_{out}$ 降低到 $r \times (d_{in} + d_{out})$。当 $r=8$，$d_{in}=d_{out}=4096$ 时，参数量降低为原来的 $\frac{2 \times 8}{4096} \approx 0.4\%$。

### 3. 初始化策略

- **A 矩阵**：Kaiming 均匀初始化，保证训练初期有合理的梯度
- **B 矩阵**：零初始化，确保训练开始时 $\Delta W = BA = 0$，不改变原始模型行为

### 4. Scaling 因子

```python
self.scaling = self.alpha / self.rank
```

$\alpha$ 是超参数，控制 LoRA 分支的贡献大小。除以 rank 使得调整 rank 时不需要重新调学习率。

### 5. Dropout

在输入进入 LoRA 分支前施加 dropout，起正则化作用，防止低秩适配过拟合。

## 验证

运行后 `diff` 的值应为一个较小的非零值（因为 B 初始化为 0，所以初始时 LoRA 输出为 0，diff 接近 0；但由于浮点精度可能有微小偏差）。
