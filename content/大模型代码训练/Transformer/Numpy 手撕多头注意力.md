---
title: Numpy 手撕多头注意力
summary: "纯 Numpy 实现多头注意力，不依赖任何深度学习框架"
pinned: false
created: "2026-05-11 00:00"
updated: "2026-05-11 00:00"
order: 5
---

面试中有时会要求不使用 PyTorch/TensorFlow，**纯 Numpy** 手撕多头注意力，考察对底层矩阵运算的理解。

---

### 完整实现

```python
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def multi_head_attention(X, W_q, W_k, W_v, W_o, h):
    """
    X: (batch, seq_len, d_model)
    W_q, W_k, W_v: (d_model, d_model)
    W_o: (d_model, d_model)
    h: number of heads
    """
    batch, seq_len, d_model = X.shape
    d_k = d_model // h
    
    # 1. 线性映射
    Q = X @ W_q  # (batch, seq_len, d_model)
    K = X @ W_k
    V = X @ W_v
    
    # 2. 切分成多头
    Q = Q.reshape(batch, seq_len, h, d_k).transpose(0, 2, 1, 3)  # (batch, h, seq_len, d_k)
    K = K.reshape(batch, seq_len, h, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, h, d_k).transpose(0, 2, 1, 3)
    
    # 3. 计算注意力
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (batch, h, seq_len, seq_len)
    attn = softmax(scores, axis=-1)
    out = attn @ V  # (batch, h, seq_len, d_k)
    
    # 4. 合并多头
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    
    # 5. 输出线性映射
    out = out @ W_o  # (batch, seq_len, d_model)
    
    return out

# 测试
batch, seq_len, d_model, h = 2, 4, 8, 2
X = np.random.rand(batch, seq_len, d_model)
W_q = np.random.rand(d_model, d_model)
W_k = np.random.rand(d_model, d_model)
W_v = np.random.rand(d_model, d_model)
W_o = np.random.rand(d_model, d_model)

out = multi_head_attention(X, W_q, W_k, W_v, W_o, h)
print(out.shape)  # (2, 4, 8)
```

---

### 逐步形状变化

```
输入:  X → (batch, seq_len, d_model)           = (2, 4, 8)

1. 线性映射:
   Q = X @ W_q → (2, 4, 8)
   K = X @ W_k → (2, 4, 8)
   V = X @ W_v → (2, 4, 8)

2. reshape + transpose 切分多头:
   Q → reshape(2, 4, 2, 4) → transpose(0,2,1,3) → (2, 2, 4, 4)
                                                     B  h  S  d_k

3. 注意力计算:
   scores = Q @ K^T → (2, 2, 4, 4)    # seq_len × seq_len
   attn = softmax(scores) → (2, 2, 4, 4)
   out = attn @ V → (2, 2, 4, 4)      # 每个头的输出

4. 合并多头:
   out → transpose(0,2,1,3) → (2, 4, 2, 4) → reshape → (2, 4, 8)
                                                          B  S  d_model

5. 输出映射:
   out = out @ W_o → (2, 4, 8)
```

---

### 关键要点

| 步骤 | 操作 | 目的 |
|------|------|------|
| `reshape + transpose` | `(B,S,D)` → `(B,S,h,d_k)` → `(B,h,S,d_k)` | 将 d_model 拆分为 h 个头，把 head 维度提前方便批量矩阵乘 |
| `/ np.sqrt(d_k)` | scores 缩放 | 防止点积过大导致 softmax 饱和、梯度消失 |
| `softmax 数值稳定` | 减去 `max` 再 `exp` | 避免 `exp` 溢出，这是面试必考细节 |
| `transpose + reshape` | `(B,h,S,d_k)` → `(B,S,D)` | 多头拼接，恢复原始维度 |

### 与 PyTorch 版本的区别

| | Numpy | PyTorch |
|--|-------|---------|
| 权重 | 裸矩阵 `(d, d)`，用 `@` 手动乘 | `nn.Linear` 封装，含 bias |
| softmax | 手写，需自行处理数值稳定性 | `F.softmax` 内置稳定处理 |
| 多头切分 | `reshape + transpose` | 同样，或用 `view + permute` |
| Mask | 需手动加 `-1e9` | `masked_fill(mask, -inf)` |
| 反向传播 | 无，纯前向 | 自动微分 |

### 面试常见追问

- **为什么 softmax 要减去 max？** 防止 `exp` 溢出，$\text{softmax}(x) = \text{softmax}(x - c)$ 对任意常数 $c$ 成立
- **transpose 的顺序为什么是 (0,2,1,3)？** 需要把 head 维度移到 seq_len 前面，这样 `@` 运算在最后两个维度上进行矩阵乘法，batch 和 head 维度自动并行
- **能加 causal mask 吗？** 可以，在 scores 上加一个上三角为 `-inf` 的矩阵：`mask = np.triu(np.ones((S,S)), k=1) * (-1e9)`
