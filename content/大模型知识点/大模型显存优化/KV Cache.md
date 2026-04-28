---
created: '2026-04-24 12:44:00.777292+00:00'
order: 0
pinned: false
summary: ''
title: KV Cache
updated: '2026-04-24 12:44:00.777331+00:00'
---

## KV Cache：学习笔记

### 1. 动机：自回归生成中的重复计算

在基于 Transformer 的自回归生成（如 GPT、LLaMA 等大语言模型推理）中，模型每次迭代仅生成一个 token，并将该 token 追加至输入序列，用于下一步的预测。对于第 \(t\) 步，输入序列为 \([x_1, x_2, \dots, x_t]\)，模型需计算第 \(t+1\) 个 token 的概率分布。此过程的核心计算量集中于注意力机制：对于每一个查询位置 \(i\)，需要计算其与所有键位置 \(j \le i\) 的注意力分数。

若朴素地每次重新计算全部注意力，第 \(t\) 步的计算复杂度为 \(O(t \cdot d)\)（忽略头数等细节）。随着生成步数增长至总长度 \(T\)，总计算量为 \(O(T^2 \cdot d)\)，其中绝大部分计算是重复的：在第 \(t\) 步，前 \(t-1\) 个 token 的键（Key）和值（Value）实际上与第 \(t-1\) 步的结果完全相同。**KV Cache 的核心思想正是缓存这些已计算出的键和值向量，避免每一步都重新计算历史 token 的键值投影**。这一优化将总计算量从 \(O(T^2)\) 降至 \(O(T)\)（注意力部分的计算开销），是 Transformer 推理得以实用化的关键手段。

### 2. 注意力机制的回顾与符号定义

设一个单头注意力层。输入序列长度为 \(L\)，每个 token 的隐藏维度为 \(d_{\text{model}}\)。通过三个可学习的权重矩阵 \(W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}\)（通常 \(d_k = d_{\text{model}} / h\)，\(h\) 为头数），将输入 \(\mathbf{X} \in \mathbb{R}^{L \times d_{\text{model}}}\) 投影为：

\[
\mathbf{Q} = \mathbf{X} W_Q, \quad \mathbf{K} = \mathbf{X} W_K, \quad \mathbf{V} = \mathbf{X} W_V
\]

其中 \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d_k}\)。

注意力输出为：

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}
\]

在因果语言模型（Causal LM）中，位置 \(i\) 的查询只能与位置 \(j \le i\) 的键值交互。为此引入下三角 mask，将 \(\mathbf{Q}\mathbf{K}^T\) 中 \(j > i\) 的元素设为 \(-\infty\)。

### 3. KV Cache 的基本原理

设第 \(t\) 步输入为最新 token \(x_t\)（及其对应位置编码）。模型维护两个缓存数组 \(\mathbf{K}_{\text{cache}}^{(t-1)}\) 和 \(\mathbf{V}_{\text{cache}}^{(t-1)}\)，形状均为 \((t-1) \times d_k\)，存储了前 \(t-1\) 个 token 的键和值。

**前向计算过程**：

1. 仅对当前 token 计算其查询、键、值：
   \[
   \mathbf{q}_t = x_t W_Q, \quad \mathbf{k}_t = x_t W_K, \quad \mathbf{v}_t = x_t W_V
   \]
   形状均为 \((1, d_k)\)。

2. 将 \(\mathbf{k}_t, \mathbf{v}_t\) 拼接到缓存末尾，得到：
   \[
   \mathbf{K}^{(t)} = \text{concat}(\mathbf{K}_{\text{cache}}^{(t-1)}, \mathbf{k}_t) \in \mathbb{R}^{t \times d_k}
   \]
   \[
   \mathbf{V}^{(t)} = \text{concat}(\mathbf{V}_{\text{cache}}^{(t-1)}, \mathbf{v}_t) \in \mathbb{R}^{t \times d_k}
   \]

3. 用当前查询 \(\mathbf{q}_t\) 与全部键 \(\mathbf{K}^{(t)}\) 计算注意力分数：
   \[
   \mathbf{s}_t = \frac{\mathbf{q}_t \mathbf{K}^{(t)T}}{\sqrt{d_k}} \in \mathbb{R}^{1 \times t}
   \]
   由于因果 mask，\(\mathbf{q}_t\) 本就不能与未来键交互，此处未来键尚未产生，因此自然满足因果性。

4. 对 \(\mathbf{s}_t\) 做 softmax 得到注意力权重 \(\mathbf{a}_t\)，然后加权值向量：
   \[
   \mathbf{o}_t = \mathbf{a}_t \mathbf{V}^{(t)} \in \mathbb{R}^{1 \times d_k}
   \]

5. \(\mathbf{o}_t\) 经输出投影后得到该 token 对应的输出。

**缓存更新**：将第 \(t\) 步结束后，更新 \(\mathbf{K}_{\text{cache}}^{(t)}, \mathbf{V}_{\text{cache}}^{(t)}\) 供下一步使用。

### 4. 计算与内存收益分析

**计算量对比**（忽略 softmax、非线性等，仅计矩阵乘法）：

- 无 KV Cache：每步 \(t\) 需计算全部 \(t\) 个 token 的 \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\)，复杂度 \(O(t \cdot d_{\text{model}} \cdot d_k)\)；注意力矩阵乘法 \(\mathbf{Q}\mathbf{K}^T\) 复杂度 \(O(t^2 \cdot d_k)\)。总计算量随 \(t\) 平方增长。
- 有 KV Cache：每步仅计算当前 token 的 \(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t\)，复杂度 \(O(d_{\text{model}} \cdot d_k)\)；注意力矩阵乘法 \(\mathbf{q}_t \mathbf{K}^T\) 复杂度 \(O(t \cdot d_k)\)。因此每步计算量随 \(t\) 线性增长。总生成 \(T\) 个 token（不含预填充阶段）的注意力计算量从 \(O(T^3)\) 降至 \(O(T^2)\)？需要澄清。

更精确地：预填充阶段（处理 prompt）通常一次性计算所有 prompt token 的 KV 并缓存，该阶段复杂度 \(O(L_{\text{prompt}}^2 \cdot d_k)\)。之后每生成一个 token 需要 \(O(t \cdot d_k)\)，其中 \(t\) 从 \(L_{\text{prompt}}+1\) 增长到 \(L_{\text{total}}\)。总生成阶段复杂度 \(\sum_{t=L_{\text{prompt}}+1}^{L_{\text{total}}} t = O(L_{\text{total}}^2 - L_{\text{prompt}}^2)\)。实际上依然有平方项，但常数远小于无缓存时的立方项？无缓存时每一步都重新计算整个序列的注意力，第 \(t\) 步复杂度 \(O(t^2)\)，总复杂度 \(O(T^3)\)。KV Cache 将总复杂度从立方降为平方。更关键的是**实际运行中的内存访问和重用收益**：避免了重复投影历史 token，大幅降低 FLOPs。

**内存开销**：缓存需要存储所有历史 token 的键和值。对于每层每头，形状为 \((L_{\text{max}}, d_k)\)。设层数 \(n_{\text{layers}}\)，头数 \(n_{\text{heads}}\)，则总缓存元素数为 \(2 \times n_{\text{layers}} \times L_{\text{max}} \times n_{\text{heads}} \times d_k\)（2 对应 K 和 V）。以 LLaMA2-7B 为例：\(n_{\text{layers}}=32\)，\(n_{\text{heads}}=32\)，\(d_k=128\)，半精度浮点（2 字节）下，每 token 每层缓存大小为 \(2 \times 32 \times 128 \times 2 = 16384\) 字节 ≈ 16 KB。对于 \(L_{\text{max}}=4096\) 的序列，总缓存约 \(32 \times 4096 \times 16\text{KB} = 2\text{GB}\)？计算：每层缓存大小 \(L \times 2 \times n_{\text{heads}} \times d_k \times \text{bytes} = 4096 \times 2 \times 32 \times 128 \times 2 = 4096 \times 16384 = 67,108,864\) 字节 ≈ 64 MB 每层，32 层约 2 GB。这是实际推理中显存占用的主要部分，也是长上下文生成时的瓶颈。

### 5. 实现中的关键细节

**缓存存储结构**：通常以连续张量形式存储，形状为 \([ \text{batch\_size}, \text{seq\_len}, \text{num\_heads}, \text{head\_dim} ]\) 或 \([ \text{batch\_size}, \text{num\_heads}, \text{seq\_len}, \text{head\_dim} ]\)。两种布局影响内存访问模式。FlashAttention 等高效实现偏好后者以减少转置开销。

**预填充与解码阶段**：在推理框架（如 HuggingFace Transformers、vLLM）中，首个请求（prompt）先经过预填充（prefill）阶段，一次性计算所有 prompt token 的 KV 并填充缓存。随后进入解码（decoding）阶段，逐 token 生成，每步仅传入新 token 并更新缓存。

**动态长度与批处理**：服务多个请求时，不同序列长度不同，KV Cache 需支持动态扩展。常见方案是预分配最大长度张量，用长度掩码指示有效部分；或使用分页式 KV Cache（如 vLLM 的 PagedAttention），以固定大小块管理，避免内部碎片。

**MQA 与 GQA 的影响**：多查询注意力（Multi-Query Attention，MQA）和分组查询注意力（Grouped-Query Attention，GQA）显著减少 KV Cache 尺寸。MQA 中所有头共享同一组 KV，缓存大小变为原来的 \(1/n_{\text{heads}}\)。GQA 将头分为若干组，每组共享 KV，在性能与质量间取得平衡。这一改进对于长序列推理至关重要。

**数值稳定性**：在 softmax 之前，\(\mathbf{q}_t \mathbf{K}^T\) 中的值可能较大，需减去最大值。由于 \(\mathbf{q}_t\) 与历史键逐点相乘，可在线计算 max 并安全进行。

### 6. 变体与进阶主题

**KV Cache 的随步更新与不变性**：对于绝对值位置编码（如原始 Transformer 的 sinusoidal），位置嵌入在输入层就已加入，因此不同步的键向量自然携带位置信息，缓存直接存储投影后向量即可。对于**旋转位置编码（RoPE）**，键和查询的旋转与绝对位置相关。在自回归生成中，历史 token 的旋转角度固定，因此仍可缓存其旋转后的键；但当前查询的旋转需结合当前步数。RoPE 实现时通常缓存旋转前的键，每步重新旋转？实际上，由于 RoPE 是相对位置编码的一种高效实现，常见做法是缓存已旋转的键（因为对于固定历史位置，其键不需要改变）。然而，为支持更灵活的注意力变体，也有实现缓存未旋转的键并在每步实时旋转。这会影响缓存命中后的计算量。

**与 FlashAttention 的协同**：FlashAttention 通过分块计算和重计算避免显存中实例化完整注意力矩阵，在训练和长序列预填充中效果显著。在自回归解码阶段，KV Cache 本身已大幅减少计算，但 FlashAttention 的分块策略可以进一步优化单步中 \(\mathbf{q}_t \mathbf{K}^T\) 的计算，尤其是当缓存很大时，通过分块减少 HBM 访问。实际上，FlashAttention-2 及后续版本均原生支持带有 KV Cache 的因果注意力解码。

**连续批处理（Continuous Batching）中的 KV Cache 管理**：传统静态批处理需等待批次中最长序列完成，导致大量空闲。连续批处理允许请求动态加入和离开，每个请求的 KV Cache 独立管理。系统需支持高效的 KV Cache 复用、丢弃、重计算策略。PagedAttention 将缓存划分为固定大小的“块”，以虚拟内存的页表方式映射，显著提升 GPU 利用率。

### 7. 易混淆点澄清

- **KV Cache 是否缓存查询（Q）？** 不。每一步的查询只对当前 token 计算一次，其后不再需要。历史查询在生成后续 token 时无任何作用，因此无需缓存。
- **缓存是否包含位置编码？** 取决于实现。若位置编码在输入嵌入之后、投影之前加入，则投影后的键值已隐含位置信息。对于 RoPE，通常将旋转应用在 Q/K 投影之后，因此键缓存存储的是旋转后的结果。
- **训练时是否使用 KV Cache？** 典型训练中，由于序列已知，使用 teacher forcing 并行计算所有位置的输出，不需要逐步缓存。但某些训练场景（如强化学习或流式训练）可能部分采用类似机制。
- **中间激活与 KV Cache 的区别**：KV Cache 本质是中间计算结果的持久化存储，用于跨时间步复用。而反向传播中保留的激活是暂时的。

### 8. 实践考量

**显存占用估算公式**：

\[
\text{Cache\_bytes} = 2 \times n_{\text{layers}} \times L_{\text{max}} \times n_{\text{heads}} \times d_k \times \text{bytes\_per\_element}
\]

以 FP16 为例，bytes_per_element=2。对于 LLaMA2-7B，单请求 4096 长度约 2GB。实际部署时需权衡最大序列长度、批大小、可用显存。

**量化与稀疏化**：KV Cache 可应用 INT8、INT4 量化甚至更激进的方法（如 KV 缓存量化至 2-bit）。研究表明，对键值缓存进行适当量化（按通道或 token-wise）可保持精度，大幅降低内存。此外，**流式 LLM** 等方法提出只缓存部分重要 token（如滑动窗口 + 永久 token），以支持无限长上下文。

**多 GPU 推理中的 KV Cache**：张量并行（Tensor Parallelism）将注意力头分布于不同 GPU，每个 GPU 只负责部分头的 KV 缓存，无需跨 GPU 复制整个缓存。流水线并行则在阶段间传递激活，KV 缓存通常随模型阶段分片。

### 9. 总结

KV Cache 是 Transformer 自回归推理的核心优化，通过空间换时间的思想，显式缓存历史 token 的键值对，避免每步重复计算。其引入的额外内存开销是长序列生成的主要瓶颈，催生了 MQA、GQA、量化、分页管理等技术。理解 KV Cache 不仅有助于高效部署大语言模型，也为设计更长上下文、更低延迟的推理系统奠定理论基础。在实际代码中，如 HuggingFace 的 `past_key_values` 参数即为该机制的直接体现。掌握其原理、计算模式与存储权衡，是深入大模型工程优化的必备能力。