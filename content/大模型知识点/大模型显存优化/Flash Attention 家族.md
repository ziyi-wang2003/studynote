---
created: '2026-04-24 12:47:31.677138+00:00'
order: 1
pinned: false
summary: ''
title: Flash Attention 家族
updated: '2026-05-01 01:02:19+08:00'
---

## Flash Attention: 从 IO 感知到硬件极致利用

### 1 动机：注意力机制的真正瓶颈

Transformer 中的自注意力机制计算流程为

\[
\mathbf{S} = \mathbf{Q} \mathbf{K}^T, \quad \mathbf{P} = \text{softmax}\left( \frac{\mathbf{S}}{\sqrt{d_k}} \right), \quad \mathbf{O} = \mathbf{P} \mathbf{V}
\]

其中 \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}\)，\(N\) 为序列长度，\(d\) 为头维度。**朴素实现需要显式物化并存储 \(N \times N\) 的注意力矩阵 \(\mathbf{S}\) 和 \(\mathbf{P}\)**，导致时间与内存复杂度均为 \(O(N^2 d)\)。对于长序列（如 \(N=64\)K），\(N^2\) 约 40 亿，存储半精度矩阵即需约 8GB 显存，这成为 Transformer 扩展至长上下文的主要障碍。

然而，计算量并非唯一的瓶颈。在 GPU 的存储层次中，**高带宽内存**（High Bandwidth Memory, HBM）容量大（如 A100 为 40–80GB）但带宽约 1.5–2.0 TB/s；**片上 SRAM** 容量小（每 SM 约 192KB，总计约 20MB）但带宽可达约 19 TB/s——差距约一个数量级。在 A100 上，从 HBM 读取数据比从 SRAM 读取慢约 15 倍。

朴素注意力实现将 \(\mathbf{S}\) 写入 HBM，读回计算 softmax，将 \(\mathbf{P}\) 重新写入 HBM，再读回与 \(\mathbf{V}\) 相乘。**大量 HBM 访问的实际开销远超浮点运算本身**，使注意力机制受限于内存带宽（memory-bound），而非计算能力（compute-bound）。在 GPU 持续增加的 FLOPS 与相对滞后的内存带宽之间，这一矛盾日益突出。

![FlashAttention IO 感知分块计算](/static/images/uploads/大模型显存优化/flashattention-io-aware-tiling.png)

图中左侧是朴素注意力的瓶颈：完整 \(N \times N\) 分数矩阵和 softmax 矩阵反复写入、读出 HBM。中间的 FlashAttention 把 Q/K/V 切成 tile 放入片上 SRAM，在块内完成打分、mask、在线 softmax 和输出累积，只保存每行的少量统计量而不物化完整注意力矩阵。右侧时间线对应家族演进：V1 解决 IO，V2 提升并行效率，V3 进一步利用 Hopper 的异步执行和 FP8 能力。

### 2 FlashAttention (V1)：IO 感知的精确注意力

FlashAttention（论文发表于 2022 年）是一种 **IO 感知**（IO-aware）的精确注意力算法，核心思想是通过重组计算过程，充分利用 GPU 的内存层次结构，在数学上与标准注意力完全等价的前提下，将 HBM 访问次数从 \(O(N^2)\) 降至 \(O(N)\)。该算法不依赖于任何近似（如稀疏化或低秩分解），其输出与标准注意力实现严格一致。

#### 2.1 分块（Tiling）

FlashAttention 将 \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) 矩阵分割为若干**块**（tile），每个块的大小受 SRAM 容量限制。算法依次将单个 \(\mathbf{Q}\) 块和连续的 \(\mathbf{K}, \mathbf{V}\) 块加载到 SRAM 中，在片上完成该块对应的所有注意力计算，更新输出块 \(\mathbf{O}\)，然后释放 SRAM 空间以处理下一组块。**整个过程不显式物化完整的 \(N \times N\) 注意力矩阵**。

#### 2.2 在线 Softmax（Online Softmax）与归一化因子的传递

分块处理的困难在于 softmax 的全局归一化性质：位置 \(i\) 的输出需要整个第 \(i\) 行的所有注意力分数才能正确归一化。在分块方案中，无法一次性获得完整行，但该问题可被解决：对于向量 \(\mathbf{x} = [x_1, x_2, \dots, x_N]\)，其 softmax 的输出为 \(y_i = e^{x_i - m} / \ell\)，其中 \(m = \max_j x_j\)，\(\ell = \sum_{j=1}^N e^{x_j - m}\)。**在线 softmax** 允许按块处理时逐步更新这两个统计量。

设已有部分统计量 \(m^{(k)}\)（当前所见块的最大值）和 \(\ell^{(k)}\)（当前所见块以 \(m^{(k)}\) 为基准的指数和），处理下一块时出现新的局部最大值 \(m_{\text{new}}\) 和局部指数和 \(\ell_{\text{new}}\)，更新为

\[
m^{(k+1)} = \max(m^{(k)}, m_{\text{new}}), \quad \ell^{(k+1)} = e^{m^{(k)} - m^{(k+1)}} \ell^{(k)} + e^{m_{\text{new}} - m^{(k+1)}} \ell_{\text{new}}
\]

每当处理完一个 \(\mathbf{K}, \mathbf{V}\) 块时，FlashAttention 用当前累计的 \(m\) 和 \(\ell\) 对该块计算出的部分输出进行归一化并将累加到最终的输出块 \(\mathbf{O}\) 中，从而保证结果的数值精度与标准实现无异。

#### 2.3 内核融合（Kernel Fusion）

所有注意力操作——矩阵乘法 \(\mathbf{Q} \mathbf{K}^T\)、掩码（mask）、softmax、与 \(\mathbf{V}\) 的乘法——都被合并到一个单一的 CUDA 核（kernel） 中。与朴素实现需要多次 HBM 往返不同，融合内核对每个块仅从 HBM 加载一次 \(\mathbf{Q}\)、\(\mathbf{K}\)、\(\mathbf{V}\) 数据，在 SRAM 中执行全流程后将最终输出写回 HBM。这种设计显著减少了 HBM 读/写操作次数。

#### 2.4 反向传播

在标准注意力的反向传播中，softmax 输出矩阵 \(\mathbf{P}\) 必须被保留，其大小为 \(O(N^2)\)。FlashAttention 采用**重计算**（recomputation）策略：前向传播时只存储每个块的 softmax 归一化因子（大小为 \(O(N)\)），反向传播时重新读取 \(\mathbf{Q}\)、\(\mathbf{K}\)、\(\mathbf{V}\) 及中间统计量，在 SRAM 中重新计算注意力分值，从而避免了存储整个 \(N \times N\) 矩阵的显存开销。

#### 2.5 IO 复杂度与最优性

对于在两级内存层次（HBM 和 SRAM）上计算注意力的问题，FlashAttention 的 IO 复杂度为 \(O(N^2 d^2 / M)\)，其中 \(N\) 为序列长度，\(d\) 为头维度，\(M\) 为 SRAM 大小。这是该类问题在给定 SRAM 容量下的理论最优复杂度，即任何精确注意力算法至少需要完成此数量的 HBM 访问。

### 3 FlashAttention-2：从 IO 最优化到计算并行化

尽管 FlashAttention 已将内存复杂度降至 \(O(N)\)，但性能仍未达到 GPU 理论算力上限，在 A100 上仅能达到 25–40% 的理论最大 FLOPS/s。FlashAttention-2 识别出三个核心瓶颈：**非矩阵乘法操作过多**、**并行粒度过粗**及**线程束（warp）间通信开销大**，并针对性地逐一解决，在 A100 上实现了 50–73% 的理论 FLOPS 利用率。

#### 3.1 减少非矩阵乘运算

GPU 通过张量核心（Tensor Core）执行矩阵乘法可获得极高的吞吐量（FP16/BF16 达 312 TFLOPs/s），而一般标量运算（FP32）仅约 19.5 TFLOPs/s。**非矩阵乘运算虽然 FLOPS 占比不高，却因 GPU 低频标量计算单元的瓶颈而成为显著开销**。

FlashAttention-2 对算法进行了调整：将累积部分输出时的**按块归一化操作延迟到循环结束后的最终步骤**，从而在每一轮迭代中绕过对角缩放乘除，仅累积未归一化的中间结果，最终一次性完成归一化。两个版本的更新对比可形式化表达如下。

FlashAttention 每轮更新输出 \(\mathbf{O}^{(2)}\) 时需要以 \(\mathrm{diag}(\ell^{(1)} / \ell^{(2)})^{-1}\) 对先前累积的输出进行缩放：

\[
\mathbf{O}^{(2)} = \mathrm{diag}(\ell^{(1)} / \ell^{(2)})^{-1} \mathbf{O}^{(1)} + \mathrm{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}
\]

FlashAttention-2 改为在迭代中只累积原始值，将归一化延迟到最后：

\[
\tilde{\mathbf{O}}^{(2)} = \mathrm{diag}\left(e^{m^{(1)} - m^{(2)}}\right) \tilde{\mathbf{O}}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}
\]

即

\[
\tilde{\mathbf{O}}^{(2)} = e^{\mathbf{s}^{(1)} - m} \mathbf{V}^{(1)} + e^{\mathbf{s}^{(2)} - m} \mathbf{V}^{(2)}, \quad \text{最终 } \mathbf{O} = \mathrm{diag}(\ell^{(\text{last})})^{-1} \tilde{\mathbf{O}}^{(\text{last})}
\]

其中 \(m = m^{(\text{last})}\) 为整行的最终最大值。这种调整减少了每次迭代的额外浮点运算量，同时避免了冗余缩放所需的 HBM 访问。

#### 3.2 重设计并行化模式

FlashAttention 的并行策略为：启动 **批次数 × 头数** 个线程块（CTA，Cooperative Thread Array），每个 CTA 负责一个批次中的一个注意力头。当批次较小（常见场景）时，大量 SM（Streaming Multiprocessor）得不到充分利用。

FlashAttention-2 在此处**交换了内外循环的顺序**，将序列长度维度引入并行。外层循环迭代 \(\mathbf{K}\) 和 \(\mathbf{V}\) 的块，内层循环迭代 \(\mathbf{Q}\) 的块——这一交换使得不同线程块可以**同时处理同一注意力头中的不同 \(\mathbf{Q}\) 块**，即使批次大小很小，也能通过序列长度维度填充 GPU 的 SM 资源，大幅提升硬件占用率（occupancy）。在正向传播和反向传播中，线程块划分策略有所不同：正向传播按 **行** 划分，反向传播按 **列** 划分，这体现了对计算图结构的深入理解。

#### 3.3 优化线程束级工作划分

一个线程块内部包含多个线程束（warp，通常 32 线程）。FlashAttention V1 的线程束划分策略是**所有 warp 共享同一个 \(\mathbf{Q}\) 块，对 \(\mathbf{K}\) 按列划分**。该方案的缺点是计算输出 \(\mathbf{O}\) 时，每个 warp 输出的部分结果需要跨 warp 进行归约求和（reduce sum），产生额外的共享内存读写和 warp 间通信。

FlashAttention-2 改为：**每个 warp 加载完整的 \(\mathbf{K}, \mathbf{V}\) 块，但对 \(\mathbf{Q}\) 按行划分**。这样每个 warp 独立计算其负责的 \(\mathbf{Q}\) 子块所对应的输出部分，无需跨 warp 归约。这一调整显著减少了共享内存访问和 warp 间通信，使不同 warp 的计算能够完全并行且无竞争地执行。

#### 3.4 FlashAttention-2 性能提升

FlashAttention V1 在 A100 上只能达到理论峰值 FLOPS 的 25–40%。FlashAttention-2 将这一数字提升至 **50–73%**，端到端训练 GPT 类模型时可实现每 A100 GPU 225 TFLOPs/s（72% 模型 FLOPS 利用率），整体速度约为 V1 的 2 倍。

### 4 FlashAttention-3 简要前瞻

FlashAttention-3（2024 年发布）专门针对 **NVIDIA Hopper（H100）架构**进行了深度优化，主要引入了**异步执行**（asynchronous execution）和**FP8 低精度计算**的支持。该设计利用 TMA（Tensor Memory Accelerator）单元实现数据加载与计算的**重叠**，并通过**异步 wgmma （warp-wide matrix multiply-add）指令**掩盖内存访问延迟。FP8 FlashAttention-3 相比 FP16 基线实现了可观的吞吐量增长，数值误差控制在较低水平——比基线 FP8 注意力低 2.6 倍。从 V1 到 V3 的演进清晰反映了注意力计算优化的两个阶段：**V1 解决了 IO 瓶颈，V2 解决了并行效率瓶颈，V3 则面向新硬件架构进一步深挖**。

### 5 核心对比总结

| 维度 | FlashAttention | FlashAttention-2 |
|---|---|---|
| 核心瓶颈认知 | IO（内存带宽） | 非矩阵乘运算 + 并行效率 + warp 通信 |
| 内存复杂度 | \(O(N)\) | \(O(N)\)（不变） |
| 分块策略 | 融合内核 + 在线 softmax | 与 V1 相同，但交换循环顺序 |
| 并行模式 | 按 batch × head | 额外增加序列长度维度并行 |
| 线程束划分 | warp 共享 \(\mathbf{Q}\)，按 \(\mathbf{K}\) 划分 | warp 按 \(\mathbf{Q}\) 划分，共享 \(\mathbf{K}, \mathbf{V}\) |
| A100 FLOPS 利用率 | 25–40% | 50–73% |
| 相对速度 | 标准注意力的 2–4 倍 | V1 的约 2 倍 |

FlashAttention 家族从 IO 感知出发，逐步深入 GPU 并行调度的微架构层面，其设计演进体现了现代高性能深度学习内核开发的基本原则：**首先识别真实瓶颈（IO vs. compute），其次针对性设计数据流，最终将调度粒度匹配至 GPU 硬件层次（线程块 → 线程束 → 指令流）**。对于希望深入 LLM 内核优化的实践者而言，FlashAttention 不仅是算法创新，更是理解 GPU 内存层次与并行模型如何协同工作的绝佳案例。
