---
created: 2026-04-24 12:47:31.677138+00:00
order: 1
pinned: true
summary: ""
title: Flash Attention 家族
updated: 2026-05-01 01:02:19+08:00
---

## Flash Attention: 从 IO 感知到硬件极致利用

### 1 动机：注意力机制的真正瓶颈

Transformer 中的自注意力机制计算流程为

\[
\mathbf{S} = \mathbf{Q} \mathbf{K}^T, \quad \mathbf{P} = \text{softmax}\left( \frac{\mathbf{S}}{\sqrt{d_k}} \right), \quad \mathbf{O} = \mathbf{P} \mathbf{V}
\]

其中 \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}\)，\(N\) 为序列长度，\(d\) 为头维度。**朴素实现需要显式物化并存储 \(N \times N\) 的注意力矩阵 \(\mathbf{S}\) 和 \(\mathbf{P}\)**，导致时间与内存复杂度均为 \(O(N^2 d)\)。对于长序列（如 \(N=64\)K），\(N^2\) 约 40 亿，存储半精度矩阵即需约 8GB 显存，这成为 Transformer 扩展至长上下文的主要障碍。

---

### 2 FlashAttention (V1)：打通“内存墙”的 IO 感知精确注意力

在大语言模型的发展史中，FlashAttention 是一个具有分水岭意义的算法。它是一种**IO 感知**的精确注意力算法。

要理解 FlashAttention 的价值，首先要理解 GPU 的**内存层次结构**与**性能瓶颈**。现代 GPU（如 A100）的计算能力（FLOPS）极其强大，但其主存（HBM，高带宽内存）的读写速度却相对滞后。A100 的 HBM 带宽约为 1.5 TB/s 至 2 TB/s，而其片上缓存（SRAM）虽然容量极小（仅约 40 MB），但带宽却高达 19 TB/s。

标准注意力机制最大的痛点不在于 $O(N^2)$ 的计算量，而在于它需要频繁地将 $O(N^2)$ 级别的中间矩阵（如注意力分数矩阵 $\mathbf{S}$ 和概率矩阵 $\mathbf{P}$）写入 HBM 再读出。这导致模型在长序列下被死死卡在“内存墙”上（Memory-bound）。

朴素注意力实现将 \(\mathbf{S}\) 写入 HBM，读回计算 softmax，将 \(\mathbf{P}\) 重新写入 HBM，再读回与 \(\mathbf{V}\) 相乘。**大量 HBM 访问的实际开销远超浮点运算本身**，使注意力机制受限于内存带宽（memory-bound），而非计算能力（compute-bound）。在 GPU 持续增加的 FLOPS 与相对滞后的内存带宽之间，这一矛盾日益突出。

![paste_1777865733308.png](/static/images/uploads/paste_1777865733308.png)


FlashAttention 的核心思想是：**通过重组计算过程，将运算尽可能留在超快的 SRAM 中完成，在数学上与标准注意力完全等价的前提下，将 HBM 的访问次数从 $O(N^2)$ 降至 $O(N)$。** 它没有任何近似、不牺牲精度、不使用稀疏化，输出与标准注意力严格一致。

![paste_1777865693533.png](/static/images/uploads/paste_1777865693533.png)


#### 2.1 分块（Tiling）：化整为零的显存管理

标准注意力会一次性计算 $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$，生成完整的 $N \times N$ 矩阵并写入 HBM。当 $N=8192$ 时，单头单层的这个中间矩阵就会消耗大量显存和 IO 时间。

FlashAttention 引入了**分块**机制，将庞大的矩阵乘法拆解为能塞进 SRAM 的小块：
1. 算法根据 SRAM 的大小，确定一个块大小（Block Size，通常由硬件参数决定，设为 $B_c$ 和 $B_r$）。
2. 在外层循环中，依次将 $\mathbf{K}$ 和 $\mathbf{V}$ 的块（大小为 $B_c \times d$）从 HBM 加载到 SRAM。
3. 在内层循环中，将 $\mathbf{Q}$ 的块（大小为 $B_r \times d$）加载到 SRAM。
4. 在 SRAM 内部直接计算局部的 $\mathbf{Q}\mathbf{K}^T$、Softmax 并乘以 $\mathbf{V}$，计算完后立刻更新当前 $\mathbf{Q}$ 块对应的输出结果 $\mathbf{O}$，并写回 HBM。

**核心收益**：整个前向传播过程中，**从头到尾都不会在 HBM 中显式物化完整的 $N \times N$ 注意力矩阵**。

#### 2.2 在线 Softmax（Online Softmax）：打破全局依赖

分块计算最大的问题是 Softmax 操作。Softmax 是一个全局操作：**对于输入向量的第 $i$ 个元素，需要整行的所有元素来计算分母（指数和）以完成归一化**。既然我们每次只加载了一部分 $\mathbf{K}$，怎么在局部算出正确的 Softmax？

FlashAttention 利用了 **Online Softmax**，通过维护两个局部统计量来增量计算：**局部最大值 $m$** 和 **局部指数和 $\ell$**。

**【计算实例推导】**
假设某一个 Query 对 4 个 Key 的注意力原始得分（未归一化）向量为 $\mathbf{x} = [2, 1, 4, 3]$。
在标准 Safe Softmax 中，为了防止数值溢出，我们会先找到全局最大值 $m = 4$。
全局指数和 $\ell = e^{2-4} + e^{1-4} + e^{4-4} + e^{3-4} = e^{-2} + e^{-3} + 1 + e^{-1} \approx 1.553$。
最终 Softmax 输出为 $[\frac{e^{-2}}{1.553}, \frac{e^{-3}}{1.553}, \frac{1}{1.553}, \frac{e^{-1}}{1.553}]$。

现在，假设受 SRAM 大小限制，我们必须将 $\mathbf{x}$ 分解为两个块：块 A 为 $[2, 1]$，块 B 为 $[4, 3]$。FlashAttention 是这样处理的：

**步骤 1：处理块 A $[2, 1]$**
* 找到局部最大值：$m^{(1)} = \max(2, 1) = 2$
* 计算局部指数和：$\ell^{(1)} = e^{2-2} + e^{1-2} = 1 + e^{-1} \approx 1.368$
* （SRAM 内暂存当前的未完全归一化结果，等待后续更新）。

**步骤 2：处理块 B $[4, 3]$**
* 找到新块的最大值：$m_{\text{new}} = \max(4, 3) = 4$
* 计算新块的指数和：$\ell_{\text{new}} = e^{4-4} + e^{3-4} = 1 + e^{-1} \approx 1.368$

**步骤 3：Online Softmax 合并与修正**

我们现在看到了更多的数据，发现真正的最大值变了！旧的统计量 $m^{(1)}$ 和 $\ell^{(1)}$ 失效了，需要修正。
* **更新全局最大值**：$m^{(2)} = \max(m^{(1)}, m_{\text{new}}) = \max(2, 4) = 4$
* **计算修正因子**：旧块是以 $m=2$ 为基准算的，现在基准变成了 4，所以旧块的所有指数项都多乘了 $e^2$。我们需要给旧的指数和乘以修正因子 $e^{m^{(1)} - m^{(2)}} = e^{2-4} = e^{-2}$。
* **更新全局指数和**：
  $$ \ell^{(2)} = \ell^{(1)} \cdot e^{-2} + \ell_{\text{new}} \cdot e^{4-4} $$
  代入得：$\ell^{(2)} = (e^0 + e^{-1}) \cdot e^{-2} + (1 + e^{-1}) \cdot 1 = e^{-2} + e^{-3} + 1 + e^{-1}$。

你可以看到，经过分块修正后得出的 $\ell^{(2)}$，与一开始标准 Softmax 一次性算出的分母**在数学上完全等价**。依靠这种增量修正公式，FlashAttention 得以在局部块上完成精确的全局归一化。

#### 2.3 内核融合（Kernel Fusion）：榨干算力的最后一步

即使有了上述算法，如果在 PyTorch 中用标准算子拼装，依然无法加速，因为每个操作（如 `matmul`, `mask`, `softmax`）都会调用独立的 CUDA Kernel。

**【IO 往返对比】**
* **标准 PyTorch 实现**：
  1. 从 HBM 读 $\mathbf{Q}, \mathbf{K}$ -> 计算 $\mathbf{S}$ -> 将 $\mathbf{S}$ 写回 HBM。
  2. 从 HBM 读 $\mathbf{S}$ -> 计算 Softmax 得到 $\mathbf{P}$ -> 将 $\mathbf{P}$ 写回 HBM。
  3. 从 HBM 读 $\mathbf{P}, \mathbf{V}$ -> 计算 $\mathbf{O}$ -> 将 $\mathbf{O}$ 写回 HBM。
  这种频繁的 HBM 读写是性能的致命伤。
* **FlashAttention（算子融合）**：
  作者手动编写了一个定制的 CUDA 核。只从 HBM 中读取**一次** $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 的当前块进入 SRAM，在 SRAM 这个极速黑盒子里一口气完成乘法、Mask、Softmax 修正和乘以 $\mathbf{V}$ 的所有逻辑，最后只把算好的 $\mathbf{O}$ 块写回 HBM。中间没有任何废动作。

#### 2.4 反向传播与重计算（Recomputation）：用计算换内存

在标准反向传播中，为了计算梯度，框架必须在显存中缓存前向传播产生的巨大的 $N \times N$ 概率矩阵 $\mathbf{P}$。

**【内存开销对比实例】**
假设序列长度 $N = 8192$，注意力头数为 12，单头维度 $d = 128$，Batch Size 为 1。
* **标准注意力**：单层单头需要存储 $8192 \times 8192$ 个 float16，约 128 MB。12 个头就是 1.5 GB。这仅仅是一层保存中间激活值的开销！
* **FlashAttention**：前向传播完毕后，直接**丢弃** $N \times N$ 的中间矩阵，只在 HBM 中保留每个分块算出的两个标量统计量 $m$ 和 $\ell$。开销从 $O(N^2)$ 骤降至 $O(N)$，单头仅需约 32 KB。

在反向传播时，FlashAttention 重新从 HBM 读取 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$，利用存下来的 $m$ 和 $\ell$，在 SRAM 中以极快的速度**现场重新计算**出局部的 $\mathbf{P}$ 矩阵用于梯度计算。由于避免了读取巨大矩阵的 IO 耗时，这种“用重复计算换取内存带宽”的策略，不仅大幅降低了峰值显存，甚至让反向传播的速度变得更快。

#### 2.5 IO 复杂度与理论最优性

综合上述设计，FlashAttention 在两级内存层次架构下的 IO（HBM 访问量）复杂度被压缩至：
$$ O\left(\frac{N^2 d^2}{M}\right) $$
其中 $N$ 为序列长度，$d$ 为头维度，$M$ 为 SRAM 大小。

**为什么 $M$ 在分母上？**
因为 SRAM 的容量 $M$ 越大，我们能切分的块也就越大。每次加载一组 $\mathbf{K}, \mathbf{V}$ 块，就能和更大批量的 $\mathbf{Q}$ 完成计算，从而减少了从 HBM 整体加载数据的循环总次数。

论文证明了，在给定的 SRAM 容量下，这已经是计算精确注意力机制的**理论最优复杂度极限**。这就是 FlashAttention 能够从根本上提升大模型上下文窗口长度的底层数学与硬件逻辑。


如果说 FlashAttention V1 的历史使命是**打破“内存墙”（Memory Wall）**，把注意力计算的显存访问复杂度从 $O(N^2)$ 降到了 $O(N)$；那么 FlashAttention-2（发表于 2023 年）的使命就是**打破“计算墙”和“调度墙”**，彻底榨干 GPU 的极限算力。

在 V1 解决了 HBM 带宽瓶颈后，研究人员发现 V1 在 A100 上只能达到理论峰值 FLOPS 的 25%–40%。为什么还有这么大的性能差距？FlashAttention-2 通过深入 GPU 的底层执行逻辑（如张量核心、SM 调度、线程束通信），打出了一套极其漂亮的优化组合拳，将利用率拉升至 70% 以上。


---

### 3 FlashAttention-2：榨干 GPU 算力的极致系统级优化

FlashAttention V1 成功将计算限制在了 SRAM 中，但依然存在三个显著的性能痛点：
1. **非矩阵乘法（Non-MatMul）操作占比过高**：每次循环都在频繁做标量加减乘除。
2. **并行度不足（Low Occupancy）**：在小 Batch Size、长上下文场景下，GPU 没吃满。
3. **线程束（Warp）间频繁通信**：引入了不必要的共享内存读写和同步开销。

FlashAttention-2 针对这三点进行了重构。

#### 3.1 减少非矩阵乘运算：延迟归一化

**【硬件直觉：Tensor Cores 与 Scalar ALUs 的鸿沟】**
现代 GPU（如 A100）配备了专门为矩阵乘法设计的张量核心（Tensor Cores），其执行 FP16/BF16 矩阵乘法的吞吐量高达 312 TFLOPs/s。然而，GPU 内部的通用标量计算单元（用于执行普通的加减乘除、指数 $e^x$ 等）非常少，算力仅约 19.5 TFLOPs/s。
在 FlashAttention V1 中，非矩阵乘法（主要是 Softmax 相关的标量运算）虽然在总计算量中占比很小，却因为堵塞了缓慢的标量单元，拖慢了极其快速的张量核心，成了“木桶的最短板”。

**【V1 的冗余缩放问题】**
在 V1 的 Online Softmax 中，每处理一个新的 $\mathbf{K}, \mathbf{V}$ 块，都会得到新的局部最大值 $m^{(2)}$ 和局部指数和 $\ell^{(2)}$。为了保证暂存在 SRAM 中的局部输出 $\mathbf{O}^{(1)}$ 是始终合法的，V1 会**在每一步迭代中立刻对 $\mathbf{O}^{(1)}$ 进行重新缩放（Rescaling）**：
$$ \mathbf{O}^{(2)} = \mathrm{diag}(\ell^{(1)} / \ell^{(2)})^{-1} \mathbf{O}^{(1)} + \mathrm{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)} $$
这意味着每次内层循环，都要对一整个矩阵块进行大量的除法和乘法。

**【V2 的优化：先攒着，最后再除】**
FlashAttention-2 巧妙地调整了代数结构，**将最终的除法归一化延迟到循环的最后一步**。
在迭代过程中，我们不需要保持 $\mathbf{O}$ 是完全归一化的状态。我们只保留指数项 $e^{\text{score} - m}$ 乘以 $\mathbf{V}$ 的**未归一化累加值 $\tilde{\mathbf{O}}$**。

迭代时的更新公式简化为：
$$ \tilde{\mathbf{O}}^{(2)} = \mathrm{diag}\left(e^{m^{(1)} - m^{(2)}}\right) \tilde{\mathbf{O}}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)} $$
在这里，即使最大值 $m$ 更新了，我们也只需要给旧的 $\tilde{\mathbf{O}}^{(1)}$ 乘上一个衰减标量 $e^{m^{(1)} - m^{(2)}}$，然后直接累加新块的结果，**全程不需要做任何除法运算**。

等到所有的 $\mathbf{K}, \mathbf{V}$ 块全部遍历完毕，得到了整行的最终最大值 $m^{(\text{last})}$ 和最终指数和 $\ell^{(\text{last})}$ 后，再做**唯一一次**归一化：
$$ \mathbf{O} = \mathrm{diag}(\ell^{(\text{last})})^{-1} \tilde{\mathbf{O}}^{(\text{last})} $$
这一改动大幅削减了内层循环的标量运算量，让 Tensor Cores 能够连续工作。

这个问题抓得非常准。理解了这一步的代数变换，就真正触及了 FlashAttention-2 压榨硬件性能的本质。

我们可以用一句话来概括 FA2 的这个技巧：**把 Softmax 的“分子”和“分母”拆开，各自独立进行按块更新，直到最后一步才让它们相除。**

![paste_1777871636429.png](/static/images/uploads/paste_1777871636429.png)

##### 具体计算示例

为了彻底弄懂，我们用几个具体的数字过一遍。
假设某 Query 对 4 个 Token 的原始得分为 $S = [2, 5, 3, 8]$，对应的 Value 为 $V = [10, 20, 30, 40]$。
受内存限制，我们分为两个块计算：块 1 为前两个，块 2 为后两个。

**处理块 1：$S_1 = [2, 5], V_1 = [10, 20]$**

*   **当前最大值**：$m^{(1)} = \max(2, 5) = 5$
*   **计算未归一化的分子 $\tilde{\mathbf{O}}^{(1)}$**：
    $\tilde{O}^{(1)} = e^{2-5} \cdot 10 + e^{5-5} \cdot 20 = e^{-3} \cdot 10 + 20 \approx 0.5 + 20 = 20.5$
*   **计算分母 $\ell^{(1)}$**：
    $\ell^{(1)} = e^{2-5} + e^{5-5} = e^{-3} + 1 = 1.05$

*(如果是 FA1，此刻会做除法：临时输出 $O = 20.5 / 1.05 = 19.52$。但 FA2 **不做除法**，把 20.5 和 1.05 存着继续往下走。)*

**处理块 2：$S_2 = [3, 8], V_2 = [30, 40]$**

*   **遇到新块的最大值**：$m_{\text{new}} = \max(3, 8) = 8$
*   **更新全局最大值**：$m^{(2)} = \max(m^{(1)}, m_{\text{new}}) = \max(5, 8) = 8$
*   **计算衰减标量（极度重要）**：$e^{m^{(1)} - m^{(2)}} = e^{5-8} = e^{-3} \approx 0.05$
*   **计算新块自己的分子**：
    $\tilde{O}_{\text{new\_block}} = e^{3-8} \cdot 30 + e^{8-8} \cdot 40 = e^{-5} \cdot 30 + 40 \approx 0.2 + 40 = 40.2$
*   **合并分子 $\tilde{\mathbf{O}}^{(2)}$ (这就是你疑问的公式)**：
    旧分子乘衰减标量 + 新分子
    $\tilde{O}^{(2)} = (20.5 \cdot 0.05) + 40.2 = 1.025 + 40.2 = 41.225$
*   **同理合半分母 $\ell^{(2)}$**：
    $\ell_{\text{new\_block}} = e^{3-8} + e^{8-8} \approx 1.0067$
    $\ell^{(2)} = (1.05 \cdot 0.05) + 1.0067 \approx 1.0592$

**循环结束，最后一步：执行除法归一化**

- $$ O_{\text{final}} = \frac{\tilde{O}^{(2)}}{\ell^{(2)}} = \frac{41.225}{1.0592} \approx 38.92 $$

- 通过这个例子可以清晰看到：在内层循环遍历块 2 时，FA2 只做了一个标量乘法（乘以 $0.05$）和一个加法（加上 $40.2$）。由于移除了耗时的除法运算，GPU 内部最强大的张量核心（Tensor Cores）就不会被低频的标量计算单元（Scalar ALUs）卡住脖子，从而实现了极高的硬件利用率。

#### 3.2 重设计并行化模式：在序列维度上切分（Sequence-level Parallelism）

**【硬件直觉：如何填满 108 个 SM？】**
A100 GPU 拥有 108 个 SM（Streaming Multiprocessors）。为了让 GPU 发挥最大性能，你需要启动足够多的线程块（Thread Blocks / CTAs）来填满这些 SM。

FlashAttention V1 的并行粒度是：**Batch Size $\times$ Number of Heads**。
假设我们在做一个推理或微调任务，Batch Size = 1，注意力头数 = 12。那么 V1 只会启动 $1 \times 12 = 12$ 个线程块。这意味着 A100 上剩下的 96 个 SM 都在**完全闲置**！这在长序列（Long Context）场景下是灾难性的资源浪费。

**【V2 的优化：内外循环对调】**
V1 是外层循环遍历 $\mathbf{K}, \mathbf{V}$ 块，内层循环遍历 $\mathbf{Q}$ 块。
V2 **交换了内外循环的顺序**：外层循环遍历 $\mathbf{Q}$ 块，内层循环遍历 $\mathbf{K}, \mathbf{V}$ 块。

![paste_1777871889647.png](/static/images/uploads/paste_1777871889647.png)

为什么要换？因为输出结果 $\mathbf{O}$ 是按 $\mathbf{Q}$ 的维度来生成的。不同 $\mathbf{Q}$ 块对应的注意力输出是**完全独立**的！
通过将 $\mathbf{Q}$ 分块，V2 成功地将**序列长度（Sequence Length）维度**引入了并行。

现在，并行粒度变成了：**Batch Size $\times$ Number of Heads $\times$ (Sequence Length / Block Size)**。
即使 Batch Size = 1，只要序列够长，比如 8192，按块大小 128 切分，就会产生 $1 \times 12 \times (8192 / 128) = 768$ 个独立的任务块。这足以将 A100 的 108 个 SM 塞得满满当当，实现了极高的硬件占用率（Occupancy）。

*(注：在反向传播中，由于梯度的依赖关系不同，V2 采用的是按 $\mathbf{K}, \mathbf{V}$ 所在的列进行划分并行。)*

#### 3.3 优化线程束级工作划分：消除 Warp 间通信同步

**【硬件直觉：Shared Memory 的同步开销】**
在 GPU 中，一个线程块（CTA）由多个线程束（Warp，每个包含 32 个线程）组成。同一个块内的 Warp 共享同一块 SRAM（Shared Memory）。如果不同 Warp 计算了同一个结果的不同部分，它们就必须把部分结果写进 SRAM，调用 `__syncthreads()` 等待大家算完，然后再做 Reduce 求和。

**【V1 的“分工错误”】**
在 V1 中，对于当前加载到 SRAM 的 $\mathbf{Q}$ 块和 $\mathbf{K}, \mathbf{V}$ 块，**所有的 Warp 共享同一个 $\mathbf{Q}$ 块，但将 $\mathbf{K}, \mathbf{V}$ 块按列切分给不同的 Warp**。
这就导致：Warp 1 算了 $\mathbf{Q}$ 乘 $\mathbf{K}_1$，Warp 2 算了 $\mathbf{Q}$ 乘 $\mathbf{K}_2$。要得到最终的 $\mathbf{O}$，必须把 Warp 1 和 Warp 2 的结果在 Shared Memory 中加起来（Cross-warp Reduction Sum）。这种频繁的同步严重阻碍了并行效率。

**【V2 的解法：独立计算，互不干扰】**
FlashAttention-2 改变了 Warp 级的分工策略：**每个 Warp 加载完整的 $\mathbf{K}, \mathbf{V}$ 块，但将 $\mathbf{Q}$ 块按行（Row）切分给不同的 Warp**。
因为不同的 $\mathbf{Q}$ 行对应的输出 $\mathbf{O}$ 的行是完全独立的，这就意味着：
* Warp 1 专心算自己的 $\mathbf{Q}_{\text{row1}}$ 对应的 $\mathbf{O}_{\text{row1}}$。
* Warp 2 专心算自己的 $\mathbf{Q}_{\text{row2}}$ 对应的 $\mathbf{O}_{\text{row2}}$。

计算全程，**Warp 之间零通信，零同步（No cross-warp reduction）**。大家各自拿着完整的 $\mathbf{K}, \mathbf{V}$ 数据，疯狂做矩阵乘法，算完直接将自己的 $\mathbf{O}$ 写回。这种“无锁”设计极大提升了微观层面的执行效率。

#### 3.4 性能飞跃与里程碑意义

通过上述三项深度重构（算法级延后计算、Block 级提升并行度、Warp 级消除同步），FlashAttention-2 达成了惊人的性能指标：

* **硬件利用率飙升**：在 A100 上，模型 FLOPS 利用率（MFU）从 V1 的 25–40% 一跃升至 **50–73%**。在不需要处理因果掩码（Causal Mask）的场景下，甚至可以逼近理论极限。
* **端到端加速**：训练 GPT 类模型时，前向+反向的整体速度几乎是 V1 的 **2 倍**，单卡可达 225 TFLOPs/s。
* **长序列的奠基石**：正是得益于 V2 极高的计算效率，后来开源社区（如 LLaMA-2/3, Qwen 等）才能够将上下文窗口从 4K、8K 轻松推向 32K、128K 乃至更长，而不会面临难以承受的训练成本。

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
