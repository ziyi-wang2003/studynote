---
created: '2026-04-22 10:09:04.575885+00:00'
order: 1
pinned: false
summary: 人类偏好对齐
title: DPO 算法
updated: '2026-04-22 14:41:04.070109+00:00'
---

> DPO（Direct Preference Optimization）是 2023 年 Rafailov 等人提出的一种偏好学习算法。它的卖点可以用一句话概括：**把 RLHF 里"先训 reward model 再跑 PPO"这两步合成一步监督学习**。数学上它不是一个经验性技巧，而是一个等价变换——在 KL 正则化 RL 这个具体设定下，最优策略和 reward 之间有一个闭式关系，把它反代进 Bradley-Terry 偏好模型，RL 那一步就消失了，剩下一个长得像二分类交叉熵的 loss。理解 DPO 的难点不在代码，在推导：一旦看懂"reward 可以用策略反解出来"这一步，后面全是顺推。

这篇笔记的组织是：先交代它要替掉的东西（RLHF 里的 PPO 阶段），然后把整个推导走一遍，再说这个目标到底在优化什么、和 PPO 的差别在哪、工程上有哪些细节绕不开、典型的 failure mode 是什么样、后续的 IPO/KTO/SimPO/ORPO 又在修什么。

![image.png](/media/images/uploads/image_5.png)

## 1. 从 RLHF 到 DPO：要替掉的是什么

标准 RLHF 是三段式的：先在人类示范上做 SFT，然后用成对偏好数据 $(x, y_w, y_l)$ 训一个 reward model $r_\phi(x,y)$，最后用 PPO 最大化这个 RM 给的分数，同时约束策略不要离 SFT 太远。第三步的目标写成：

$$
\max_{\pi_\theta}\;\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\big[r_\phi(x,y)\big]-\beta\,D_{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
$$

$\pi_{\text{ref}}$ 通常就是 SFT checkpoint 的冻结副本，$\beta$ 控制"你能偏离 ref 多远"。

这一套跑起来挺重的。内存里同时要放四份模型（policy、ref、reward、value），训练要 on-policy 采样，reward model 本身可能 reward hacking，critic 学不好 advantage 就是噪声，PPO 超参也不少。对一个做 RLHF 的人来说，这里的工程负担远大于推理本身。

DPO 的切入点就是问一个很直接的问题：**既然最终目的是学出一个偏好最大化、KL 不爆的策略，能不能跳过 reward model 和 PPO，直接从偏好数据监督学这个策略？** 答案是能——而且推导简短得令人吃惊。

## 2. 核心推导：从 KL 正则化目标到偏好损失

推导分三步：先写出上面那个 RL 目标的闭式最优解，再把 reward 从最优解里反解出来，最后代进 Bradley-Terry 偏好模型。

### 2.1 KL 正则化 RL 的最优策略有闭式解

固定 $x$，目标就是在单个 prompt 上对 $\pi(\cdot|x)$ 做优化：

$$
\max_{\pi}\;\mathbb{E}_{y\sim\pi}\big[r(x,y)\big]-\beta\,D_{KL}\big(\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
$$

把 KL 展开，整个目标等价于最小化

$$
\mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\frac{1}{\beta}r(x,y)\right]
=\mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)\exp\!\big(r(x,y)/\beta\big)}\right]
$$

分母上那个东西除以一个归一化常数

$$
Z(x)=\sum_y \pi_{\text{ref}}(y|x)\exp\!\big(r(x,y)/\beta\big)
$$

就是一个合法概率分布。再把 $\log Z(x)$ 加进去（它不含 $\pi$，不影响优化），整个目标就变成 $\pi(\cdot|x)$ 和这个新分布之间的 KL。KL 最小值在两者相等时取到，于是：

$$
\boxed{\;\pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\!\big(r(x,y)/\beta\big)\;}
$$

这是一个很漂亮的闭式解，但它不是一个可计算的策略——$Z(x)$ 要对所有 $y$ 求和，对 LM 来说 $y$ 是一整段文本，这个和你根本算不出来。所以直接采样最优策略的路被堵死了，PPO 才要用 RL 迭代逼近。DPO 的招是：**我不去算 $\pi^*$，我只把它作为一个数学对象用在下一步里**。

### 2.2 从最优策略反解 reward

对上面那个式子两边取对数，整理一下就得到：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)
$$

这一步很关键。它说的是：**只要我们相信"真实 reward 对应的最优策略是 $\pi^*$"这件事，那么 reward 就可以用 $\pi^*$ 和 $\pi_{\text{ref}}$ 的对数比值表达出来**，只差一个和 $y$ 无关的项 $\beta\log Z(x)$。

到这里还没用到任何偏好数据，纯粹是 KL 正则化 RL 目标的数学结构。

### 2.3 代入 Bradley-Terry，$Z(x)$ 自己消掉

Bradley-Terry 是偏好建模的经典假设：对任意一对候选 $(y_w, y_l)$，人类标成 $y_w\succ y_l$ 的概率由它们的 reward 差决定——

$$
P(y_w\succ y_l\mid x)=\frac{\exp r(x,y_w)}{\exp r(x,y_w)+\exp r(x,y_l)}=\sigma\!\big(r(x,y_w)-r(x,y_l)\big)
$$

这就是训 reward model 时用的那个假设。现在把上一步的反解代进 $r(x,y_w)-r(x,y_l)$：

$$
r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

那个烦人的 $\beta\log Z(x)$ 在两个 $y$ 上是一样的，**相减时直接抵消**。这就是 DPO 能成立的数学关键。

现在把 $\pi^*$ 换成我们要学的 $\pi_\theta$，对整个偏好数据集做 MLE——极大化 $\prod P(y_w\succ y_l\mid x)$，也就是极小化负对数似然：

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]\;}
$$

看一眼这个式子：里面只剩下 $\pi_\theta$（要训练的策略）、$\pi_{\text{ref}}$（冻结参考）、$\beta$（一个标量超参）、和偏好数据。**reward model 没了，value function 没了，online 采样没了，整个 RL 循环没了**。梯度可以一阶反传，Adam 直接训。

## 3. 这个损失到底在干什么

DPO 的损失长得像二分类交叉熵，里面塞了一个"隐式 reward"。把它明确写出来：

$$
\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

这是 DPO 里最需要内化的概念。它说：**你的策略和参考策略的对数比值（放大 $\beta$ 倍）就是 DPO 内部"以为"的 reward**。训练过程在做的事情就是：让 $\hat r_\theta(x,y_w)>\hat r_\theta(x,y_l)$，差距越大损失越小，差距太大时 sigmoid 饱和、梯度自然衰减。

梯度写出来更直观。定义 $\Delta=\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$，那么：

$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=-\beta\,\mathbb{E}\Big[\sigma(-\Delta)\cdot\big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\big)\Big]
$$

这个式子里两件事值得停下来看。

**方向项** $\nabla\log\pi_\theta(y_w|x)-\nabla\log\pi_\theta(y_l|x)$ 就是在说"把赢家的概率推上去，把输家的概率拉下来"。和 pairwise learning-to-rank 一模一样。

**权重项** $\sigma(-\Delta)$ 是自动 curriculum。当模型已经把 $y_w$ 排得比 $y_l$ 高很多（$\Delta$ 大正数），这个权重接近 0，该样本几乎不再贡献梯度；当模型还没学明白甚至排反了（$\Delta$ 小或为负），权重接近 1，梯度全上。这意味着 DPO 不需要你手动做 hard example mining——已经学对的会自己退出，还没学对的会被持续推。

**$\beta$ 在干什么。** 它在 DPO 里同时控制两件事：一是 KL 约束强度（和 PPO 一样，大 $\beta$ = 更靠近 ref），二是 reward gap 的尺度（因为隐式 reward 里直接带着 $\beta$）。典型取 0.01~0.5，常见默认 0.1。$\beta$ 太大，$\hat r$ 对 log ratio 过于敏感，训练保守学不动；$\beta$ 太小，reward gap 轻易被放大，策略很容易偏离 ref 过远导致通用能力退化。

还有一件事要澄清一下：DPO 是在做 MLE，但**它的"数据生成过程"假设其实不是真的**——BT 模型假设偏好来自某个真实 reward 下的 sigmoid 概率，DPO 假设这个 reward 对应的最优策略正好是 $\pi_\theta$ 所能表达的某个分布。这两个假设在 LM 上都只是近似。所以 DPO 的理论保证是有条件的，把它当作"一个恰好好用、梯度自带 curriculum 的 pairwise 分类 loss"来理解会更稳。

## 4. DPO vs PPO：本质区别在哪

很容易把 DPO 理解成"PPO 的简化版"，这个类比能帮你入门，但深究下去会误导你。两者在底层机制上其实差得挺远。

**PPO 是 online 的。** 它要持续用当前策略采样新数据，reward 是 reward model 在这些新样本上打的分，然后用 importance sampling 和 clip 做稳定更新。策略分布和训练数据分布是"自动对齐"的——你改策略，下一批数据就跟着变。

**DPO 是 offline 的。** 训练数据是固定的偏好 pair 集合，整个训练过程策略会漂移，但数据不会跟着漂。这意味着你的 $\pi_\theta$ 最终是在"训练数据覆盖的那部分分布上"把 $\hat r_w>\hat r_l$ 学好了，但它在**自己采样出来的新分布上**到底表现如何，DPO 本身并不直接优化。

**PPO 学的是显式 reward。** reward model 是一个独立的模型，给任意 $(x,y)$ 打分，训练时策略的梯度方向由 RM 的梯度决定。

**DPO 学的是隐式 reward。** 它没有独立的 RM，"reward" 就是 $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，策略一变，reward 定义就跟着变。这既是它简洁的原因，也是它有些奇怪 failure mode 的原因（下一节会说）。

**KL 约束的位置不同。** PPO 里 KL 是个显式正则项（或者通过 clip 近似），训练时你能直接监控它、在它过大时 early stop。DPO 里的 KL 是"内嵌"在损失形式里的——那个 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 的减法本身就是一种 KL 结构，但训练时你不会像 PPO 那样一步步看着 KL 变化，也没有 clip 这种保险丝。

**计算代价上。** PPO 内存里要 policy + ref + reward + value 四份，DPO 只要 policy + ref 两份（ref 通常 frozen 且不反传，甚至可以 offload）。训练 step 上 PPO 一轮包含采样 + GAE + mini-batch SGD，DPO 一轮就是前向两个模型、算 log prob、BCE 反传，快得多。

一个简化的判断：**如果你的偏好数据 coverage 足够好、分布和最终想要的行为分布接近，DPO 几乎总是更划算**；**如果你需要在训练中让策略探索新的 $y$ 并持续修正方向、或者 reward model 的信号比静态偏好对更可靠，PPO 仍然有它的位置**。RLHF 社区现在倾向于 DPO 系列做第一轮对齐，PPO/GRPO 系列做后续更精细的优化（比如 reasoning RL）。

## 5. 实现上绕不开的几件事

**数据格式。** 最常见是 triplet $(x, y_w, y_l)$：一个 prompt 加一个被选中的回答和一个被拒绝的回答。偏好标注可以来自人类、也可以来自更强模型（所谓 AI feedback，RLAIF）。数据量通常几万到几十万对。

**如何算 $\log\pi_\theta(y|x)$。** 把 prompt 和 response 拼起来一次前向，拿到每个 response token 的 logits，softmax 之后取 ground truth token 的 log prob，**对 response 部分所有 token 求和**（prompt 部分不算 loss）。注意是 sum 不是 mean——如果你对不同长度的 response 做了 mean，相当于在 loss 里引入了长度归一化，训出来和标准 DPO 不等价。SimPO 正好是反过来，把 mean 作为卖点。

**reference model 的处理。** $\pi_{\text{ref}}$ 通常是 SFT 模型的一份冻结副本。实现上需要每个 batch 都对 $(x, y_w)$ 和 $(x, y_l)$ 跑 ref 前向拿到 $\log\pi_{\text{ref}}$。因为 ref 不反传，可以开 `torch.no_grad()` 省一半显存。工业界常见的节约内存做法有几种：ref 放到 CPU offload、ref 用低精度前向、或者 LoRA-DPO——policy 是 base + LoRA adapter，ref 是 base 本身（不加 adapter），这样两者共享绝大多数权重，算 ref 的代价等于临时把 adapter 关掉前向一次。

**prompt / response 的 tokenization 对齐。** 很容易出的 bug：prompt 末尾有没有空格、special token 怎么加、$y_w$ 和 $y_l$ 的 tokenization 是不是在同一个上下文下。任何一处不对齐都会让 log prob 算错、ratio 错位。写 DPO 前先写一个"给定 $(x,y)$ 输出 $\log\pi(y|x)$"的函数，单独拿几条样本验证它和手动算的一致，比后面 debug loss 省事。

**数值稳定性。** 实际代码里不会写 $\log\sigma(\cdot)$，会用 `F.logsigmoid`，它在 $\Delta$ 很负时保持数值稳定。另外两个 log ratio 的数量级有时很大（尤其是长序列），可以顺手把它们打印出来看看分布。

**初始化从哪来。** DPO 默认从 SFT checkpoint 开始，且 $\pi_{\text{ref}}$ 就是那个 SFT 本身。**千万不要**从随机初始化开始 DPO，那样 $\pi_{\text{ref}}$ 已经很强、$\pi_\theta$ 还是乱的，log ratio 爆炸，loss 直接发散。

## 6. DPO 的典型 failure mode

DPO 发表后的两年里，社区陆续发现了一批它特有的坑。这些不是实现 bug，是算法本身的结构性问题。理解它们比会写 loss 重要得多。

**(1) Chosen probability 反而下降。** 这是最著名也最反直觉的现象。你训完 DPO，查一下 $\pi_\theta(y_w|x)$ vs $\pi_{\text{ref}}(y_w|x)$，常常会发现**赢家的概率也下降了**——只是下降得比 $y_l$ 少。DPO loss 只约束二者的差（$\hat r_w > \hat r_l$），没约束绝对 likelihood。一条把 $y_l$ 压得很猛、顺便把 $y_w$ 也压一点的路径，loss 照样降。极端情况下整个策略分布向"什么都不想生成"的方向塌缩，生成质量反而变差。

实际上这和 DPO 的梯度方向是自洽的：梯度是 $\nabla\log\pi(y_w)-\nabla\log\pi(y_l)$，只要差值在往正确方向走，两项各自往哪边走并不被约束。社区的缓解方法有好几种：在 DPO loss 上加一个 SFT 项（在 $y_w$ 上做 NLL，防止它被压下去）、改用 IPO（下一节会讲）、或者换成 SimPO 这种不依赖 ref 的 formulation。

**(2) 分布偏移（off-policy 问题）。** DPO 是 offline 的，训练数据一旦采好就不再变。但训练过程中 $\pi_\theta$ 在移动，它自己采样出来的 $y$ 的分布和训练集里 $y_w, y_l$ 的分布会越来越不一样。数据集里 coverage 不到的那部分区域，DPO 完全没有信号——它可能在那些区域学出任意奇怪的行为。PPO 因为 online 采样自动处理了这个，DPO 没有类似机制。实践中的缓解是做多轮 iterative DPO：训一轮 DPO → 用当前策略采一批新数据 → 用更强的判别器（RM 或更强模型）重新打偏好标签 → 再训一轮。

**(3) 长度偏差。** $\log\pi(y|x)$ 是所有 token log prob 之和，长的 $y$ 天然绝对值更大。如果你的数据里 $y_w$ 系统性地比 $y_l$ 短（或者反过来），DPO 会把这个长度信号当成"偏好信号"学进去，训出来的模型要么啰嗦要么过短。SimPO 明确用 length-normalized log prob 解决这个；标准 DPO 里常见的做法是在数据准备阶段做长度平衡、或加显式长度惩罚。

**(4) 对标注质量极其敏感。** PPO 里 reward model 是个独立训练的模型，它对单条数据的噪声有一定的 averaging。DPO 没有这个中间层，偏好 pair 直接作用在策略梯度上。一对错标签 = 一条直接把策略往错方向推的梯度。在真实数据里，偏好标注本来就是高噪声的（人类一致性往往只有 60-70%），这让 DPO 对数据质量比 PPO 更敏感。

**(5) $\beta$ 不好调。** 前面说过它同时控制 KL 约束和 reward 尺度，两个作用耦合在一起。不同规模的模型、不同类型的偏好数据，最佳 $\beta$ 差别很大。0.1 是个常见起点，但不是万能值。

**(6) BT 假设本身的脆弱性。** Bradley-Terry 假设偏好来自一个潜在的标量 reward，且偏好概率是 reward 差的 sigmoid。真实人类偏好不完全符合这个——尤其是当偏好近乎确定时（比如 $y_w$ 明显好于 $y_l$，人类几乎总是选 $y_w$），BT 模型为了拟合这个接近 1 的概率会把 reward gap 推到无穷大，DPO 跟着就会让 $\hat r_w-\hat r_l$ 无限拉开，造成过拟合和数值问题。IPO 正是针对这个 failure 提出来的。

## 7. 主要变体：大家在修什么

DPO 出来之后涌现了一堆变体，基本都是在修上面某个 failure mode。挑几个最有代表性的讲。

**IPO（Identity Preference Optimization）**。Azar 等人 2023 年的工作，直接针对 BT 在确定性偏好下过拟合的问题。它把 DPO 的 $\log\sigma$ 形式换成一个 squared-loss regression：

$$
\mathcal{L}_{\text{IPO}}=\mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}-\frac{1}{2\beta}\right)^2\right]
$$

它明确要求 log ratio 的差等于一个固定的目标 $\frac{1}{2\beta}$，而不是越大越好。这避免了 DPO 在标签接近确定性时无限拉开 reward gap 的倾向。实际体验上 IPO 更稳，但超参感觉没 DPO 那么"友好"。

**KTO（Kahneman-Tversky Optimization）**。Ethayarajh 等人 2024 年的工作，出发点很不同：它不需要 pairwise 数据。你只需要给每条 $(x,y)$ 标一个 "desirable" 或 "undesirable" 的 binary 标签就行。损失函数借鉴了行为经济学的 prospect theory（对 gain 和 loss 的敏感度不对称），结构上有点像加了 reference point 的 sigmoid。对数据收集友好——很多场景下二元标签比 pairwise 偏好更容易标。

**SimPO（Simple Preference Optimization）**。Meng 等人 2024 年的工作，砍得最狠的一个：**完全去掉 $\pi_{\text{ref}}$**。损失形式是：

$$
\mathcal{L}_{\text{SimPO}}=-\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l|x)-\gamma\right)
$$

两个变化值得留意。一是用 response 长度做归一化（$/|y|$），直接把长度偏差治了；二是引入一个 target margin $\gamma$，类似于 hinge loss 里的 margin，防止 reward gap 无限扩大。没了 ref model，内存省一半，训练也快，很多基准上效果和 DPO 相当或更好。代价是它完全脱离了 KL 正则化的推导，理论上没有 DPO 那个优美的等价性保证。

**ORPO（Odds Ratio Preference Optimization）**。Hong 等人 2024 年的工作，把 SFT 和偏好学习合成一个 loss：SFT 项在 $y_w$ 上做 NLL（拉高赢家的概率），偏好项用 odds ratio 的 log sigmoid 形式做对比。同样不需要 ref model，而且不需要先做 SFT——一个 loss 从 base model 训到对齐完毕。实际中对小模型、数据量不大的场景友好。

这些变体没有"哪个绝对最好"的说法。它们在修的 failure mode 不同，适合的场景也不同。当前的经验大致是：偏好数据质量高、有足够 pairwise 数据时 DPO 或 IPO 依然稳；只有 binary 反馈用 KTO；在乎训练效率和长度控制用 SimPO；想一步到位（没有独立 SFT 阶段）用 ORPO。

## 8. 把 DPO 压回到一句话

DPO 的贡献是展示了一件看似不可能的事：在 KL 正则化 RL 这个具体设定里，你其实不用真的去做 RL。从最优策略的闭式解反推回去，reward model 和偏好学习可以折叠成一个单一的监督分类目标，normalization constant 在取差时神奇地消掉。这个数学等价性把 RLHF 的工程成本从"四个模型 + 一个 RL 循环"降到了"两个模型 + 一个 BCE-like loss"。

但等价性是有代价的。你继承了 Bradley-Terry 的所有假设缺陷，你失去了 PPO 那种 online 修正分布偏移的能力，你获得了一个只约束 reward 差、不约束绝对 likelihood 的目标——于是就有了 chosen 概率反而下降、分布塌缩、长度偏差这些奇怪的 failure mode。DPO 不是 PPO 的免费替代，它是一个在"数据足够好、分布足够对齐"时非常漂亮的捷径，在数据不够好时同样会暴露出它独有的脆弱性。理解 DPO 的真正收获不是记住那个损失函数——那个损失函数一眼就记住了——而是理解"reward 可以用 $\beta\log\frac{\pi}{\pi_{\text{ref}}}$ 隐式表示"这件事本身意味着什么：一旦采纳这个视角，reward model 只是显式化了一个策略本来就在编码的信息，而 RLHF 的本质可以被重新表述为"直接在策略空间里做偏好学习"。后面 SimPO 连 ref 都敢砍掉，ORPO 敢把 SFT 合进来，其实都站在这个视角上。