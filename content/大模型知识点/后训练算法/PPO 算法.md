---
created: '2026-04-20 01:11:23.185076+00:00'
order: 0
pinned: false
summary: 后训练经典强化学习算法
title: PPO 算法
updated: '2026-04-22 09:59:13.509230+00:00'
---

> PPO（Proximal Policy Optimization）是一种基于策略梯度的强化学习算法。要讲清楚 PPO，得先把它放回它的语境里：它是 TRPO 的工程化简化版。TRPO 强调"每次更新不要让新旧策略离得太远"，用显式的 KL 约束把它写成带约束优化；PPO 保留了同样的动机，只是把约束换成了一个可以直接塞进目标函数、用普通 Adam 一把梭的近似形式。现在大家说"PPO"，默认指的都是 **PPO-Clip** 这个版本——KL penalty 版虽然也在原论文里，但工程上没那么稳，用的人少。

![image.png](/media/images/uploads/image_1.png)

下面按这个顺序展开：先看 PPO 想解决的问题来自哪，再看它怎么解决的，最后看它在代码里到底长什么样、哪些细节不踩一遍坑就学不会。

## 1. 问题设定与策略梯度

RL 的基本设定是一个 MDP：状态 $s$，动作 $a$，策略 $\pi_\theta(a|s)$，目标是最大化期望折扣回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right]
$$

其中 $\tau=(s_0,a_0,r_0,\dots)$ 是一条轨迹，$\gamma\in(0,1]$ 是折扣因子。麻烦之处在于轨迹分布本身由策略决定——策略动一下，采样分布就跟着动，所以你不能像监督学习那样"固定数据分布算梯度"。策略梯度的做法是用 log-derivative trick，把梯度挪到策略上。

轨迹概率写开：

$$
p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)\,P(s_{t+1}|s_t,a_t)
$$

环境转移 $P$ 不含 $\theta$，所以 $\log p_\theta(\tau)$ 里跟 $\theta$ 有关的只剩策略项：

$$
\nabla_\theta\log p_\theta(\tau)=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

代回去：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,R(\tau)\right]
$$

这就是最原始的 REINFORCE。但 $R(\tau)$ 是整条轨迹的回报，方差大得离谱。利用"因果性"把它换成**从当前时刻起的未来回报** $G_t=\sum_{l\ge t}\gamma^{l-t}r_l$ 能小一些；再进一步用值函数替掉 $G_t$，就得到大多数教材写的那个形式：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,A^\pi(s_t,a_t)\right]
$$

这里用到了三个量：$Q^\pi(s,a)=\mathbb{E}[G_t\mid s_t=s,a_t=a]$，$V^\pi(s)=\mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$，和 **优势函数** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$。优势回答的是一个很直白的问题："这个动作比'平均水平'好多少？" $V$ 给出"在状态 $s$ 下按策略混的期望回报"，$Q$ 给出"非要在 $s$ 选 $a$，之后再按策略走"的期望回报。两者相减：正的说明这个动作在这个状态下超出预期，该把概率抬上去；负的说明低于预期，该把概率压下来。整个策略梯度在直觉上就是这么一回事。

方向有了，但要真的训练起来还有两个麻烦：**梯度方差大**、**步长不可控**。方差问题留给 GAE 去解决，步长问题就是 PPO 真正要攻的点。策略网络输出的是分布，参数只动一点点，某些动作的概率可能就从 0.01 跳到 0.3；再加上你通常想对同一批采样数据跑几轮 SGD（否则样本效率太低），几步下来新策略可能已经离采样策略很远，原来估的优势也就不再对应当前策略下的真实优势了。

## 2. 从 TRPO 到 PPO-Clip

TRPO 的想法很干净：不直接最大化真实回报，而是最大化一个"局部近似"——在旧策略 $\pi_{\theta_{\text{old}}}$ 附近成立的一阶 surrogate——再显式约束新旧策略不要差太多：

$$
\max_\theta\;\hat{\mathbb{E}}_t\big[r_t(\theta)\hat A_t\big]
\quad\text{s.t.}\quad
\hat{\mathbb{E}}_t\big[D_{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\big]\le\delta
$$

其中

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

是**重要性采样比率**，含义是"同一个动作，在新策略下概率是旧策略的多少倍"。你用旧策略的数据去估新策略的期望，这个比率就是 importance sampling 的修正因子。

TRPO 理论很漂亮，但实现很重：带 KL 约束的优化要算 Fisher 向量积、跑共轭梯度、做线搜索。PPO 的精神是"别较真那个硬约束，把它揉进目标函数里，用一阶 SGD 就能跑"。论文里给了两种揉法。

第一种是直接把 KL 当惩罚（**PPO-Penalty**）：

$$
L^{\text{KLPEN}}(\theta)=\hat{\mathbb{E}}_t\left[r_t(\theta)\hat A_t-\beta\,D_{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\right]
$$

$\beta$ 自适应——KL 太大就加大惩罚逼它回来，KL 太小就减小惩罚放开手脚。直观但 $\beta$ 不好调，实际工程上没 Clip 版本稳。

第二种，也是几乎所有实现的默认版本，**PPO-Clip**：

$$
L^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_t\left[\min\Big(r_t(\theta)\hat A_t,\;\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\Big)\right]
$$

$\epsilon$ 典型取 $0.1$ 或 $0.2$。这式子虽然短，但第一次看容易懵：为什么 clip？为什么取 min？得分两种情况看。

**优势为正的情形。** $\hat A_t>0$ 说明这个动作比平均好，直觉上希望增加它的概率，也就是希望 $r_t>1$。如果 $r_t$ 还在 $[1-\epsilon,1+\epsilon]$ 里，裁剪项等于原始项，目标是 $r_t\hat A_t$，正常按梯度走。如果 $r_t$ 已经爬过 $1+\epsilon$，说明这个好动作的概率已经被抬得够多了——裁剪项变成 $(1+\epsilon)\hat A_t$，是一个不再随 $r_t$ 变化的常数；取 $\min$ 就把目标卡在这个常数上，再继续抬高 $r_t$ 拿不到新的激励。**对好动作，你可以加概率，但不许一次加太狠**。

**优势为负的情形。** $\hat A_t<0$ 说明这个动作差，我们想减它的概率，即希望 $r_t<1$。同样地，区间内一切照旧；一旦 $r_t$ 跌到 $1-\epsilon$ 以下，裁剪项变成 $(1-\epsilon)\hat A_t$。因为 $\hat A_t$ 是负的，继续压低 $r_t$ 会让原始项 $r_t\hat A_t$ 变得更负（即目标变得更小），$\min$ 会选这个更小的值——目标反而被拽下去。**对坏动作，你可以压概率，但不许一步砍太狠**。

**为什么是 min 不是 max？** 因为 PPO 要的是一个"悲观"的替代目标：对每个样本都有

$$
\min(r_t\hat A_t,\operatorname{clip}(r_t)\hat A_t)\le r_t\hat A_t
$$

它永远不比未裁剪的 surrogate 更乐观。在"好方向上走得太远"时把目标封顶，这样多轮 mini-batch 更新时就不会被自己过度放大的比率往外拽。

这里有一个**特别容易被误解的点**必须强调：PPO 的 clip 约束的是"样本级比率对目标函数的贡献"，不是严格意义上的全局 KL 约束。它不保证新旧策略的真实 KL 一定小，也不像 TRPO 那样提供可证明的信赖域。它只是让"越过阈值后进一步优化的收益消失"，在优化动力层面把策略变化压住。正因为它不是硬约束，实际实现里**常常还会额外监控 KL**，超过阈值就 early stop 本轮更新。把 PPO 当成"有严格信赖域的 TRPO 廉价版"会误导你；它更像一个"足够好用的启发式"。

## 3. 把优势估出来：GAE

前面一直在用 $\hat A_t$，但没说它怎么来。直接用蒙特卡洛 $G_t-V(s_t)$，无偏但抖得厉害；用一步 TD 误差 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，方差小但偏差大（因为 $V$ 本身是学出来的，不准）。**GAE（Generalized Advantage Estimation）就是在这两者之间做几何加权折中**：

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

$\lambda\in[0,1]$ 控制你对未来 TD 误差的采纳深度。$\lambda=0$ 退化成一步 TD（偏差大方差小），$\lambda\to 1$ 近似蒙特卡洛（偏差小方差大），中间值是两者的折中。PPO 的经验默认是 $\gamma=0.99,\lambda=0.95$——不是什么定理，就是大量任务上稳定下来的经验值。

一个形象的类比：你要评价刚才那步棋走得好不好。

- 只看对手下一步反应（$\lambda=0$，一步 TD）：快但容易误判；
- 把整盘下完再回头看（$\lambda=1$，蒙特卡洛）：准但极抖；
- GAE 的做法是"近几步看清楚，远处的看个大概"，用 $\lambda$ 控制衰减节奏。

| $\lambda$ | 观察范围 | 偏差 | 方差 |
|---|---|---|---|
| 0 | 1 步 | 大 | 小 |
| 0.95 | 约 20 步（$0.95^{20}\approx 0.36$） | 中 | 中 |
| 1 | 到 episode 结尾 | 小 | 大 |

一个容易被忽略的细节：**$s_{t+1}$ 如果是真实终止状态，未来价值应该是 0**；如果只是因为 rollout 被截断或者达到 time limit，就应该 bootstrap，用 $V(s_{t+1})$ 接上。这个区别后面还会再提，它是 PPO 实现里最常见的 bug 源头之一。

critic 的监督信号通常就是 return target：

$$
\hat R_t=\hat A_t+V_{\phi_{\text{old}}}(s_t),\qquad L^{VF}(\phi)=\hat{\mathbb{E}}_t\big[(V_\phi(s_t)-\hat R_t)^2\big]
$$

注意 $\hat R_t$ 是 rollout 结束后算好、冻结住的目标，mini-batch 内部不会随 critic 参数更新重新计算。

## 4. 完整的损失函数（加上 critic 和熵）

把 actor、critic、熵放一起，PPO 最大化形式的总目标是：

$$
L^{\text{total}}(\theta,\phi)=\hat{\mathbb{E}}_t\left[L_t^{\text{CLIP}}(\theta)-c_1 L_t^{VF}(\phi)+c_2\, \mathcal{H}[\pi_\theta](s_t)\right]
$$

三项各司其职：$L^{\text{CLIP}}$ 管"往哪边改、改多狠"，$L^{VF}$ 让 critic 学对（critic 学不好 advantage 就是噪声），熵项 $\mathcal{H}$ 鼓励探索、防止策略过早塌缩成确定性动作。训练前期熵项能防止策略过早变确定，后期可以适当减小。PyTorch 实现里都是最小化，写成：

$$
\mathcal{L}=-\hat{\mathbb{E}}_t[L_t^{\text{CLIP}}]+c_1\hat{\mathbb{E}}_t[L_t^{VF}]-c_2\hat{\mathbb{E}}_t[\mathcal{H}]
$$

**符号最容易写反**。原论文是最大化目标，代码里是最小化负目标，不留神就训成了反方向。

**Value clipping**（可选但很常见）。OpenAI Baselines 和不少 RLHF 框架会对 value 也做一次裁剪：

$$
V_{\text{clip}}(s_t)=V_{\text{old}}(s_t)+\operatorname{clip}\big(V_\phi(s_t)-V_{\text{old}}(s_t),-\epsilon_v,\epsilon_v\big)
$$

$$
L_t^{VF}=\max\big((V_\phi(s_t)-\hat R_t)^2,(V_{\text{clip}}(s_t)-\hat R_t)^2\big)
$$

动机和 policy clip 一样：防止 critic 一次更新过猛。critic 剧烈波动会直接污染 advantage，advantage 再污染 actor，整个训练就失真了。这不是 PPO 理论的核心，纯粹工程稳态——有时候有帮助，有时候会让 value 学得太保守。

## 5. 训练流程、伪代码、以及"为什么还能多 epoch 复用数据"

![image.png](/media/images/uploads/image_2.png)

一轮 PPO 更新大致分四步。

**(1) 采样。** 用当前策略 $\pi_{\theta_{\text{old}}}$ 在 $N$ 个并行环境里跑 $T$ 步，总共 $NT$ 个 transition。对每步存下

$$
(s_t,a_t,r_t,\log\pi_{\theta_{\text{old}}}(a_t|s_t),V_{\phi_{\text{old}}}(s_t),\text{done}_t)
$$

**old log prob 和 old value 一定要存**，后面整轮更新它们都是冻结参考系。

**(2) 反向算 GAE。** 用最后一个状态的 value 做 bootstrap，从后往前扫：

$$
\delta_t=r_t+\gamma V(s_{t+1})m_t-V(s_t),\qquad \hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1}
$$

$m_t$ 是 mask，真实终止时为 0，否则为 1。算完 advantage 和 return 后通常做一次 advantage normalization（减均值除标准差）。这一步不是理论必须，但几乎所有实现都会做，因为不同 batch 的 advantage 尺度差距可能很大，不归一化 policy loss 会抖得离谱。严格讲减 batch 均值不是一个 state-only baseline，会引入轻微偏差，但实践中利远大于弊。

**(3) 多轮 mini-batch 更新。** 把整批数据打乱，分成 mini-batch，重复 $K$ 个 epoch。每次前向重新算当前策略的 log prob、entropy、value，在 log 空间里算 ratio——$r_t=\exp(\log p_t-\log p_t^{\text{old}})$ 比直接除概率稳得多——然后按 clip surrogate + value loss + entropy bonus 组合起来反传。

**(4) 扔掉这批数据，进入下一轮采样。**

写成伪代码就是：

```python
for update in range(num_updates):

    # 1. 用旧策略采样
    rollout_buffer = []
    for t in range(T):
        a_t, logp_old_t, v_old_t = policy.act(obs_t)
        next_obs, r_t, terminated, truncated, info = env.step(a_t)
        rollout_buffer.append(obs_t, a_t, r_t, logp_old_t, v_old_t,
                              terminated, truncated)
        obs_t = next_obs

    # 2. bootstrap + GAE（注意 terminated vs truncated）
    next_value = value(obs_t)
    advantages = zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if true_terminal(t):           # terminated=True
            nextnonterminal = 0
            nextvalues = 0
        else:                           # truncated 或未终止都要 bootstrap
            nextnonterminal = 1
            nextvalues = value_of_next_state(t)
        delta = r_t + gamma * nextvalues * nextnonterminal - v_old_t
        advantages[t] = lastgaelam = (
            delta + gamma * lam * nextnonterminal * lastgaelam
        )
    returns = advantages + values_old

    # 3. normalize
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. 多 epoch + minibatch 更新
    for epoch in range(K):
        for mb in shuffled_minibatches(rollout_buffer):
            logp, entropy, value = policy.evaluate(mb.obs, mb.act)
            ratio = exp(logp - mb.logp_old)   # 分母冻结!

            surr1 = ratio * mb.adv
            surr2 = clip(ratio, 1-eps, 1+eps) * mb.adv
            policy_loss = -mean(min(surr1, surr2))

            if value_clip:
                v_clipped = mb.v_old + clip(value - mb.v_old, -eps_v, eps_v)
                value_loss = 0.5 * mean(max((value - mb.ret)**2,
                                            (v_clipped - mb.ret)**2))
            else:
                value_loss = 0.5 * mean((value - mb.ret)**2)

            loss = policy_loss + vf_coef * value_loss - ent_coef * mean(entropy)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        if approx_kl > target_kl:   # 可选 early stop
            break
```

真正重要的不是会背这套代码，而是你要时刻清楚手里每一个张量到底是"当前策略算出来的"还是"旧策略存下来的"。一旦这两类张量混起来，PPO 的语义就崩了。

**最关键的细节：old_logprob 必须冻结。** ratio 的分母是采样时的旧策略概率，不是每个 SGD step 重新算一次。如果你在 epoch 内部把它也跟着更新，相当于"参考系在动"，clip 就彻底失去了意义——它不再约束"相对旧策略走多远"，而是在约束"相对上一次 mini-batch 走多远"，那是另一个东西。

**为什么 PPO 能多 epoch 复用同一批数据？** 这是它相对原始 on-policy 策略梯度最大的工程收益，原因有两条：重要性采样比率 $r_t$ 本来就在部分修正"新策略已不是采样策略"这个偏差；clip 又压住了这种修正能走多远。两者加起来让"用旧数据多做几步"变得可控。但它**没有**变成 off-policy——因为重要性比率只修正了动作分布的偏移，没有修正状态分布随策略改变的漂移，也没有真正的 replay buffer 做长期经验复用。所以 epoch 不能太大（一般 3~10），旧数据也不能复用太久，否则训练还是会失真。PPO 的样本效率一般不如 SAC、DQN 这类真 off-policy 方法，原因就在这。

## 6. 离散动作和连续动作怎么落地

**离散**最简单：网络输出 logits，softmax 成类别分布 $\pi_\theta(a|s)=\operatorname{softmax}(z_\theta(s))$，采样得到 action，log prob 就是对应类别的对数概率，熵 $\mathcal{H}=-\sum_a\pi(a|s)\log\pi(a|s)$。Atari 之类的任务基本都是这套。

**连续**通常用对角高斯：网络输出均值 $\mu_\theta(s)$，同时维护一个（可学习或状态相关的）标准差 $\sigma_\theta(s)$，动作分布 $\pi_\theta(a|s)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))$。log prob 是各维度**求和**：

$$
\log\pi_\theta(a|s)=-\frac12\sum_i\left[\left(\frac{a_i-\mu_i}{\sigma_i}\right)^2+2\log\sigma_i+\log(2\pi)\right]
$$

**是求和，不是求平均。** 这里第一次写 PPO 的人十有八九会踩坑——用 mean 的话 ratio 的尺度直接错一个数量级，训练完全跑不起来。

如果动作需要限制在 $[-1,1]$（比如 MuJoCo 里的很多任务），常见两种处理：

- **直接 clip**：高斯采完直接截到边界。省事，但采样分布和 log prob 对不上，不严谨。
- **Squashed Gaussian**：$u\sim\mathcal{N}(\mu,\sigma)$，$a=\tanh(u)$。这时 log prob 必须做 change-of-variables 修正：

$$
\log\pi(a|s)=\log\mathcal{N}(u;\mu,\sigma)-\sum_i\log(1-a_i^2+\varepsilon)
$$

不加这个修正项，ratio 和 entropy 都是错的，但表面上 loss 还是能下降，所以 bug 藏得很深。

## 7. 那些不踩一遍就学不会的工程细节

**`terminated` 和 `truncated` 的区别。** Gymnasium 现在把 done 拆成两个，这是 PPO 实现里最容易出 bug 的地方之一。`terminated=True` 是真终止（任务完成或失败），未来价值是 0，不能 bootstrap；`truncated=True` 是时间上限或 rollout 被切，episode 并没有真正结束，这时必须用 $V(s_{t+1})$ 接上。如果把所有 done 都当真终止，advantage 会被系统性低估；如果把真终止也拿去 bootstrap，value target 会被污染。很多 PPO 跑不起来的根源在这，而不是 clip。

**三种 normalization。** 几乎必做。尺度不稳定对 PPO 特别致命——advantage 抖让 policy loss 尺度抖；reward 太大让 value loss 压过 policy loss，critic 主导训练；observation 尺度不稳定让网络前向就开始飘。MuJoCo 一类连续控制任务尤其依赖 obs normalization；RLHF 里的 reward whitening 也是同一件事。

**Gradient clipping。** 和 ratio clipping 是两码事。ratio clipping 修改的是目标函数形状，gradient clipping 直接限制反传梯度范数 $\|\nabla_\theta\|_2\le c$（典型 0.5 或 1.0），防止偶发的坏 mini-batch 把参数炸飞。

**Learning rate annealing。** 很多实现让 LR 从初始值线性衰减到 0。早期大步探索，晚期小步收敛，经验上更稳。

**KL monitoring / early stopping。** clip 不是硬约束，所以实现里常额外估计近似 KL，太大就提前结束本轮更新。两种常见近似：

$$
\widehat{KL}\approx\tfrac1B\textstyle\sum_t(\log\pi_{\text{old}}-\log\pi_\theta),\qquad
\widehat{KL}\approx\tfrac1B\textstyle\sum_t(r_t-1-\log r_t)
$$

第二种基于 $k_3$ estimator，恒正且方差更小，很多实现偏好它。

**超参数的直觉。** 我不想把它写成一张默认值表，那没什么用。每个参数真正在调的是什么——

- **$\epsilon$（clip range）**：单次更新允许偏离旧策略多远。小了太保守学得慢，大了形同虚设。0.1~0.2 是常规区间，连续高维动作任务里往往需要更谨慎。
- **$\gamma$**：长期 vs 短期权重。多数任务 0.99 或 0.995。
- **$\lambda$**：GAE 的 bias-variance 旋钮。默认 0.95。advantage 太抖降一点，太短视升一点。
- **epoch 数**：同一批数据跑几轮更新。3~10 典型。太少没榨干，太多过拟合旧 batch。
- **minibatch size**：太小梯度噪声大、KL 抖；太大更新太"硬"，容易陷在保守区间。要和总 rollout size 一起设计。
- **entropy coef**：离散稀疏奖励任务常要大一点，连续控制后期常很小甚至为 0。
- **value coef**：经典 0.5。太大 critic 主导、policy 学不动；太小 critic 差、advantage 噪声大。
- **learning rate**：PPO 对它敏感。大了 ratio 爆、KL 飞；小了学不动。$3\times10^{-4}$ 是常见起点，但任务间差异很大。

**训练时该盯的日志。** reward 是滞后指标，不够用。真正重要的是：

- `approx_kl`：太大说明更新过猛，太小说明没在学（LR 太小或 clip 太紧）。
- `clip_fraction`：被 clip 到的样本比例。过高意味着大量样本已经撞到信赖边界，更新过猛；过低可能说明数据没被充分利用。
- `entropy`：掉得太快说明策略过早确定化、探索不足；长期太高说明学不动。
- `value_loss` 和 `explained_variance`。后者定义 $\text{EV}=1-\operatorname{Var}(y-\hat y)/\operatorname{Var}(y)$，$y$ 是 return target，$\hat y$ 是 value 预测。趋近 1 说明 critic 拟合好；长期接近 0 或为负说明 critic 基本没学到东西——这种情况下 actor 再怎么调都救不回来。

**最容易踩的坑，按出现频率大致排序：**

1. ratio 算错。正确写法是 $r_t=\exp(\log p_t-\log p_t^{\text{old}})$，log 空间做减法，不是直接两个概率张量相除。
2. 连续动作的 log prob 没有对维度求和，ratio 尺度直接错。
3. old logprob 没冻结，epoch 内部重新算。
4. advantages 跟着 critic 实时变，而不是 rollout 后固定。
5. truncation 当成真终止，没 bootstrap。
6. return target 没 detach，critic 目标反过来参与反传。
7. policy loss 或 entropy 项符号反了——PPO 代码里最常见的 bug 不是公式不会写，是正负号。
8. 有 clip 但没 KL 监控，某些任务上策略还是会偷偷漂走。
9. tanh squash 时忘了 log-det 修正。
10. reward 尺度太大，value loss 碾压 policy loss。

顺带说一句 actor 和 critic 的结构选择：两者可以共享 trunk 也可以完全分开。共享省算力，但 value loss 的梯度可能干扰 policy 的特征学习；分开更干净但更贵。小型任务里共享 trunk 很常见，RLHF 这种大模型场景里分头设计更多。需要记住的是 **critic 不是辅助模块**——它直接决定 advantage 质量，advantage 再直接决定 actor 梯度方向，critic 学坏 actor 基本也跟着坏。

## 8. PPO 为什么有效、它的局限，以及 RLHF 里的变形

如果要用一句话说 PPO 为什么 work：它把"往哪边改"、"方差多大"、"一步走多远"三件事分别交给了策略梯度、GAE、clip，然后在工程上把三者平衡得很稳。它没发明新的梯度方向，也没发明真正的信赖域——它只是把这三件事凑到了"用 Adam 一键训练就能跑得动"的程度。这种"工程平衡点"就是 PPO 的真正贡献。

局限也很清楚：它是 on-policy，样本效率一般；clip 只是启发式不是严格约束；对 reward 设计、critic 质量、observation 尺度都敏感；在非常高维、长时序、稀疏奖励、强延迟信用分配的问题上未必占优；超参数虽然比 TRPO 好调，但绝不是"随便设都能跑"。它是一个很好的 baseline，不是万能的。

最后说一下 **RLHF 里的 PPO**。骨架其实没变，只是对象换了：状态变成 token 前缀，动作变成下一个 token，策略就是语言模型本身（通常从 SFT checkpoint 初始化），value head 接在 LM 上预测每个位置的状态价值。奖励一般是 reward model 对完整回复打的分，再加一个对 reference model 的 KL 惩罚，避免策略偏离初始分布太远：

$$
r=r_{\text{RM}}-\beta\,D_{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})
$$

然后在 token 序列上做 token-level return、advantage、value learning。clip、value clip、GAE 这些核心组件基本都保留。

换句话说，RLHF-PPO 就是把"连续交互环境里的 trajectory"换成了"文本生成里的 token trajectory"，把"环境奖励"换成了"reward model 分数 + KL shaping"。只要标准 PPO 想清楚了，RLHF-PPO 基本就是换皮。