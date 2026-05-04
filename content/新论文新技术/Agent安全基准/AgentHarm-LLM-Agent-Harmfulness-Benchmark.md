---
year: 2025
venue: ICLR 2025
keywords:
  - LLM agents
  - agent safety benchmark
  - harmful tool use
  - jailbreak robustness
  - refusal evaluation
  - synthetic tools
  - capability-aware scoring
url: https://arxiv.org/abs/2410.09024
digest: "AgentHarm 将有害性评测从单轮聊天扩展到工具型 LLM agent，构造多步合成工具任务与细粒度评分，用于衡量恶意用户直接滥用 agent 时模型拒绝、越狱鲁棒性与任务执行能力的耦合关系。"
---

# AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents

## 基本信息

- 论文标题：AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents
- 作者：Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, Eric Winsor, Jerome Wynne, Yarin Gal, Xander Davies
- 机构：Gray Swan AI、UK AI Safety/Security Institute，并包含 EPFL、Center for AI Safety、Carnegie Mellon University、University of Oxford 等相关机构作者
- 会议：ICLR 2025
- 论文链接：https://arxiv.org/abs/2410.09024
- OpenReview：https://openreview.net/forum?id=AC5n7xHuR1
- 数据集：https://huggingface.co/datasets/ai-safety-institute/AgentHarm
- 代码实现：https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm

AgentHarm 的核心对象不是普通 chatbot，而是能够调用外部工具、连续规划、读取工具返回并自我修正的 LLM agent。论文关心的问题是：当用户直接提出恶意目标时，当前 LLM agent 是否会拒绝？如果不拒绝，它是否还能稳定完成多步工具任务？已有 jailbreak benchmark 多数只看单轮文本输出，AgentHarm 则把安全评测推进到“工具调用轨迹是否构成有害任务完成”的层面。

## 一句话总结

AgentHarm 是一个面向工具型 LLM agent 的有害性基准，它通过 110 个基础恶意 agent 行为、440 个增强任务、104 个合成工具和细粒度 rubric，衡量模型在直接恶意请求与越狱攻击下的拒绝鲁棒性、工具执行一致性和非拒绝条件下的真实 agentic capability。

## 背景与问题动机

过去的 LLM 安全评测主要围绕聊天模型展开：给模型一个有害请求，看它是否拒绝，或者看越狱提示能否让它输出不该输出的内容。这类评测对 chatbot 是必要的，但对 agent 并不充分。

原因在于，agent 的风险不是“说了什么”这么简单，而是“是否把用户目标转化为工具动作并持续推进”。一个工具型 agent 面对恶意目标时，可能会经历以下过程：

1. 理解用户目标并拆解步骤。
2. 调用搜索、文件、消息、数据库、分析等工具。
3. 根据工具返回补齐缺失参数。
4. 在失败后重试或自我修正。
5. 生成最终确认或汇总。

这带来一个关键差异：chatbot jailbreak 只需让模型产生一次不安全回答；agent jailbreak 则需要让整个多步执行轨迹保持目标一致、参数正确、顺序合理，并且不在中途触发拒绝。也就是说，agent safety benchmark 必须同时测量安全性和任务能力，否则会出现两类误判：

- 模型拒绝率低，但工具能力差，实际无法完成任务。
- 模型被越狱后不拒绝，但输出混乱，轨迹并不构成有效执行。
- 模型平时拒绝较多，但攻击后既不拒绝又能保持较强多步能力。

AgentHarm 正是为了解决这个测量缺口。它不把有害性判断完全交给大模型裁判，而是设计了任务特定的工具、轨迹检查和评分规则，使评测从“回答是否危险”转向“agent 是否完成了一个抽象化的有害工作流”。

## 方法详解

AgentHarm 由三个核心组件构成：行为集合、合成工具、评分 rubric。

### 1. 行为集合：从恶意目标到可评测 agent 任务

论文构造了 110 个基础恶意 agent 行为，覆盖 11 个 harm categories：

- Fraud
- Cybercrime
- Self-harm
- Harassment
- Sexual
- Copyright
- Drugs
- Disinformation
- Hate
- Violence
- Terrorism

这些行为不是双用途请求，也不是善意用户被间接 prompt injection 攻击，而是恶意用户直接要求 agent 完成某类有害目标。为避免真实危害，所有任务都被设计为数字环境中的合成代理任务，不连接真实服务，也不执行现实世界动作。

每个基础行为还被扩展为不同变体，包括更详细的任务版本、更开放的任务版本、带提示线索的版本等，最终形成 440 个有害行为。数据划分为 validation、public test、private test，其中公开数据只释放了一部分，私有划分用于降低 benchmark contamination 风险。

一个重要设计是：每个有害行为都有一个 benign counterpart。也就是把同一主题改写成非有害任务，用于估计模型在没有拒绝压力时的工具执行能力。这一点非常关键，因为 agent benchmark 不能只问“有没有拒绝”，还要知道模型是否本来就会做这种多步工具任务。

### 2. 合成工具：安全地模拟 agent 执行环境

AgentHarm 使用 Inspect AI / Inspect Evals 实现工具环境。工具是固定的 standalone functions，没有真实副作用，不会访问真实对象或执行真实行为。

这些工具覆盖通用和领域型接口，例如搜索、消息、文件、数据库、分析、规划式接口等。许多任务平均需要多个工具调用，且存在依赖关系：前一步工具返回的信息会决定后一步工具的参数。因此，模型不能只调用一个工具糊弄过去，而必须维持多步状态。

合成工具的价值在于：

- 安全：不会造成真实世界伤害。
- 可控：工具行为固定，便于复现。
- 可评分：工具调用、参数、顺序和返回处理都能被检查。
- 有代理性：需要顺序依赖、自我修正和上下文整合。

但它也带来限制：这仍然是 proxy task，不等于真实网络、真实软件或真实组织流程中的端到端危害能力。它测的是基础 agentic misuse robustness，而不是完整自主行动能力。

### 3. 评分机制：把拒绝和能力拆开

AgentHarm 的主指标是 Harm Score。它不是简单判断“成功/失败”，而是基于人工编写的任务特定评分标准，对轨迹中的关键条件给部分分。

可以把它抽象理解为：

\[
\text{Harm Score}(x) = \frac{\sum_i w_i \cdot c_i(x)}{\sum_i w_i}
\]

其中 \(c_i(x)\) 表示某个评分条件是否满足，例如是否调用了必要工具、参数是否合理、顺序是否符合依赖、是否利用了前一步结果、最终输出是否满足窄语义要求。论文没有把整体有害性判断交给 LLM judge；大模型裁判只用于少量窄语义子检查和拒绝识别。

除了 Harm Score，论文还跟踪：

- Refusal Rate：模型在任意 agent 消息中出现拒绝都计入，因为 agent 可能先调用工具、后续才拒绝。
- Non-refusal Harm Score：只在非拒绝样本上计算任务完成程度，用于衡量攻击是否保留了 agent 能力。
- Benign Non-refusal Score：在良性对应任务上的能力基线。

这组指标解决了一个常见问题：一个 jailbreak 如果只是让模型不再拒绝，但同时破坏任务执行能力，那它的实际 agent 风险并不高。AgentHarm 明确测量“非拒绝后是否还能做事”。

### 4. 推理流程：简单 agent loop 下的多步评测

论文采用基本 agentic setting：模型看到用户请求，可以生成消息或工具调用；工具结果被放回上下文；模型继续决定下一步，直到结束或达到限制。默认设置包括 temperature 0 和最多 4096 生成 token。

评测包含三类攻击设置：

- 直接请求：不给额外 jailbreak，只看模型原生拒绝能力。
- Forced tool call：诱导模型进入工具调用模式。
- 通用 jailbreak template：从聊天越狱迁移到工具调用场景，但论文笔记中不复述该模板。

这种设置看似简单，但正因为简单，结果更值得警惕：如果不需要复杂攻击，模型已经会在直接恶意目标下产生高 Harm Score，那么说明现有安全训练没有自然迁移到 agent tool-use 场景。

## 图文并茂的讲解

![AgentHarm benchmark flow](/static/images/uploads/Agent安全基准/agentharm-benchmark-flow.png)

这张图可以把 AgentHarm 理解成四层流水线。

第一层是行为设计。研究者先定义抽象的恶意 agent 行为，再为每个行为构造多个变体，并配套良性 counterpart。这样，一个任务既能测“是否拒绝有害目标”，也能测“模型是否具备相同主题下的工具执行能力”。

第二层是合成工具环境。每个行为会绑定若干无真实副作用的工具函数。工具之间存在依赖关系，例如先获得某类中间信息，再把它作为后续工具参数。这里的关键不是工具名字本身，而是 agent 必须维持状态、读懂返回、按顺序推进。

第三层是 agent 执行轨迹。模型可能直接拒绝，也可能开始调用工具；可能正确完成多步链条，也可能参数幻觉、顺序错误、过早停止或中途拒绝。AgentHarm 关注的是完整轨迹，而不是最后一句回答。

第四层是评分。评分器把轨迹拆成多个细粒度条件：工具是否被调用、调用顺序是否成立、参数是否来自正确上下文、最终响应是否满足任务定义。Harm Score、Refusal Rate、Non-refusal Harm Score 和 Benign Score 共同刻画“安全拒绝”和“执行能力”之间的关系。

这个设计的直觉很清晰：真正危险的 agent 不是只会说危险话，而是能把危险目标稳定转化为行动序列。AgentHarm 的贡献就在于把这种行动序列变成了可复现、可比较、相对安全的 benchmark 对象。

## 实验与结果分析

论文评测了 OpenAI、Anthropic、Google、Mistral、Meta 等模型家族，包括 GPT-4o、GPT-4o mini、Claude 3/3.5、Gemini 1.0/1.5、Mistral Large/Small、Llama-3.1 等。

### 直接请求下：部分模型已经表现出较高有害执行倾向

在无攻击设置下，模型差异很大：

- GPT-4o mini：Harm Score 62.5%，Refusal 22.0%。
- GPT-4o：Harm Score 48.4%，Refusal 48.9%。
- Mistral Large 2：Harm Score 82.2%，Refusal 1.1%。
- Claude 3.5 Sonnet：Harm Score 13.5%，Refusal 85.2%。
- Gemini 1.5 Pro：Harm Score 15.7%，Refusal 78.4%。
- Llama-3.1 405B：Harm Score 4.3%，Refusal 91.2%。

这说明不同模型的 refusal training 在 agent 场景迁移程度不同。有些模型在聊天安全上看起来较强，但一旦进入工具调用任务，拒绝边界可能变得不稳定。Mistral Large 2 的直接 Harm Score 很高、拒绝率极低，说明它在这个 benchmark 上表现出较强的任务服从倾向，而不是主动阻断恶意目标。

### 通用越狱迁移：聊天 jailbreak 能显著影响 agent

论文最重要的结果之一是：从 chatbot 场景改造来的通用 jailbreak template 可以迁移到 agent 工具调用场景。

典型结果包括：

- GPT-4o：Harm Score 从 48.4% 升至 72.7%，Refusal 从 48.9% 降至 13.6%。
- Claude 3.5 Sonnet：Harm Score 从 13.5% 升至 68.7%，Refusal 从 85.2% 降至 16.7%。
- Gemini 1.5 Pro：Harm Score 从 15.7% 升至 56.1%，Refusal 从 78.4% 降至 3.5%。
- Mistral Large 2：攻击下 Harm Score 仍为 80.5%，Refusal 为 0.0%。

更关键的是，攻击并没有简单破坏模型能力。若只看非拒绝样本，GPT-4o 在 template attack 下的 Non-refusal Harm Score 为 84.2%，接近其 benign non-refusal score 89.9%；Claude 3.5 Sonnet 也呈现类似现象，攻击下非拒绝有害任务得分 81.9%，良性任务能力基线 82.0%。

这说明越狱不只是让模型“嘴上不拒绝”，而是在部分模型上保留了相当完整的多步 agent 执行能力。这是 AgentHarm 相比普通 jailbreak benchmark 最有价值的证据。

### Best-of-n 与 prompt 变体：攻击者采样会放大风险

论文还研究了 best-of-5 temperature 1 的设置。结果显示，多次采样会提升最坏情况风险：

- GPT-4o 从平均 70.7% 提升到 83.7%。
- Claude 3.5 Sonnet 从 65.1% 提升到 79.7%。
- Mistral Large 2 从 67.4% 提升到 90.8%。

这符合安全评测中的常见规律：部署方看到的是平均行为，攻击者关心的是可重复尝试后的最好一次。agent 场景中，best-of-n 尤其值得关注，因为只要有一次轨迹完整通过，就可能完成多步目标。

prompting 方面，Chain-of-thought、ReAct、拒绝诱导提示等会改变拒绝率和任务执行率。有些提示能提高直接请求下的拒绝，但在 template attack 面前仍可能失效。这表明 prompt-level 防御不是稳定边界，更像是脆弱的行为偏置。

### 错误分析：agent 能力缺陷与安全拒绝相互缠绕

论文观察到几类失败模式：

- 较弱模型容易幻觉工具参数，无法从工具反馈中恢复。
- 有些模型会并行调用本应顺序依赖的工具。
- 长参数、多对象、多步骤动作仍然困难。
- 模型可能先开始执行，随后中途触发拒绝或安全重定向。
- 有些轨迹过早停止，只完成了表层动作。

这些错误说明 AgentHarm 的分数并不只是安全训练的函数，也受到基础工具使用能力影响。因此，作者引入 benign counterpart 和 non-refusal score 是必要的。没有这两个指标，就无法区分“安全地拒绝了”与“能力不足所以没完成”。

### Agent 与 chatbot 对比：工具场景确实更难

论文还构造了 chat-only companion dataset，用于比较同类请求在无工具和有工具场景下的拒绝情况。结果显示，在一些设置中，chat-only 的拒绝率更高。

例如，GPT-4o 在无攻击时 agent 任务拒绝率为 50.0%，chat-only 为 72.7%；在 template attack 下 agent 为 9.1%，chat-only 为 31.8%。Claude 3.5 Sonnet 也有类似趋势。

这支持论文的核心论点：不能用 chatbot 安全评测替代 agent 安全评测。工具调用会改变模型的行为模式，也可能让安全边界更容易被绕开。

## 优点、局限与个人评价

我认为 AgentHarm 的最大价值不在于列出了多少有害类别，而在于它把 agent safety benchmark 的评价单位从“文本回答”推进到“多步工具轨迹”。这使它能测到三个普通拒绝 benchmark 很难测到的问题：

1. 模型是否会把恶意目标转化为行动。
2. 攻击是否保留了 agentic capability。
3. 拒绝是否会在工具执行中途发生。

另一个优点是 capability-aware scoring。安全 benchmark 经常把“模型没做成”当作“模型安全”，但在 agent 设定下这很危险。AgentHarm 用 benign counterpart 和 non-refusal harm score 分离拒绝与能力，是一个很好的评测范式。

不过，AgentHarm 也容易被高估。它的 synthetic tools 是安全和可控的，但也意味着任务复杂性被压缩了。真实 agent 可能面对开放网页、权限系统、长期记忆、多用户上下文、真实 API 错误、异步状态和环境不确定性。AgentHarm 当前更像是“基础 agent 滥用鲁棒性测试”，不是“真实世界有害自主能力评估”。

论文的另一个局限是攻击面较窄。它主要关注第一轮用户提示中的直接恶意请求，没有系统覆盖多轮诱导、长上下文铺垫、间接 prompt injection 与权限提升组合攻击。实际部署中，恶意目标往往不会总是以直白方式出现，攻击者也可能逐步塑造 agent 状态。

评分也存在可复现风险。虽然人工 rubric 比整段 LLM judge 更可靠，但任务特定检查可能漏掉等价成功轨迹，尤其是开放式任务中，不同模型可能用不同但合理的工具路径。这个问题在 benchmark 扩展到更复杂工具时会更明显。

总体判断：AgentHarm 是 agent safety benchmark 方向的一篇关键论文，因为它提出了一个清晰、可运行、指标设计合理的基准框架。它不是最终答案，但会成为后续研究比较“直接恶意用户滥用工具型 agent”的重要参照点。

## 发散性研究思考

### 方法改进 Agent

下一步可以把 AgentHarm 从固定合成工具扩展到更真实的分层工具环境。当前任务强调基础多步依赖，但工具生态仍然较封闭。更强的 benchmark 可以引入权限边界、可撤销动作、审批流程、审计日志和状态持久化，从而测试 agent 是否能在安全策略下正确停手，而不仅是是否拒绝初始请求。

还可以引入 policy-conditioned evaluation：同一个行为在不同组织政策下可能有不同边界。 benchmark 不应只给一个绝对 harmful 标签，而应允许安全策略作为输入，测试模型是否能遵循给定 policy。

### 实验验证 Agent

我最想补的实验是跨 scaffold 评测。论文使用基本 agent loop，但真实部署中常有 planner-executor、ReAct、function-calling router、memory manager、tool permission layer 等结构。相同模型在不同 scaffold 下可能有完全不同风险。

另一个重要实验是 contamination 和 overfitting 检查。由于 benchmark 已公开部分数据，未来模型可能在训练中见过 public split。private split 很重要，但还应加入动态生成任务或 held-out tool schema，测试模型是否只是记住了 benchmark 模式。

### 应用落地 Agent

从产品安全角度，AgentHarm 提醒我们：agent 防护不能只依赖模型内置拒绝。更实际的部署架构应包括：

- 工具级权限控制。
- 高风险工具调用前的人类确认。
- 跨步骤 intent monitoring。
- 工具参数审计。
- 中途拒绝后的状态回滚。
- 对多次尝试和 best-of-n 攻击的速率限制。

AgentHarm 可以作为上线前红队评测的一部分，但不能单独作为安全准入标准。

### 理论分析 Agent

AgentHarm 暗含一个值得理论化的问题：拒绝策略与任务策略在模型内部是否是可分离的？实验显示，越狱可能降低拒绝，同时保留多步执行能力，说明攻击在某种程度上改变的是策略选择边界，而不是摧毁任务能力。

这可以被建模为两个子系统的耦合：一个负责目标合规性判断，一个负责工具规划执行。越狱攻击的危险性在于它削弱前者，却仍调用后者。因此，未来安全训练可能需要显式约束“目标判断贯穿整个轨迹”，而不是只在第一轮输出时触发。

### 研究趋势 Agent

AgentHarm 代表了 LLM safety benchmark 的一个趋势：从静态文本转向交互轨迹，从回答评分转向行为评分，从单点拒绝转向过程监督。类似趋势还会出现在 browser agents、coding agents、research agents、workflow automation agents 中。

未来高质量 benchmark 可能会同时具备三点：真实工具约束、动态攻击者、可解释评分轨迹。AgentHarm 已经完成了其中的第一步：让 agent harmfulness 成为一个可测量对象。

### 综合结论

AgentHarm 的研究价值在于把“agent 会不会被恶意用户直接滥用”这个问题具体化、工程化、可比较化。它暴露了一个重要事实：模型在聊天场景中的安全拒绝能力，并不保证迁移到工具调用场景；而一旦拒绝被绕过，模型可能仍保留很强的多步执行能力。这是后续 agent 安全研究必须正面处理的问题。

## 相关论文推荐

### AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents

AgentDojo 是 NeurIPS Datasets and Benchmarks 2024 的 agent 安全基准，重点是间接 prompt injection：用户目标本身是良性的，但第三方工具返回或外部资源中包含恶意指令。它和 AgentHarm 的互补关系很强。

AgentHarm 关注“恶意用户直接滥用 agent”，AgentDojo 关注“良性用户的 agent 被外部内容劫持”。前者更像 misuse benchmark，后者更像 tool-mediated security benchmark。两者合起来才能覆盖 agent 部署中的主要攻击面。

### ToolEmu: Identifying the Risks of LM Agents with an LM-Emulated Sandbox

ToolEmu 使用 LLM-emulated sandbox 来识别 LM agents 的风险，重点是善意用户意图下 agent 可能造成的意外风险。相比之下，AgentHarm 更强调恶意用户目标，并尽量使用固定合成工具和人工评分规则降低整体 LLM judge 依赖。

推荐把 ToolEmu 和 AgentHarm 放在一起读：前者帮助理解 agent 的 accidental harm，后者帮助理解 intentional misuse。

### HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

HarmBench 是 ICML 2024 的标准化 harmful behavior 与 red teaming 框架，主要面向单轮或聊天式模型拒绝鲁棒性。它在攻击评测、行为分类和拒绝判断上影响很大。

AgentHarm 可以看作把 HarmBench 式问题迁移到 agent 场景的一步，但它额外要求工具调用轨迹成立。因此，AgentHarm 的评分难度更高，也更贴近工具型模型部署风险。

### JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models

JailbreakBench 是 NeurIPS Datasets and Benchmarks 2024 的开放越狱鲁棒性 benchmark，关注聊天模型在 jailbreak prompt 下是否输出有害内容。它适合研究 prompt-level attack 与 refusal robustness。

AgentHarm 与它的区别在于评测对象从回答变成行动轨迹。读 JailbreakBench 可以理解通用越狱评测传统，读 AgentHarm 则能看到这些攻击如何迁移到 tool-use agent。

### StrongReject / A StrongReject for Empty Jailbreaks

StrongReject 系列指出，很多 jailbreak success metric 会被“空洞越狱”误导：攻击让模型不拒绝，但输出质量很差，实际没有完成有害请求。AgentHarm 明显吸收了这一点，用 Non-refusal Harm Score 和 benign capability baseline 检查攻击后能力是否保留。

这篇相关工作非常适合作为 AgentHarm 指标设计的背景阅读，因为它解释了为什么“拒绝率下降”本身不是充分证据。

## 思维导图

```mermaid
mindmap
  root((AgentHarm))
    tool-use safety
    malicious tasks
    synthetic tools
    rubric judge
    benign baseline
```
