---
year: 2025
venue: Research Map
keywords:
  - agent safety
  - benchmark
  - tool use
  - jailbreak robustness
url: https://arxiv.org/abs/2410.09024
digest: "Agent 安全基准方向关注工具型 LLM agent 在直接恶意请求、间接注入、工具误用和拒绝鲁棒性下的可测量风险。"
---

# Agent安全基准_思维导图

```mermaid
mindmap
  root((Agent安全基准))
    Tool-use harm
      AgentHarm
        malicious tasks
        synthetic tools
        rubric judge
        benign baseline
        private split
    Indirect attack
      AgentDojo
        prompt injection
        tool workflow
        dynamic defense
    Sandbox risk
      ToolEmu
        emulated tools
        accidental harm
        safe testing
    Refusal robustness
      HarmBench
        red teaming
        harmful behaviors
        refusal metric
      JailbreakBench
        jailbreak prompts
        open benchmark
        robustness
```
