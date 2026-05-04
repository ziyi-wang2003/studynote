---
title: 数学 Reasoning 思维导图
summary: 数学推理方向论文与技术路线的 Mermaid 思维导图。
created: '2026-04-30 00:00:00+08:00'
updated: '2026-05-04 00:00:00+08:00'
order: 99
pinned: false
---

# 数学 Reasoning 思维导图

```mermaid
mindmap
  root((数学 Reasoning))
    核心问题
      答案正确率
        GSM8K
        MATH
        AIME
      推理过程可靠性
        step-wise soundness
        proof repair
        scalable oversight
      证明形式
        informal proof
        formal proof
        informal-formal bridge
    方法类别
      Test-time search
        rStar-Math
          MCTS
          code-augmented CoT
          process preference model
      Process evaluation
        IneqMath
          inequality proving
          bound estimation
          relation prediction
          LLM-as-judge
      Formal theorem proving
        MiniF2F
        DeepSeek-Prover-V2
      Training and post-training
        WizardMath
        process supervision
        theorem-guided reasoning
      Latent recurrent reasoning
        LoopFormer
          elastic depth
          shortcut consistency
          budget-conditioned loops
        Ouro LoopLM
          large-scale looped LM
          learned depth allocation
        Mixture-of-Recursions
          token-level recursive depth
    代表论文
      rStar-Math
        小模型自演化数学推理
      IneqMath
        NeurIPS 2025 Spotlight
        答案正确不代表证明正确
        细粒度过程 judge
      LoopFormer
        ICLR 2026
        可变预算隐空间推理
        t 与 Δt 调制循环轨迹
    后续研究
      theorem retrieval
      proof planning
      proof repair
      symbolic verifier
      Lean4 integration
      loop trajectory diagnostics
      adaptive budget scheduling
```
