# ARPO: Agentic Reinforced Policy Optimization — 阅读笔记

##  论文动机

* 现有 **轨迹级 RL (trajectory-level RL)**：只对完整 rollout 比较，难以捕捉 **多轮工具调用** 中的细粒度行为。
* 观测到：LLM 在 **工具调用之后的 10–50 个 token**，token 熵显著上升 → 表示 **不确定性增加**。
* 传统 RL 忽视了这类“高熵阶段”的探索机会。

---

##  核心方法

ARPO 提出两个关键机制：

### 1. 熵驱动的自适应 Rollout (Entropy-based Adaptive Rollout)

* **先采样 N 条完整轨迹**，作为全局探索。
* 每次 **工具调用后**，额外生成 k 个 token，计算熵变化：

  $$
  \Delta H_t = Normalize(H_t - H_{initial})
  $$
* 根据公式计算分支概率：

  $$
  P_t = \alpha + \beta \cdot \Delta H_t
  $$

  * 若 $P_t > \tau$ → **分支 Z 条局部采样**。
  * 否则继续当前轨迹。
* 直到补足总预算 M。

效果：在高不确定性步骤自适应地拓展更多路径，提升探索效率。

---

### 2. 优势归因估计 (Advantage Attribution Estimation)

* 问题：自适应分支后，轨迹包含 **共享部分 (prefix)** 和 **分支部分**，如何分配 reward？
* **Hard 方案**：

  * 共享 token → 平均 advantage。
  * 分支 token → 各自 advantage。
* **Soft 方案（默认）**：

  * 基于 GRPO，自动通过 importance ratio 区分共享/分支 token。
  * 实验更稳定，reward 更高。

效果：避免 reward 被错误归因，强化模型对分支差异的学习。

---

## 实验结果

* 在 **数学推理、知识推理、深度搜索** 13 个 benchmark 上，ARPO **全面优于** GRPO/DAPO/REINFORCE++ 等轨迹级 RL。
* 在同等条件下：

  * **只用一半的工具调用预算** 就能达到更好性能。
  * 提升平均准确率 \~4%。

---