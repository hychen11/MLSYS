# 投机解码

## 动机

[第 1 章](01-basics.md#decode-的性能瓶颈)说过，小 Batch 的 Decode 是 Memory-bound：每生成一个 Token 都要把全部权重读一遍，而算力大量闲置。

如果一次前向能**同时验证多个 Token**，读一遍权重就能产出多个 Token，闲置的算力就被用起来了。这是投机解码（Speculative Decoding）的出发点：用一个便宜的方式先「猜」几个 Token，再让目标模型一次性验证。

## 基本流程

1. **起草（Draft）**：用一个小而快的草稿模型（或其他机制）自回归地生成 $k$ 个候选 Token $x_1,\dots,x_k$，同时记下草稿分布 $q_i(\cdot)$。
2. **验证（Verify）**：把这 $k$ 个候选拼在已有上下文后面，目标模型**一次前向**算出 $k+1$ 个位置上的目标分布 $p_1,\dots,p_{k+1}$。这一步和一次小 Prefill 类似，权重只读一遍。
3. **接受 / 拒绝**：从 $x_1$ 开始逐个判断，遇到第一个被拒绝的位置就停下，丢弃它和之后的所有候选。
4. **补一个 Token**：在被拒绝的位置按修正分布重新采样一个 Token；如果全部接受，就从 $p_{k+1}$ 再采样一个。

所以每一轮至少产出 1 个 Token（不比普通 Decode 差），最多产出 $k+1$ 个。

## 接受规则：保证输出分布不变

对候选 $x_i$，以概率

$$
\min\left(1,\ \frac{p_i(x_i)}{q_i(x_i)}\right)
$$

接受。如果被拒绝，从修正分布

$$
p'(x)=\frac{\max\big(0,\ p_i(x)-q_i(x)\big)}{\sum_{x'}\max\big(0,\ p_i(x')-q_i(x')\big)}
$$

中重新采样。

可以证明，这样得到的 Token 严格服从目标分布 $p_i$。直觉是：草稿模型「高估」的 Token（$q>p$）按比例被拒掉，「低估」的部分（$p>q$）由修正分布补上。所以**投机解码是无损的**，输出质量和直接用目标模型采样完全相同，只是更快。

贪心解码时规则更简单：候选 Token 等于目标模型的 argmax 就接受，否则用目标模型的 argmax 替换并停止。

## 能快多少

设每个候选的接受率为 $\alpha$（假设各位置独立），每轮起草 $k$ 个，则每轮期望产出的 Token 数为

$$
\mathbb{E}[\text{tokens}]=\frac{1-\alpha^{k+1}}{1-\alpha}
$$

例如 $\alpha=0.8$、$k=4$ 时约为 3.4 个，也就是目标模型的前向次数减少到约 30%。实际加速比还要扣除草稿模型的开销和验证步变重的代价。

影响收益的因素：

- **接受率**：草稿和目标模型越接近越好。代码补全、改写这类「可预测」的任务接受率高；开放式创作较低。
- **Batch 大小**：投机解码用的是 Decode 阶段闲置的算力。Batch 很大时 GPU 已经接近 Compute-bound，多验证的 Token 不再「免费」，收益会下降甚至变负。所以它主要用于低延迟、小 Batch 的场景。
- **$k$ 的选择**：$k$ 越大，单轮可能产出越多，但后面的候选越容易被拒，浪费的起草和验证计算也越多。

## 常见的起草方式

| 方式 | 草稿来源 | 特点 |
| --- | --- | --- |
| 独立草稿模型 | 同词表的小模型（如 0.5B 配 7B） | 通用，但要额外部署和占显存 |
| N-gram / Prompt Lookup | 从 Prompt 或已生成内容里匹配重复片段 | 无需额外模型，适合摘要、代码修改等大量复制输入的任务 |
| Medusa | 在目标模型上加多个预测头，各自预测第 $i$ 个未来 Token | 不需要独立模型，候选组成树状结构一起验证 |
| EAGLE 系列 | 轻量草稿层，基于目标模型的隐状态做特征级自回归 | 接受率高，是目前常用的方案之一 |
| MTP | 模型训练时自带多 Token 预测模块（如 DeepSeek-V3） | 推理时可直接作为草稿 |

## 在推理引擎中的位置

在 vLLM V1 中，投机 Token 被统一进了 [Token 预算调度](06-chunked-prefill.md#v1-统一调度)：一个请求这一步要处理的 Token 数是 `1 + 候选数`，调度器只需把 `num_computed_tokens` 推向 `num_tokens_with_spec`。被拒绝的候选对应的 KV 槽位会被回滚，下一步覆盖写入。

[nano-vllm](nano-vllm/01-engine.md) 没有实现投机解码，它的 `SamplingParams` 和调度器都只考虑每步一个 Token。

## 参考

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：原论文，提出接受规则并证明无损。
- [Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)：免独立草稿模型的方案。
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/stable/features/spec_decode/)
