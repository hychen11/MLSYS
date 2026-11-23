# 推理基础与性能指标

## Prefill、Decode 与 KV Cache

Prefill 并行处理输入 Prompt，是 TTFT（Time to First Token，首 Token 延迟）的主要组成部分；Decode 逐 Token 生成输出，通常用 TPOT（Time per Output Token，平均每个后续输出 Token 的耗时）衡量。

KV Cache 的生命周期：

- **分配**：请求进入后，引擎按需为 KV Cache 分配显存；具体粒度取决于缓存管理方式。
- **填充**：Prefill 阶段一次性把整段 Prompt 的 K、V 算出来写进缓存；Decode 每步再追加 1 个 Token 的 K、V
- **使用**：每一步 Decode 都要读取全部历史 KV 来算 Attention
- **释放**：请求生成结束（EOS 或达到长度上限），显存归还

如果不保存 KV，每生成一个 Token，都要重新计算已有上下文的 K、V；KV Cache 用显存换掉了这部分重复计算。

在标准全注意力、各层 KV 形状相同且不计块元数据等开销时，缓存容量约为：

$$
\boxed{ M_{\text{KV}}=2\times B\times L\times S\times H_{kv}\times D\times P }
$$

其中：

| 参数       | 含义                             | 示例           |
| ---------- | -------------------------------- | -------------- |
| \(2\)      | K 和 V 两份缓存                  | 2              |
| \(B\)      | Batch Size（同时缓存的序列数）   | 1              |
| \(L\)      | Transformer 层数                 | 32             |
| \(S\)      | Sequence Length（缓存 Token 数） | 4096           |
| \(H_{kv}\) | KV Heads 数量                    | 8              |
| \(D\)      | Head Dimension                   | 128            |
| \(P\)      | 每个元素的字节数                 | BF16 = 2 Bytes |

注意：这里使用的是 KV Heads，而不是 Query Heads。

**算一笔账**：以上表的配置（接近 Llama-3-8B：32 层、8 个 KV 头、head_dim 128、BF16）为例，每个 Token 的 KV 占用为

$$
2\times32\times8\times128\times2\ \text{Bytes}=128\ \text{KiB}
$$

一条 4096 Token 的序列就要 512 MiB。如果要同时服务 64 条这样的序列，KV Cache 需要 32 GiB，比 16 GiB 的 BF16 权重还大。**KV Cache 能放多少，基本决定了能同时服务多少请求**，后面几章的显存管理都围绕它展开。

## GQA 与 KV 头数

若分组大小 `G = 4`，每 4 个 Query 头共享一组 K/V 头。MHA 中 `H_q = H_kv`；GQA 中 `H_q = G × H_kv`；MQA 则只有一个 KV 头。

**关键点**：KV Cache 大小与 KV 头数成正比。若 Query 头数为 32、KV 头数为 8，其他条件不变时，相比 32 个 KV 头的 MHA，缓存约降至 **1/4**。GQA/MQA 减少显存占用，但效果和性能仍需按模型评估。

## KV Cache 的碎片化

早期系统给每个请求按最大长度预留一段连续显存，会造成两类浪费：

- **内部碎片**：请求实际只生成了 100 个 Token，却按 2048 预留，剩下的空间全部空占。
- **外部碎片**：不同请求的预留区之间留下的空隙大小不一，凑不成一整块可用空间。

碎片化让 KV Cache 的实际利用率大打折扣。这正是 vLLM 的 [PagedAttention](03-paged-attention.md) 要解决的核心问题：把连续大块换成操作系统式的分页管理。

## 性能指标

### 延迟

| 指标 | 含义 |
| --- | --- |
| TTFT | 从提交请求到收到首个输出 Token 的时间，包含排队、预处理与 Prefill 等开销。 |
| TPOT | 首 Token 之后，平均每个输出 Token 的耗时。 |
| E2E latency | 从提交请求到收到完整响应的时间。 |

输出长度为 `N ≥ 1` 时，若 TPOT 取后续 Token 间隔的平均值，则 `E2E latency = TTFT + (N - 1) × TPOT`。短输出通常更受 TTFT 影响，长输出则更受 TPOT 影响。

所以先测清楚自己业务的输入、输出长度分布，再谈优化。

### 吞吐

| 指标              | 含义                        | 适用视角               |
| :---------------- | :-------------------------- | :--------------------- |
| Token/s（系统）   | 全系统每秒生成的总 Token 数 | 衡量集群整体产能与成本 |
| Token/s（单用户） | 单个用户感受到的出字速度    | 约等于 1/TPOT          |
| RPS（Requests/s） | 每秒成功完成的请求数        | 衡量并发承载能力       |

系统吞吐会随并发上升而增长，直到 GPU 资源饱和后趋于平稳甚至回落。

**吞吐和延迟通常是一对矛盾**：把更多请求塞进一个 Batch 能提高系统吞吐，但每个请求的排队和计算时间变长，单用户延迟变差。

### 尾延迟与 Goodput

平均延迟是最容易骗人的指标。真实流量下，少数请求可能慢得离谱，而平均值把它们藏了起来。所以生产环境更看**尾延迟**

- **P95 / P99**：95% / 99% 的请求比它快。用户投诉往往来自这部分慢请求。

**Goodput（有效吞吐）** 比原始 QPS 更贴近真实价值：它只统计**满足 SLO（例如 TTFT < 1s 且 TPOT < 50ms）的那部分请求的吞吐**。一个系统 QPS 很高但大部分请求都超时，Goodput 就很低。

## Decode 的性能瓶颈

小 Batch 的 Decode 往往受显存带宽限制（推导见 [Roofline](../performance/roofline.md)）：以 FP16/BF16 线性层为例，单 Token 对每个权重约做一次乘加（约 2 FLOPs），读取一个 2 字节权重，忽略激活和缓存流量时，权重相关算术强度约为 **1 FLOP/Byte**。这只是简化估算，不能据此直接推断整卡利用率。

将多个请求组成 Batch 后，线性层计算可看作 `[B, D] × [D, D]`；多个 Token 复用同一批权重，通常能提高吞吐。收益会受 KV 读取、Batch 大小、Kernel 和硬件限制。

1. **Batching**：同一次前向让多个 Decode Token 分摊权重读取，通常提高系统吞吐；Batch 过大也会拉长单步耗时及排队时间。
2. **Weight-only [量化](../quantization/basics.md)**：仅比较原始权重，INT4 的位宽是 FP16 的 **1/4**。实际节省和加速还受量化元数据、反量化、KV 访问与 Kernel 实现影响。

| 环节        | 做什么                                                  | 计算特性           | 潜在瓶颈                    |
| :---------- | :------------------------------------------------------ | :----------------- | :-------------------------- |
| Tokenize    | 把输入文本切成 Token ID 序列                            | CPU 任务           | 长文本或高并发时 CPU 成瓶颈 |
| Prefill     | 并行处理全部 Prompt Token，产出首 Token 与初始 KV Cache | GPU，Compute Bound | Prompt 很长时拉高 TTFT      |
| Decode Loop | 逐 Token 生成，每步读全量 KV、追加新 KV                 | GPU，Memory Bound  | 显存带宽，决定 TPOT         |
| Sampling    | 从输出分布按 Temperature/Top-p 等采样下一个 Token       | GPU，轻量          | 复杂采样或约束解码略增开销  |
| Detokenize  | 把 Token ID 拼回可读文本，流式返回                      | CPU 任务           | 与主循环争抢 CPU            |

Tokenize / Detokenize 是 **CPU 任务**，却容易被忽视。当 GPU 优化到位后，CPU 上的这些环节反而可能成为新瓶颈——这正是 vLLM V1 引擎用多进程把它们和 GPU 核心循环重叠起来的原因。
