# day 1

prefill TTFT first token

decode TPOT  Time Per Output Token

kv cache生命周期

- **分配**：请求进来，推理引擎为它预留一块显存放 KV Cache
- **填充**：Prefill 阶段一次性把整段 Prompt 的 K、V 算出来写进缓存；Decode 每步再追加 1 个 Token 的 K、V
- **使用**：每一步 Decode 都要读取全部历史 KV 来算 Attention
- **释放**：请求生成结束（EOS 或达到长度上限），显存归还

没有kv cache相当于每次decode都算一次prefill？

\[ \boxed{ M_{\text{KV}}=2\times B\times L\times S\times H_{kv}\times D\times P } \]

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

### GQA

如果G=4，4个query共享一组kv head

标准MHA 就是H_q=H_kv

GQA变成H_q = H_kv * G

**关键点**：在公式里是线性因子，这正是 **GQA（分组查询注意力）/ MQA（多查询注意力）**能大幅省显存的原因——让多个 Query 头共享一组 KV 头。比如 32 个 Query 头共享 8 个 KV 头（GQA），KV Cache 直接砍到原来的 ，而模型表达能力几乎不受影响。现代大模型几乎清一色采用 GQA，就是这笔账算出来的。

#### 碎片化

请求实际只生成了 100 个 Token，却按 2048 预留，剩下的座位全部空占——这叫**内部碎片**；

不同请求预留块之间的空隙又凑不成一整块可用空间——这叫**外部碎片**。

碎片化会让 KV Cache 的实际显存利用率大打折扣，大量显存被无效预留白白浪费。这正是 vLLM 的 PagedAttention 要解决的核心问题——它把连续大块换成了操作系统式的分页管理

### metrics

#### latency

TTFT

TPOT

E2E Latency 从提交请求到收到完整响应的总时间，**短输出场景 TTFT 主导，长输出场景 TPOT 主导**

E2E latency = TTFT + TPOT *(output_len-1)

先测清楚自己的输入输出长度分布，再谈优化

#### throughput

| 指标              | 含义                        | 适用视角               |
| :---------------- | :-------------------------- | :--------------------- |
| Token/s（系统）   | 全系统每秒生成的总 Token 数 | 衡量集群整体产能与成本 |
| Token/s（单用户） | 单个用户感受到的出字速度    | 约等于 1/TPOT          |
| RPS（Requests/s） | 每秒成功完成的请求数        | 衡量并发承载能力       |

系统吞吐会随并发上升而增长，直到 GPU 资源饱和后趋于平稳甚至回落。

**吞吐和延迟通常是一对矛盾**：把更多请求塞进一个 Batch 能提高系统吞吐，但每个请求的排队和计算时间变长，单用户延迟变差。

#### tail latency & Goodput

平均延迟是最容易骗人的指标。真实流量下，少数请求可能慢得离谱，而平均值把它们藏了起来。所以生产环境更看**尾延迟**

- **P95 / P99**：95% / 99% 的请求比它快——这才是用户投诉的来源

**Goodput（有效吞吐）**是比 Raw QPS 更贴近真实价值的指标。它只统计**满足 SLO（比如 TTFT < 1s 且 TPOT < 50ms）的那部分请求的吞吐**。一个系统 QPS 很高但大部分请求都超时，Goodput 就很低——用户根本用不爽

### Decode

decode mem-bound fp16搬运W数据，计算W/2，乘加算两个，所以AI=(2*W/2)/W = 1 FLOP/Byte，GPU 算力利用率理论上还不到 1%

所以需要batch，会放大batch 倍

X [B,L,D],W [D,D]

qkv BLD

GEMM的时候会展开成 [BL, D]

所以增大batch size会增大 AI

1. **为什么 Batch 能大幅提升吞吐却几乎不增加延迟**：把多个请求的 Decode 拼成一个 Batch，权重只搬一次就能服务整个 Batch 的所有 Token，算术强度成倍提升，把闲置的算力利用起来——这正是 Continuous Batching 的理论依据。
2. **为什么量化（尤其是 Weight-only）对 Decode 加速明显**：Decode 瓶颈在搬权重，把权重从 FP16 压到 INT4，要搬的字节数直接减少到 ，延迟随之大幅下降。



| 环节        | 做什么                                                  | 计算特性           | 潜在瓶颈                    |
| :---------- | :------------------------------------------------------ | :----------------- | :-------------------------- |
| Tokenize    | 把输入文本切成 Token ID 序列                            | CPU 任务           | 长文本或高并发时 CPU 成瓶颈 |
| Prefill     | 并行处理全部 Prompt Token，产出首 Token 与初始 KV Cache | GPU，Compute Bound | Prompt 很长时拉高 TTFT      |
| Decode Loop | 逐 Token 生成，每步读全量 KV、追加新 KV                 | GPU，Memory Bound  | 显存带宽，决定 TPOT         |
| Sampling    | 从输出分布按 Temperature/Top-p 等采样下一个 Token       | GPU，轻量          | 复杂采样或约束解码略增开销  |
| Detokenize  | 把 Token ID 拼回可读文本，流式返回                      | CPU 任务           | 与主循环争抢 CPU            |

Tokenize / Detokenize 是 **CPU 任务**，却容易被忽视。当 GPU 优化到位后，CPU 上的这些环节反而可能成为新瓶颈——这正是 vLLM V1 引擎用多进程把它们和 GPU 核心循环重叠起来的原因

# day 2