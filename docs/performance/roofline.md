# Roofline 与 Nsight

判断一个 Kernel「慢在哪」，要回答两个问题：

1. **理论上**：给定 Kernel 的算术强度，它更可能受计算能力限制，还是受内存带宽限制？
2. **实际上**：这个 Kernel 用了 GPU 多少计算能力和内存带宽？如果两者都没跑满，瓶颈在哪？

第一个问题用 Roofline 模型估算，第二个问题用 Nsight Compute 实测。

## Roofline 模型

### 算术强度

**算术强度（Arithmetic Intensity，AI）** 是每从 HBM 搬运 1 Byte 数据所做的浮点运算次数：

$$
AI=\frac{\text{FLOPs}}{\text{HBM Bytes}}\quad(\text{FLOP/Byte})
$$

AI = 100 的意思是：每读写 1 Byte 数据，做 100 次运算。

### 屋顶与屋脊点

一个 Kernel 能达到的性能上限是：

$$
\boxed{P_{\mathrm{roof}}=\min(P_{\mathrm{peak}},\ AI\times BW_{\mathrm{peak}})}
$$

画成图，横轴是 AI，纵轴是可达到的 FLOP/s：左侧是一条斜率为带宽的斜线（Memory-bound 区），右侧是一条水平线（Compute-bound 区，高度等于峰值算力）。两条线的交点叫**屋脊点（Ridge Point）**，对应的 AI 叫临界算术强度：

$$
AI_{\text{critical}}=\frac{\text{Peak Compute (FLOP/s)}}{\text{Peak HBM Bandwidth (Byte/s)}}
$$

- $AI < AI_{\text{critical}}$：理论上 Memory-bound，加算力没用，要减少访存或提升带宽利用率。
- $AI > AI_{\text{critical}}$：理论上 Compute-bound，要提高计算效率（例如用上 Tensor Core）。

**例**：某 GPU 峰值算力 100 TFLOPS、带宽 1 TB/s，则 $AI_{\text{critical}} = 100\ \text{FLOP/Byte}$。一个 AI = 100 的 Kernel 恰好落在屋脊点上，两种资源的理论限制相等。

几张常见卡的屋脊点（BF16 稠密 Tensor Core 峰值 / HBM 带宽，取官方规格的近似值）：

| GPU | BF16 稠密峰值 | HBM 带宽 | $AI_{\text{critical}}$ |
| --- | ---: | ---: | ---: |
| A100 SXM 80GB | 312 TFLOPS | 2.0 TB/s | ≈ 153 |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | ≈ 295 |

新一代 GPU 的算力增长通常快于带宽增长，屋脊点不断右移，越来越多的算子会落入 Memory-bound 区。

### 峰值算力怎么来

对传统 FP32 CUDA Core：

$$
P_{\mathrm{FP32}} = N_{\mathrm{FP32\ lanes}} \times 2 \times f_{\mathrm{clock}}
$$

其中 2 是因为一次 FMA（乘加）算两次浮点运算。例如 10,000 个 FP32 lane、2 GHz：

$$
P=10000\times2\times2\times10^9=40\ \mathrm{TFLOPS}
$$

!!! warning
    FP32 CUDA Core 峰值不等于 FP16/BF16 Tensor Core 峰值，后者通常高一个数量级。做 Roofline 分析时，峰值算力要和 Kernel 实际使用的数据类型、计算单元对应。

## 例子：为什么 Prefill 是 Compute-bound，Decode 是 Memory-bound

关键在于 Prefill 能用矩阵乘法，让**同一份权重被多个 Token 重复使用**。

考虑一个 Linear 层 $Y=XW$：

- $X\in\mathbb{R}^{M\times K}$：输入的 hidden states，$M$ 是本次处理的 Token 数
- $W\in\mathbb{R}^{K\times N}$：权重
- $Y\in\mathbb{R}^{M\times N}$：输出

用 FP16/BF16（每元素 2 Bytes）：

$$
\text{FLOPs}=2MKN,\qquad \text{Bytes}=2(MK+KN+MN)
$$

$$
AI=\frac{2MKN}{2(MK+KN+MN)}=\frac{MKN}{MK+KN+MN}
$$

取 $K=N=4096$：

$$
AI=\frac{4096M}{2M+4096}
$$

- **Decode**（$M=1$）：$AI = 4096/4098 \approx 1$。每读 1 Byte 权重只做约 1 次运算，远低于任何 GPU 的屋脊点，是典型的 Memory-bound。
- **Prefill**（$M$ 为 Prompt 长度）：$M$ 越大 AI 越高，上限趋近 2048。

令 $AI > AI_{\text{critical}}$ 可以解出「多少 Token 才能把这个 Linear 喂成 Compute-bound」：A100 上约 $M > 165$，H100 上约 $M > 345$。所以几百 Token 以上的 Prefill 基本是 Compute-bound；而 Decode 要靠 **Batching** 把多个请求的 Token 拼成一个大 $M$ 来提高 AI，这正是 [Continuous Batching](../inference/04-continuous-batching.md) 的理论依据。

!!! note
    这里只算了 Linear 层的权重流量。Decode 的 Attention 还要读全部历史 KV，而 KV 是每个请求私有的，不能靠 Batching 复用，所以长上下文 Decode 的 Attention 部分始终是 Memory-bound。

## 实际中怎么判断瓶颈

1. **理论估算**：算出 FLOPs 和 HBM Bytes，与目标 GPU 的屋脊点比较。
2. **Nsight Compute Profiling**：看 DRAM Throughput、SM Throughput、Tensor Core 利用率，以及 Memory / Compute 相关的 stall 指标。
3. **对照实验**：优化访存后性能明显提升，说明存在访存瓶颈；提高计算效率后显著加速，则受计算瓶颈限制。

## Nsight Compute 常用指标

| 指标 | 含义 | 如何理解 |
| --- | --- | --- |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽利用率 | 接近 100% 说明 DRAM 吞吐接近工具定义的持续峰值 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 综合吞吐利用率 | 高值说明 SM 内至少某些计算或数据通路压力较大 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved Occupancy | 活跃 Warp 占理论可驻留 Warp 上限的比例 |
| `smsp__warps_eligible.avg.per_cycle_active` | Eligible Warps | 每周期有多少 Warp 已准备好发射指令 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | Issue 活跃度 | 指令发射管线的使用程度 |
| `gpu__time_duration.sum` | Kernel 执行时间 | 用来换算实际 FLOP/s 和 Byte/s |

Kernel 性能明显低于 Roofline 上限时，先看 SM / DRAM Throughput：

- **DRAM 吞吐很高**：HBM 接近饱和。检查 HBM 流量是否必要、L2 命中率、全局内存访问是否合并（Coalescing）。
- **SM 吞吐很高**：计算管线接近饱和。检查是否用上 Tensor Core、指令构成（Instruction Mix）、计算效率。
- **两者都不高**：GPU 没有持续做有效工作。继续看 Occupancy、Eligible Warps、Warp Stall 原因、Grid 大小和 Kernel 时长（太短的 Kernel 会被启动开销主导，见 [图优化](../inference/07-graph-optimization.md)）。

Nsight Systems（`nsys`）则从时间线角度看整个程序：Kernel 之间有没有空隙、CPU 是否来不及发射、数据拷贝和计算是否重叠。通常先用 `nsys` 找到值得优化的 Kernel，再用 `ncu` 深挖单个 Kernel。

## ITL Spike

在线推理时偶尔会看到 ITL（Inter-Token Latency，相邻两个输出 Token 的间隔）突然变大，称为 ITL spike。常见原因：

| 原因 | 现象 | 缓解方式 |
| --- | --- | --- |
| 长 Prompt 的 Prefill 与 Decode 挤在同一步 | 新请求到达时，同批 Decode 请求的出字明显变慢 | [Chunked Prefill](../inference/06-chunked-prefill.md)、调小 `max_num_batched_tokens` |
| KV Cache 不足触发抢占 | 被抢占请求恢复时要重算，自身延迟跳变；重算的 Prefill 又会拖慢别人 | 降低并发、提高 `gpu_memory_utilization`、开启 Prefix Cache |
| Batch 大小超出 CUDA Graph 捕获范围 | 退回 eager 执行，单步耗时上升 | 调整 `cudagraph_capture_sizes` |
| CPU 侧阻塞 | Python GC、Tokenize/Detokenize 或调度器开销占满 CPU | 多进程前后处理、减少每步 Python 开销 |

排查时用 `nsys` 时间线对齐「ITL 突增的那一步」和「这一步里调度了什么」，通常能直接定位原因。
