# Attention 后端与图优化

Attention 后端决定注意力计算的实现；CUDA Graph 和 `torch.compile` 则用于减少前向执行中的 CPU 启动与小算子开销。

**同样一批请求的一次前向计算，怎么让它跑得更快？**

- 一是选对高效的 **Attention 后端 Kernel**；
- 二是用 **CUDA Graph + `torch.compile`** 消除 Decode 阶段那些看不见却致命的 CPU 开销。这也是 vLLM V1 相比 V0 提速的重要来源。

## Decode 的第二重等待：CPU 启动开销

当模型不大、Batch 也不大时，还有一个更隐蔽的瓶颈会冒出来：**CPU 发射 Kernel 的开销**。

GPU 自己不会主动干活，它每做一个操作（一次矩阵乘、一次 LayerNorm、一次激活……），都要由 CPU 通过一次「启动（launch）」命令告诉它。跑一遍 Transformer，有成百上千个这样的小操作，就对应成百上千次 CPU→GPU 的启动调用。每次启动本身有固定的 CPU 开销（构造参数、驱动调度、提交队列），量级在微秒级。

平时这点开销不算什么——因为 Prefill 一步要处理成千 Token，每个 GPU Kernel 一跑就是几百微秒甚至毫秒，CPU 那点启动开销被淹没了。但 **Decode 每步只处理 1 个（或很少几个）Token**，每个 Kernel 的实际计算可能就几微秒，结果**CPU 发射这个 Kernel 的时间比 GPU 执行它的时间还长**。于是 GPU 干几微秒就停下来，眼巴巴等 CPU 发下一条命令——算力再次闲置，只不过这次不是等数据，是**等指令**。

Kernel 越短，CPU 发射开销在单步耗时中的占比越可能上升。

Decode 阶段有两重「等待」——**等数据**（Memory Bound，靠 Batching/量化缓解）和**等指令**（CPU launch 开销，靠本节的图优化缓解）。小模型、小 Batch 时，后者尤其突出。这就是为什么图优化对 Decode 加速格外有效。

## Attention 后端

| 后端 | 关注点 |
| --- | --- |
| FlashAttention | IO 感知的高性能注意力实现。 |
| FlashInfer | 面向推理的注意力 Kernel 与 KV Cache 布局。 |
| FlashMLA 等 | 适配 MLA 模型的专用实现。 |
| Triton 实现 | 用 Triton 编写的特定注意力 Kernel。 |

后端由模型架构、GPU、数据类型、KV 布局和 vLLM 版本共同决定。通常先使用自动选择，再针对实际的 Prefill/Decode 比例做 Benchmark；不要只凭后端名称推断性能。

## 后端如何处理 Prefill/Decode 混合批次

这里和[上一章](06-chunked-prefill.md)直接相关。vLLM V1 的统一调度器会把 **Prefill 的 Chunk 和 Decode 请求塞进同一步**——那么这一步的 Attention 该怎么算？毕竟 Prefill 部分要算「一整段新 Token 之间 + 对历史的注意力」，Decode 部分只算「1 个新 Token 对全部历史的注意力」，两者的计算形态不一样。

执行层必须正确处理混合批次中的变长 Query 和 KV。具体后端可能使用变长接口、分段处理或不同 Kernel 路径，不能一概而论为「一次 Kernel 调用完成」。

## CUDA Graph

CUDA Graph 的思路是「录一次，放多次」：先把一整串 Kernel 启动连同参数**录制（capture）**成一张图，之后每一步只需一次 `graph.replay()`，由驱动在 GPU 侧连续执行整张图里的所有 Kernel。成百上千次 CPU 启动被压缩成一次，「等指令」的问题就消失了。

```python
g = torch.cuda.CUDAGraph()
static_input = torch.zeros(batch, hidden, device="cuda")
with torch.cuda.graph(g):                 # 录制：期间的 Kernel 只记录、不真正执行
    static_output = model(static_input)

static_input.copy_(real_input)            # 每步把新数据拷进固定地址的输入缓冲区
g.replay()                                # 一次调用重放全部 Kernel
result = static_output.clone()
```

注意图里记录的是**固定的显存地址**，所以输入输出都要用预先分配好的静态缓冲区，每步把数据拷进去。

CUDA Graph 有个硬性前提：**录制的操作序列和张量形状必须固定**。这对 Decode 是天作之合——每步都是「处理固定 Batch 个 Token」，形状稳定，可以针对不同的 Batch Size 分别录制好图，运行时按当前 Batch Size 选对应的图重放。

vLLM 可为若干捕获尺寸预先录图，运行时将实际 Token 数补齐到合适的尺寸重放。是否能使用完整 CUDA Graph，还取决于 Attention 后端和图模式；较新的版本同时支持 full、piecewise 等模式，不能简单认为只有 Decode 才可能进入 CUDA Graph。具体捕获尺寸以所用版本配置为准。

## `torch.compile`

CUDA Graph 解决了「启动开销」，但 Kernel 数量本身没变。`torch.compile` 把模型的前向计算**编译**成优化后的图，通过算子融合（fusion）等手段减少 Kernel 数量和中间结果的显存读写。两者互补，vLLM V1 默认同时使用。

## Piecewise CUDA Graph

CUDA Graph 要求整段操作形状固定，而 **Attention 恰恰是最难被 CUDA Graph 兼容的部分**——它涉及变长的 KV、复杂的 Mask，形状不那么规整。如果因为 Attention 不好处理就放弃整张图的 CUDA Graph，就太可惜了。

vLLM V1 的巧解是 **Piecewise CUDA Graph（分段 CUDA Graph）**：

- 把整个前向图**在 Attention 算子处切开**（Attention 被包装成一个「不透明」的自定义算子，编译器不去动它）。
- **Attention 之外的部分**（LayerNorm、各种 GEMM、激活、残差……这些都是规整的 token-wise 操作）用 CUDA Graph 捕获、重放。
- **Attention 本身保持 eager（即时）执行**，保留它处理变长 KV 的灵活性。

**核心概念**：Piecewise CUDA Graph 在不适合整图捕获的算子处切分，分别捕获可重放的片段。完整图能否覆盖 Attention，由后端能力及图模式决定。

纯 Decode 往往更容易重放完整图；Prefill 和混合批次是否可用，需要结合后端支持情况判断。

## 为什么将 Attention 作为编译边界？

Attention 的麻烦在于：它需要处理不同长度的 Query、KV Cache、Attention Metadata 等动态信息。

但 RMSNorm、MLP、Linear 等操作通常更规整，更适合 CUDA Graph。

于是 vLLM 用自定义算子把 Attention 作为一个边界：
```
torch.ops.vllm.unified_attention_with_output(...)
```

这里的黑盒并不是说 Attention 不能被优化，而是：

**Dynamo 不深入追踪它的内部实现，而是把整个 Attention 当作一个原子操作。**

这样它仍然可以被放在整体图里表示，同时 vLLM 可以选择在它的位置切开计算图。

```python
from vllm import LLM
from vllm.config import CompilationConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        # 示例：为这些捕获尺寸录图；实际行为取决于图模式与后端。
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
        # 可选：为特定静态形状做 Inductor 自动调优（首次较慢，默认关闭）
        compile_sizes=[1, 2, 4, 8],
    ),
)
```

- `cudagraph_capture_sizes`：控制为哪些 Batch Size 捕获图。具体默认列表随版本变化；列表越长，覆盖越细、Padding 浪费越少，但捕获耗时和显存占用越多。
- `compile_sizes`：让 Inductor 为指定静态形状做自动调优，找更快的 Triton 配置——收益是运行更快，代价是首次编译更慢，所以默认关闭。

编译产物会写入**编译缓存**，默认位于 `~/.cache/vllm/torch_compile_cache`（缓存键包含所有相关配置与被追踪的源文件）。缓存目录可在部署间**直接拷贝**以省掉重复编译时间；调试时若想强制重编，可用 `VLLM_DISABLE_COMPILE_CACHE=1` 关闭缓存

这里编译的是**Graph（计算图），但最终生成的是 Kernel（GPU 可执行代码）**

- **收益**：Decode 阶段 CPU 开销大幅下降，小模型/小 Batch 下的吞吐和延迟明显改善；`torch.compile` 的算子融合进一步减少 Kernel 数量。
- **代价**：
  - **启动变慢**：服务启动时要做编译和 CUDA Graph 捕获，首次拉起耗时增加（vLLM 用编译缓存缓解，缓存目录可复制以加速后续启动）。
  - **显存占用**：捕获多个 Batch Size 的 CUDA Graph 需要额外显存。
  - **调试复杂度**：编译后的图不如 eager 模式好调试，排查数值问题时可能需要临时关掉。

调试时可用 `--enforce-eager` 对比 eager 执行；它会跳过编译和 CUDA Graph 捕获。生产配置仍应以实测延迟、吞吐和启动成本为准。
