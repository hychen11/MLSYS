# LLM 推理

这一部分按「基础指标 → vLLM 使用 → 显存管理 → 调度 → 前缀复用 → 图优化 → 投机解码 → nano-vllm 实现」的顺序展开。

| 章节 | 解决的问题 | 关键词 |
| --- | --- | --- |
| [1. 推理基础与性能指标](01-basics.md) | 推理慢在哪、怎么衡量 | Prefill / Decode、KV Cache、TTFT / TPOT、Goodput |
| [2. vLLM：使用与核心能力](02-vllm.md) | 怎么用 vLLM 跑起来 | 离线推理、OpenAI 兼容服务、采样参数 |
| [3. PagedAttention](03-paged-attention.md) | KV Cache 显存碎片 | Block Table、引用计数、抢占、BlockPool |
| [4. Continuous Batching](04-continuous-batching.md) | GPU 空转、排队 | 迭代级调度、waiting / running、抢占 |
| [5. Prefix Cache 与 RadixAttention](05-prefix-cache.md) | 重复前缀的重复计算 | 块哈希、LRU 淘汰、Radix Tree |
| [6. Chunked Prefill 与 Token Budget](06-chunked-prefill.md) | 长 Prefill 拖慢 Decode | Token 预算、`num_computed_tokens` |
| [7. Attention 后端与图优化](07-graph-optimization.md) | CPU Kernel 启动开销 | CUDA Graph、Piecewise Graph、`torch.compile` |
| [8. 投机解码](08-speculative-decoding.md) | Decode 每步只出一个 Token | 草稿模型、拒绝采样、接受率 |
| [nano-vllm 源码走读](nano-vllm/01-engine.md) | 把以上概念落到代码 | LLMEngine、Qwen3、ModelRunner |

几条贯穿全篇的主线：

- **Decode 是访存瓶颈**：每步都要把全部权重和历史 KV 从显存读一遍，所以 Batching、量化、KV 压缩都在想办法「一次搬运，多次使用」。
- **显存决定并发**：KV Cache 能放多少，就能同时服务多少请求。PagedAttention、Prefix Cache、抢占都是围绕 KV 块的分配和回收展开的。
- **调度决定延迟分布**：同一批硬件，调度策略不同，TTFT、TPOT 和尾延迟的表现可以差很多。

## 参考资料

- [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/)：离线与在线调用。
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)：块哈希、共享和淘汰。
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)：抢占、Chunked Prefill 与性能调参。
- [vLLM CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)：完整图与分段图及后端兼容性。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：分页 KV Cache 管理。
- [SGLang 论文](https://arxiv.org/abs/2312.07104)：RadixAttention 与 Cache-Aware Scheduling。
- [nano-vllm 源码](https://github.com/GeeeekExplorer/nano-vllm)：精简实现。
- [FlashAttention 分页 KV 接口](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)：`page_block_size` 约束。
- [PyTorch 多进程实践](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)：CUDA 与 Poison Fork。
- [投机解码原论文](https://arxiv.org/abs/2211.17192)：草稿生成、目标模型验证与输出分布保持。
