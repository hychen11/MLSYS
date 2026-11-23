# MLSys 笔记

这里收录我在 LLM 推理系统、GPU 性能分析、CUDA 算子和模型量化方面的学习笔记。内容以「原理 → 源码 → 工程取舍」为主线，尽量把每个优化放回它要解决的瓶颈里去理解。

## 内容地图

<div class="grid cards" markdown>

-   **LLM 推理**

    ---

    从 KV Cache 和性能指标出发，依次讲 vLLM 的 PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill、图优化和投机解码，最后走读 nano-vllm 的精简实现。

    [:octicons-arrow-right-24: 开始阅读](inference/index.md)

-   **性能分析**

    ---

    Roofline 模型、算术强度估算，以及用 Nsight Compute 判断 Kernel 瓶颈的方法。

    [:octicons-arrow-right-24: Roofline 与 Nsight](performance/roofline.md)

-   **CUDA 算子**

    ---

    以 Sum Reduce 为例，从朴素实现一步步优化到 Warp Shuffle + 向量化加载；另附 Shared Memory Bank Conflict 专题。

    [:octicons-arrow-right-24: Reduce 优化](cuda/reduce.md)

-   **量化**

    ---

    整数与浮点格式、量化的基本映射与粒度、W4A16 与 W8A8 各自加速了什么，以及 AWQ / SmoothQuant 的核心思想。

    [:octicons-arrow-right-24: 量化基础](quantization/basics.md)

</div>

## 推荐阅读顺序

1. [Roofline 与 Nsight](performance/roofline.md)：先建立「Compute-bound / Memory-bound」的判断框架，后面所有优化都会回到这里。
2. [推理基础与性能指标](inference/01-basics.md)：Prefill / Decode 的区别、KV Cache 显存估算、TTFT / TPOT / Goodput。
3. LLM 推理第 2–8 章：按章节顺序阅读，各章之间有前后依赖。
4. [nano-vllm 源码走读](inference/nano-vllm/01-engine.md)：用约一千行代码把前面的概念串起来。
5. [量化](quantization/basics.md) 与 [CUDA 算子](cuda/reduce.md) 可以独立阅读。

!!! note "关于版本"
    vLLM、SGLang 等框架迭代很快。文中涉及的默认值、内部类名和代码路径以写作时阅读的版本为准，伪代码只还原主干逻辑，并非逐字源码。部署和调参请以所用版本的官方文档为准。
