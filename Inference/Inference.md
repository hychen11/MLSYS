# day 1 basic

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

# day 2 vllm

| 特性                    | 说明                                                         |
| :---------------------- | :----------------------------------------------------------- |
| **PagedAttention**      | 借鉴操作系统虚拟内存分页思想管理 KV Cache，显存利用率接近最优 |
| **Continuous Batching** | 动态地将新请求插入正在处理的批次，而非等一批全部完成再开始下一批 |
| **OpenAI 兼容 API**     | 直接替换 OpenAI 客户端的 `base_url`，无需修改业务代码        |
| **200+ 模型支持**       | 覆盖 Llama、Qwen、Mistral、DeepSeek 等主流架构，含 MoE 和多模态模型 |
| **多硬件支持**          | NVIDIA GPU、AMD GPU、Google TPU、Intel Gaudi、CPU 等         |
| **丰富的量化方案**      | FP8、INT8、INT4、GPTQ、AWQ、GGUF 等开箱即用                  |

* PagedAttention -> high gpu util
* Continuous Batching -> high throughput
* CUDA Kernels -> low latency

LLM 推理的 Decode 阶段需要缓存之前所有 token 的 Key 和 Value 向量（即 KV Cache），以避免重复计算。传统做法是为每个请求**预分配一块连续的显存空间**，大小按模型支持的最大序列长度计算。

这就带来了三个问题：

- **内部碎片**：大多数请求用不到最大长度，预留空间大量浪费
- **外部碎片**：请求结束后释放的空间大小不一，难以被新请求复用
- **无法共享**：Beam Search 等场景中，多个候选序列共享前缀部分的 KV Cache，但连续内存布局无法实现共享

**提示**：对于一个 13B 参数的模型，单个请求的 KV Cache 在最大序列长度下可能占用数 GB 显存。如果同时处理几十个请求，显存瓶颈会比模型权重本身更严重。

1. **分块存储**：KV Cache 不再要求连续内存，而是被切分成固定大小的 Block（如 16 个 token 为一个 Block）
2. **按需分配**：每生成一个 token，只在当前 Block 未满时追加，填满后才分配新的 Block
3. **映射表管理**：每个请求维护一个 Block Table，记录自己的逻辑 Block 到物理 Block 的映射关系
4. **内存共享**：多个请求如果有相同的前缀（如 System Prompt），可以共享同一组物理 Block，大幅节省显存

**注意**：PagedAttention 的分块存储意味着 KV Cache 在物理上是不连续的，vLLM 需要专门的 CUDA Kernel 来高效地从分散的 Block 中读取数据进行 Attention 计算。

### Offline

离线推理适合**不需要实时响应**的场景，如批量文本生成、数据标注、评测跑分等。vLLM 提供 `LLM`类作为离线推理的主入口

```
llm.generate()
```

### Chat 就是带有role的

```python
llm.chat(messages_list, sampling_params)

# 手动应用 chat template
texts = tokenizer.apply_chat_template(
    messages_list,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate(texts, SamplingParams(temperature=0.7))
```

会构造这种role，模型在 SFT（Supervised Fine-Tuning）阶段接触了大量按照 Chat Template 构造的训练数据

```
<|im_start|>user
1 + 1 = ?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

### Online

```
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 2
```

```
v1/chat/completions
```

| `POST /v1/completions`      | 文本补全 |
| --------------------------- | -------- |
| `POST /v1/chat/completions` | 对话补全 |

采样参数（SamplingParams）

| 参数                 | 说明                                     | 默认值       |
| :------------------- | :--------------------------------------- | :----------- |
| `temperature`        | 控制随机性，越高越随机                   | 1.0          |
| `top_p`              | 核采样，只从累计概率前 p 的 token 中采样 | 1.0          |
| `top_k`              | 只从概率最高的 k 个 token 中采样         | -1（不限制） |
| `max_tokens`         | 最大生成 token 数                        | 16           |
| `repetition_penalty` | 重复惩罚系数，>1 减少重复                | 1.0          |
| `stop`               | 遇到指定字符串时停止生成                 | None         |
| `n`                  | 为每个 prompt 生成的序列数               | 1            |
| `seed`               | 随机种子，设置后可复现结果               | None         |

**提示**：对于大多数生产场景，推荐 `temperature=0.7, top_p=0.9`作为起点。追求确定性输出时设 `temperature=0`。

| 参数                         | 说明                 | 推荐值           |
| :--------------------------- | :------------------- | :--------------- |
| `--model`                    | HuggingFace 模型标识 | -                |
| **`--tensor-parallel-size`** | 张量并行 GPU 数      | 根据模型大小决定 |
| `--gpu-memory-utilization`   | GPU 显存使用比例     | 0.9（默认）      |
| `--max-model-len`            | 模型最大序列长度     | 按需设置         |
| `--dtype`                    | 模型权重精度         | auto             |
| `--quantization`             | 量化方式             | None             |
| `--max-num-seqs`             | 最大并发序列数       | 256              |
| **`--enforce-eager`**        | 禁用 CUDA Graph      | 调试时使用       |

#### tensor parallel

对于大部分场景，**张量并行是首选**，配置最简单且效果最好

# day 3 PagedAttention

Attention 计算时，CUDA Kernel 通过 Block Table 拿到每个逻辑位置对应的 `physical_block_number`（物理块号）和 `physical_block_offset`（块内偏移），就能从分散的物理块里正确读到所有历史 KV。

传统 Attention 假设一个序列的 K、V 在显存里是**连续排列**的，Kernel 直接用一个基地址加偏移就能顺序扫过去。PagedAttention 打破了连续性，于是 Kernel 必须先查块表，把"逻辑位置"翻译成"物理地址"。核心是两个量：

- **`physical_block_number`（物理块号）**：决定去哪一个物理块里取数，乘以块的跨度（block stride）得到块的基址。
- **`physical_block_offset`（块内偏移）**：定位这个 Token 在块内的第几格。

vLLM 官方 Kernel 文档里，Key 指针的地址计算大致是这样一个形式（简化示意，非逐字照搬）：

```
// 根据物理块号 + 头偏移 + 块内偏移，算出这个 token 的 Key 起始地址
const scalar_t* k_ptr = k_cache
                      + physical_block_number * kv_block_stride   // 定位到物理块
                      + kv_head_idx           * kv_head_stride    // 定位到对应 KV 头
                      + physical_block_offset * x;                // 定位块内 token
```

间址查表的代价，就藏在这每次迭代的地址重算里——这也是下文"代价与权衡"要讨论的开销来源。

vLLM 官方这篇 Kernel 走读文档明确标注为"基于原始论文的历史文档，不再反映 vLLM 当前代码"。所以上面的指针公式请当作**理解原理的示意**，而非当前源码。真正想读实现，要去看 V1 引擎的 Attention 后端

### 内存共享与 Copy-on-Write

**跨请求共享前缀**：一批请求用了相同的 System Prompt，它们的前缀 KV 完全相同，就没必要各存一份——让它们的块表都指向同一批物理块即可。这种共享是**只读**的

**同一请求内的分叉**：从同一个 Prompt 并行采样多个回答（`n > 1`）或做 Beam Search 时，多个候选序列共享 Prompt 的 KV，但会在某个块**内部**开始分叉——这才是需要下面 CoW 的场景

**那如果共享同一个块的两个序列，后来生成的内容分叉了怎么办？**

* 只读共享时，大家指向同一块，靠**引用计数（ref_cnt）**记录有几个请求在用

- 当某个请求要往一个被共享的块里写新内容时，先把这个块复制一份成为它的私有块，在副本上写，再更新自己的块表指向副本。其他请求不受影响。

一般来说 COW 通常不是因为要修改已经生成的历史 Token 的 KV，而是因为要向尚未填满的共享 Block 中追加新 Token 的 KV

### Preemption

某一步 Decode 需要为若干请求追加新块，但空闲物理块不够。调度器会挑选部分请求"踢出"，释放它们占用的物理块。被踢出的请求，其已经算好的 KV Cache 有两种处理方式

| 恢复方式                    | 做法                                                         | 优点                    | 代价                                    |
| :-------------------------- | :----------------------------------------------------------- | :---------------------- | :-------------------------------------- |
| **换出（Swapping）**        | 把 KV Cache 从 GPU 显存拷贝到 CPU 内存暂存，恢复时再拷回来   | 不用重算，省算力        | 占用 CPU 内存，来回拷贝有 PCIe 传输开销 |
| **重计算（Recomputation）** | 直接丢弃 KV，恢复时把该请求已生成的 Token 当成新 Prompt 重新 Prefill 一遍 | 不占 CPU 内存，实现简单 | 要重跑一次 Prefill，浪费算力            |

**提示**：上表是 PagedAttention 论文和 V0 的两种做法。vLLM V1 的抢占只走重计算：被抢占的请求释放全部块、`num_computed_tokens` 清零后放回等待队列头部，恢复时重新 Prefill。官方调优文档的解释是，V1 架构下重计算的开销更低。开着 Prefix Cache 时，被释放的满块只要还没被淘汰，恢复时还能直接命中，实际要重算的部分会少一些。

**关键点**：抢占是 vLLM 在显存压力下**保证不 OOM、请求不失败**的兜底机制。如果你在日志里频繁看到 preemption 相关的告警，说明并发压得太满——要么调低并发、要么增大 `gpu_memory_utilization` 给 KV Cache 更多空间、要么加卡。它是”系统在硬扛”的信号，不是常态。

**提示**：绝大多数场景下，你**不需要手动改 `block_size`**——默认值已在多数模型上调优过。真正常动的旋钮是 `gpu_memory_utilization`（压榨显存）和 `enable_prefix_caching`（有共享前缀就开）。

### KVCacheManager vs BlockPool

Scheduler -> KVCacheManager -> BlockPool

- **`KVCacheManager`**（`kv_cache_manager.py`）：**请求视角**的管理者。调度器每一步都通过它给请求申请、释放块。核心方法和前面讲的概念一一对应：
  - `get_computed_blocks`：前缀缓存查询，返回该请求能命中的、已经算好的块，以及命中了多少 Token。这是第 6 节”共享前缀”的入口。
  - `allocate_slots`：为请求新增的 Token 申请槽位/块，顺带释放不再需要的块，并检查空闲块是否够用；不够就返回 `None`（交给调度器触发抢占）。它对应第 3 节的”按需逐块分配”。
  - `free`：请求结束时**逆序**释放它的块——让尾部的块先被回收，因为头部的块更可能被前缀缓存复用，留得越久越好。
- **`BlockPool`**（`block_pool.py`）：**全局视角**的物理块仓库，不关心哪个请求，只管”哪些块空着、哪些块被缓存着”。

`BlockPool` 里所有块存在一个普通列表 `self.blocks: list[KVCacheBlock]`（按 block id 索引），但**空闲块**单独用一个 `FreeKVCacheBlockQueue` 管理——它内部是一条**双向链表**，而不是简单的栈或 `deque`

原因是空闲块要同时支持两种操作：

- **分配时**从头部快速取走 N 个（`popleft_n`）；
- **前缀缓存命中时**，一个原本”空闲但仍缓存着有效内容”的块要被重新激活——需要从链表**中间任意位置**把它摘出来（`remove`）

普通队列没法 O(1)*O*(1) 地从中间删除，双向链表可以。前缀缓存要求”空闲块也可能携带可复用缓存”，所以 vLLM 选了双向链表

这里free block只代表没有running request在用它

因此，一个 Block 可以同时满足：

- `ref_cnt = 0`：没有活跃请求引用它，可以被重新分配。
- `block_hash != None`：它仍然记录着之前计算过的 KV Cache 内容，可以被其他请求复用。

所以 **空闲块也可能是一个有效的缓存块**。

```
class KVCacheBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None

        # 双向链表指针
        self.prev_free_block = None
        self.next_free_block = None
```

这里一个block对象，之前prev和next，就可以直接O(1)删除了

1. **释放不等于清空**：`ref_cnt` 归零的块只是回到空闲链表，**内容还在**。如果后续请求的前缀哈希命中它，`touch()` 直接把它复活复用，这就是 Automatic Prefix Caching 省掉重复 Prefill 的底层机制。
2. **带哈希的块留得更久**：释放时，vLLM 把”没缓存价值的块”放链表头（优先被再分配覆盖，源码注释说这样还能让刚释放的块被立即复用，GPU 局部性更好），“带哈希、有复用潜力的块”放链表尾（晚点才被淘汰）。分配总从链表头取，于是带哈希的块按释放先后被淘汰，效果就是 LRU。

**V1 的 BlockPool 里没有 CoW。** 上面三个函数只做 `ref_cnt` 的增减和空闲链表的摘挂，没有任何”复制块”的逻辑。V1 能绕开 CoW，靠的是两点设计：

- **只共享写满的块**：前缀缓存只对已写满的块计算哈希并共享，未写满的尾块不参与共享；新生成的 Token 一律追加到请求自己的新块里。因此跨请求的前缀共享始终是只读的，没有”往共享块里写”的时刻。
- **`n > 1` 拆成独立子请求**：V1 把并行采样拆成 `n` 个 `n = 1` 的子请求（见 `vllm/v1/engine/parallel_sampling.py` 中的 `ParentRequest`），子请求之间通过前缀缓存共享 Prompt 的满块，不再需要块内分叉。

```python
# 伪代码：还原 BlockPool 的关键逻辑（非逐字源码）
def get_new_blocks(self, num_blocks):
    if num_blocks > self.get_num_free_blocks():
        raise ValueError("空闲块不足")          # 触发上层抢占
    blocks = self.free_block_queue.popleft_n(num_blocks)
    for blk in blocks:
        assert blk.ref_cnt == 0                 # 新分配的块必须无人引用
        blk.ref_cnt += 1
    return blocks

def free_blocks(self, ordered_blocks):
    for blk in ordered_blocks:
        blk.ref_cnt -= 1
    to_free = [b for b in ordered_blocks if b.ref_cnt == 0]   # 没人用了才真正回收
    no_hash = [b for b in to_free if b.block_hash is None]
    has_hash = [b for b in to_free if b.block_hash is not None]
    self.free_block_queue.prepend_n(no_hash)    # 没有缓存价值：放链表头，最先被复用
    self.free_block_queue.append_n(has_hash)    # 带哈希：放链表尾，最晚被淘汰

def touch(self, blocks):                        # 前缀缓存命中时调用
    for blk in blocks:
        if blk.ref_cnt == 0:                    # 从空闲链表里"抢救"回来
            self.free_block_queue.remove(blk)
        blk.ref_cnt += 1

```

# day 4 Continuous Batching

批处理提升的是**吞吐**，而且几乎不牺牲单请求延迟——因为瓶颈在搬权重，多算几个 Token 的计算量对 Memory Bound 的 Decode 来说近乎”免费”。

1. **收新请求**：把新到达的请求放进 Waiting（等待）队列。
2. **调度决策**：调度器决定这一步谁上车。它要同时兼顾两拨请求——正在生成的 Running 队列请求（做 Decode），和等待队列里能塞进来的新请求（做 Prefill）。是否有空间，取决于显存里还有多少空闲 KV Block（这正是 PagedAttention 提供的信息）。
3. **前向计算一步**：对选中的这批请求跑一次模型前向，每个 Decode 请求产出 1 个新 Token，Prefill 请求产出首 Token。
4. **处理产出**：把新 Token 追加到各请求的 KV Cache；检查哪些请求触发了 EOS 或长度上限，将其标记完成、**释放其占用的 KV Block** 归还给空闲池。
5. **循环**：回到第 1 步。

**提示**：当显存不够、Running 队列里请求太多时，调度器还会做**抢占（Preemption）**：暂停部分请求、释放它们的 KV Block，腾出显存保证其他请求推进。vLLM V1 被抢占的请求一律丢弃 KV、恢复时重算，第 6.3 节会看到对应的代码。

| 技术                | 解决的问题         | 提供的能力                   |
| :------------------ | :----------------- | :--------------------------- |
| PagedAttention      | 显存碎片、利用率低 | 细粒度、可即时回收的显存分配 |
| Continuous Batching | GPU 空转、排队延迟 | 迭代级调度、动态批次构成     |

**PagedAttention 管”显存怎么放”，Continuous Batching 管”请求怎么调度”，二者相乘才有 vLLM 的高吞吐**

V1 相比 V0 最大的设计变化是——**调度器眼里不再有”Prefill 阶段”和”Decode 阶段”之分**。源码注释里明确写了 “no ‘decoding phase’ nor ‘prefill phase’“：每个请求只是不停地把 `num_computed_tokens`（已计算 Token 数）推向 `num_tokens_with_spec`（含投机 Token 的总目标数）。这种统一视角，正是 2.4 节 Chunked Prefill 能把 Prefill 和 Decode 混在一步里跑的前提。

### scheduler

```
                Scheduler.schedule()
                       |
          +------------+------------+
          |                         |
       RUNNING                    WAITING
          |                         |
     分配计算资源               尝试加入 Batch
          |                         |
     KV 不足 → Preempt       资源足够 → RUNNING
          |                         |
          +------→ WAITING          |
                                    |
                          参与 Prefill / Decode
```

- **`self.running`**：一个列表，装当前正在计算的请求。
- **`self.waiting`**：一个**优先级队列**，装排队等待的请求。调度策略有两种——`FCFS`（先来先服务）和 `PRIORITY`（按优先级）。v0.32.0 里还有几个辅助队列：`kv_holding_waiting`、`deferred_waiting`，以及每步临时使用的 `step_skipped_waiting` 等，用来暂存本步因依赖未就绪（如异步 KV 传输尚未完成）而不能调度的请求。读主干逻辑时可以先忽略它们。

”调度 → 前向 → 处理产出”循环：

- **`schedule()`**：一步调度的核心。它给一个 **token 预算（`token_budget`）**，先遍历 `running` 队列给正在生成的请求分配 Token 和 KV 块，预算还有富余、且没发生抢占时，再从 `waiting` 队列拉新请求进来。最后打包成 `SchedulerOutput` 交给模型执行。
- **`update_from_output()`**：拿到模型这一步的输出后，把采样出的 Token 追加回各请求、检查停止条件（EOS/长度上限）、释放已完成请求的资源，并组装成返回给客户端的结果。

```python
# 伪代码：还原 V1 Scheduler.schedule() 的主干逻辑（非逐字源码）
def schedule(self):
    token_budget = self.max_num_batched_tokens
    scheduled = []
    preempted = []

    # 第一优先级：喂饱正在运行的请求（Decode 为主）
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        req = self.running[req_index]
        num_new = compute_new_tokens(req, token_budget)   # 受预算/max_model_len 限制
        while True:                                       # 申请不到就抢占，再重试
            blocks = self.kv_cache_manager.allocate_slots(req, num_new)
            if blocks is not None:
                break
            victim = self.running.pop()                   # FCFS 踢最后一个（最晚加入的）
            self._preempt_request(victim)                 # 释放块，打回 waiting 队头
            preempted.append(victim)
            if victim is req:                             # 踢到自己，本请求这步不跑了
                break
        if blocks is None:
            break
        scheduled.append(req)
        token_budget -= num_new
        req_index += 1

    # 第二优先级：预算有富余且本步没发生抢占，才拉新请求做 Prefill
    if not preempted and token_budget > 0:
        while self.waiting and token_budget > 0:
            req = self.waiting.pop_front()
            if not self._can_admit(req):                  # 检查 running 上限/LoRA 等
              
              break
            self.running.append(req)
            scheduled.append(req)

    return SchedulerOutput(scheduled)

```

1. **在途请求优先于新请求**：先保证已经在跑的请求继续推进，剩余预算才用于接纳新请求。这避免了”新请求不断插队、老请求迟迟跑不完”的饥饿。
2. **抢占发生在显存耗尽时**：当 `allocate_slots` 返回 `None`（申请不到 KV 块），调度器就挑一个”受害者”抢占，释放它的块后**重试分配**，直到成功或者**踢到当前请求自己**为止。FCFS 策略下踢掉 `running` 里的最后一个（最晚加入的），PRIORITY 策略下踢 `(priority, arrival_time)` 最大的那个，即优先级最低、同优先级里来得最晚的请求。



被抢占的请求在 `_preempt_request()` 里经历一套清理：释放它的 KV 块和编码器缓存 → 状态置为 `PREEMPTED` → **`num_computed_tokens` 归零** → 重新塞回 `waiting` 队列的**队头**，等有资源了优先恢复。

**关键点**：注意 `num_computed_tokens` 归零这一步——它意味着 V1 的抢占恢复策略是**重计算（Recompute）**：被抢占的请求下次调度时，从头把 Prompt 和已生成的 Token 重新 Prefill 一遍，而不是从 CPU 换回 KV（这与 2.1 节讲的两种恢复方式呼应）。官方调优文档说明 V1 默认用重计算而非换出，因为在 V1 架构下重计算开销更低；开着 Prefix Cache 时，被释放的满块若还没被淘汰，恢复时可以直接命中。

**提示**：读调度器源码时抓住这条主线——**`waiting`/`running` 两队列 + `token_budget` 预算 + `allocate_slots` 返回值驱动抢占**。第 4 节那个抽象的”调度循环”，落到代码就是这三者的相互作用。更深入的调度细节（Chunked Prefill 如何切分）见 2.4 节；第 3 章会用 nano-vllm 的精简 Scheduler 再走一遍同样的流程。



Continuous Batching 虽好，但调度器面对的是一组相互冲突的目标，工程上要清楚这些权衡：

- **吞吐 vs 延迟**：一步里塞进的请求越多，系统吞吐越高，但每步的前向计算耗时也越长，单请求的 TPOT 会略微变差。批次大小需要在两者间取平衡。
- **Prefill vs Decode 的争抢**：新请求的 Prefill 是 Compute Bound 的重活，如果某一步塞进一个超长 Prompt 的 Prefill，会明显拖慢同批 Decode 请求的出字速度（TPOT 抖动）。这个问题正是下一节 **Chunked Prefill** 要解决的——把长 Prefill 切块，避免它一次性霸占整个 step。
- **公平性 vs 效率**：先来的请求是否优先？长请求会不会被短请求反复插队饿死？调度策略需要在整体效率和请求间公平之间做取舍。

**注意**：Continuous Batching 让延迟指标变得**对负载敏感**。低负载时批次小、延迟低；高负载时批次变大、排队变长，P95/P99 尾延迟会明显上升。做容量规划时，一定要在**目标并发下**压测延迟，而不是看空载数据。

# Mark

**sheduler的running，waiting**

Waiting 是queue， running是list，然后waiting queue左老右新，append右边，peek左边，popleft 左边，preempt的req appendleft

running抢占pop尾部，加入也是append 尾部

running 没有用链表来做到 O(1) 删除，因为调度时要按下标遍历，抢占时还要从尾部 pop。n 本来就很小，用 list 写起来简单，缓存局部性也好



这里nano-vllm 和 vllm v1里

prefill 影响首字延迟，decode 影响出字间隔。

### nano-vllm：一步只能跑一种，

每一步都要做选择：这一步给新请求做 prefill，还是给老请求做 decode？

如果选"先 decode"会怎样？ running 里只要还有序列，每一步都会是 decode，因为它们每步都有活干。新请求只能等 running 里所有序列都生成完才能进来，可能要等几百步。新请求的首字延迟非常长，而且 batch 越跑越小，GPU 越来越空闲。

所以只能"先 prefill"：有新请求就先把它 prefill 完，让它加入 running，之后大家一起 decode。代价是做 prefill 的那几步里，老请求停着不动，出字会卡一下。  

### vLLM v1：一步能混着跑，所以两种都能做

既然一步里两种都能放，问题就变成了：token 预算有限时，先满足谁？

decode 很便宜：每条序列只要 1 个 token。running 里有 50 条序列，也只用掉 50 个预算。prefill 很贵：一个 prompt 可能就要几千个 token。

所以 v1 先把 running 里每条序列的 1 个 token 扣掉，保证老请求每步都能出一个字；剩下的大部分预算再给新请求做 prefill，放不下就分块。

结果是两边都照顾到了：老请求每步都前进，新请求也在同一步里开始 prefill。这里"先 decode"的意思不是让新请求等着，而是先扣掉decode的部分预算

# day 5 Prefix Cache & RadixAttention

前缀能复用的前提，是显存管理支持”多个请求指向同一份 KV”

- KV Cache 以**固定大小的块**存储，请求通过块表间接引用物理块。
- 只要两个请求的某些前缀块内容相同，就让它们的块表**指向同一个物理块**，配合**引用计数**记录有几个请求在用。
- 新生成的内容写在各请求自己的新块里，共享块只读。vLLM V1 只共享已写满的块，因此跨请求复用不会出现”往共享块里写”的情况，也就不需要 Copy-on-Write。

所以 Prefix Cache 本质上是**把”共享前缀块”这件事从’同时并存的请求之间’扩展到’先后到达的请求之间’**：前一个请求算完的前缀块，先不急着扔，留在缓存里；后来的请求如果前缀匹配，直接复用这些块，跳过对应的 Prefill 计算。

**核心概念**：**Prefix Cache = 跨请求、跨时间地复用前缀 KV 块。** PagedAttention 的块共享是空间维度（并存请求共享），Prefix Cache 把它延伸到时间维度（先后请求复用）。

### vllm APC

vLLM 的方案叫 **Automatic Prefix Caching（APC，自动前缀缓存）**。“Automatic”是它的一大亮点——**你不需要手动声明哪段是共享前缀，引擎自动检测并复用**。它靠的是对 KV 块做哈希。

```
block_hash=Hash(parent_hash, block_tokens, extra)
```

- extra（如 LoRA ID、多模态输入的哈希、cache salt 等）

把”父块哈希”纳入计算是精髓所在：它保证了**只有从序列开头到当前块的整条前缀都完全一致，哈希才会相同**。这样就精确刻画了”前缀相同”这件事

vLLM 自 v0.11 起默认使用 `sha256` 作为哈希算法，也可通过 `--prefix-caching-hash-algo` 切换为 `sha256_cbor`（跨语言可复现）、`xxhash`、`xxhash_cbor` 等。非加密算法（如 xxhash）更快，但碰撞风险略高。

新请求进来时，vLLM 逐块计算前缀哈希，拿去缓存表（hash → block ID 的映射）里查：

- **命中**：对应的物理块还在缓存里，直接把请求的块表指向它、增加引用计数，**跳过这部分 Prefill**。
- **未命中**：正常 Prefill 计算，算完后把新块的哈希登记进缓存，供后续请求复用。

**关键点**：命中前缀缓存能**直接砍掉那部分 Prefill 的计算**，最直接的收益是**大幅降低 TTFT**（首 Token 更快出来），同时省下的算力还能服务更多请求。共享前缀越长、命中率越高，收益越大。

#### V1 的”零开销”设计

早期版本（V0）里，前缀缓存因为命中率低时 CPU 记账开销不划算，默认是关闭的。vLLM **V1 引擎**对此做了重写：所有块在初始化时预分配成一个块池，用嵌入块内的双向链表指针做 O(1) 的移动与淘汰，把记账开销压到极低。官方设计文档称前缀缓存”几乎是免费的午餐”（almost a free lunch），V1 发布博客给出的数据是：即使命中率为 0%，吞吐下降也不到 1%。因此 **V1 中前缀缓存默认开启**。

**为什么 V0 的 Prefix Caching 开销比较大**

如果未来没有任何 Request 复用这些 KV Cache，那么这些 Hash 计算、元数据维护和缓存管理就是额外开销。

而 vLLM 的 Scheduler 运行在 CPU 上，对于高吞吐、短 Request 场景，这些 Python 操作可能成为瓶颈

**V1：把所有 KVCacheBlock 元数据提前创建好**

首先需要区分：

- **GPU KV Cache 内存**：真正存储 Key、Value Tensor。
- **CPU KVCacheBlock 对象**：存储 Block ID、Hash、Ref Count 和链表指针等元数据。

V1 初始化时就建立固定数量的 `KVCacheBlock` 对象：

```python
self.blocks = [
    KVCacheBlock(block_id=i)
    for i in range(num_gpu_blocks)
]
```

后续 Request 到来时，主要是在已有的 Block 对象上更新状态，而不是不断新建 Python Block 对象。

这就减少了对象分配、垃圾回收和内存管理的开销

**V1 的 `KVCacheBlock` 包含两个指针：**

```python
class KVCacheBlock:
    block_id: int
    block_hash: int | None
    ref_cnt: int

    prev_free_block: "KVCacheBlock | None"
    next_free_block: "KVCacheBlock | None"
```

这样每个 Block 对象本身就能成为链表节点，不需要再创建额外的包装节点。

### cache

1. 最近最少使用的 Block 优先淘汰。
2. 如果 Block 的最近使用时间相同（例如属于同一个 Request），则前缀更长的 Block 优先淘汰。

- **正在被使用的块不能淘汰**：引用计数 > 0 的块（还有请求在用）受保护。
- **空闲块按 LRU 淘汰**：请求结束后，它的块引用计数归零、进入空闲队列，但**内容和哈希先保留着**——万一马上有匹配的新请求来，还能命中复用。当显存吃紧需要腾地方时，才真正回收最久没被用到的空闲块。

一个有意思的细节：vLLM 释放块时，是把请求的块**按逆序**放进空闲队列尾部。因为一个请求的**最后一个块**哈希了最多的 Token（前缀最长），最不容易被别的请求命中复用，所以让它**优先被淘汰**；而靠前的块（短前缀，如共享 System Prompt 的开头）更可能被复用，就让它们更晚淘汰。

**提示**：这个”逆序入队”的小设计体现了一个朴素直觉——**越靠近序列开头的前缀，越通用、越值得留**；越靠后的前缀越个性化，留着也难命中。

## SGLang  RadixAttention

vLLM 的 Hash 方案高效，但它有个隐含限制：**匹配是”块对齐”且”线性前缀”的**——按块边界比对一条从头开始的前缀。而多轮对话、树状采样等场景里，请求的共享关系是**树状**的：一个共同前缀分叉出多个不同的后续。

SGLang 提出的 **RadixAttention** 用一棵 **Radix Tree（基数树/压缩前缀树）** 来管理 KV Cache 的复用：

- 树的每条边代表一段 Token 序列，从根到某节点的路径就是一段前缀，对应缓存的 KV。
- 新请求到来时，沿树做**最长前缀匹配**，匹配到的路径部分直接复用 KV，只对分叉之后的新内容做计算。
- 分叉天然对应”共同前缀 + 不同后续”，非常契合多轮对话（同一段历史派生多轮）和并行采样（同一 Prompt 采样多个回答）。
- 每个节点带一个**引用计数**，记录有几个运行中的请求正在用它；只有引用计数为 0 的节点才可淘汰。淘汰按 LRU 进行，并且**先淘汰叶子**，这样祖先节点上的公共前缀能留得更久。
- 调度时优先处理与缓存共享前缀最长的请求（论文称为 cache-aware scheduling），在同样的缓存容量下提高命中率。

Radix Tree 的特点是路径压缩

好处

* 粒度更细

* 显式树结构方便管理分叉和淘汰

* Cache-Aware Scheduling

  * SGLang 的 Cache-Aware Scheduling 正是考虑请求之间的前缀相似性，在适合的场景下把能够共享 KV 的请求安排得更接近。

    这属于**调度优化，而不只是 Radix Tree 数据结构本身的优化**

  * ACB，执行 C 可能把 A 中有价值的 `[System][Tool A]` 缓存淘汰掉，等 B 运行时，就可能需要重新 Prefill

  * ABC，B 可以更早利用 A 已经计算过的前缀 KV，从而降低 Cache Thrashing

  * 也就是说这里是scheduler单线程处理的，比如处理A 再C 再B，然后前缀匹配主要在cpu上完成，

> **为什么一个可以并行化的 CPU 操作，推理框架却选择串行执行？**
>
> 核心原因有三个：
>
> 1. **单次 Prefix Matching 通常足够快**，并行带来的收益可能小于线程调度开销。
> 2. **Radix Tree 不是纯只读数据结构**，匹配时可能修改树，导致并发控制复杂。
> 3. **Scheduler 的关键目标是快速组织 GPU Batch**，而不只是最大化 CPU 核心利用率。

不要把它们对立起来看。vLLM 的方案在绝大多数场景已经很够用，且随 V1 做到了近乎零开销；RadixAttention 是在**高度树状复用**的负载下（如复杂 Agent、大量并行采样）展现出额外优势。SGLang 论文在这类结构化 LLM 程序负载上报告了最高 6.4 倍的吞吐提升，这个数字来自 RadixAttention 与其他优化的组合。选型时看你的负载形态，而非简单认为”树一定比哈希好”。

- **CPU**：保存 Radix Tree 的树结构、Token Key、引用计数和访问时间。
- **CPU**：执行最长前缀匹配（Longest Prefix Matching）。
- **GPU**：存储实际的 KV Cache Tensor，并执行 Attention。
- **CPU/GPU 之间**：通过 KV Index 将匹配到的前缀映射到对应的 KV 存储位置。

`BlockPool` 初始化时会预建几套结构协同工作（呼应 2.1 节的源码走读）：

| 结构                             | 作用                                                         |
| :------------------------------- | :----------------------------------------------------------- |
| Block Pool（块列表）             | 所有物理块的池子，进程启动时一次性预分配，避免运行时创建 Python 对象的开销 |
| Free Block Queue（空闲双向链表） | 管理空闲块，支持 O(1)*O*(1) 从中间摘除（前缀命中时”抢救”块） |
| Cache（哈希 → 块 ID 映射）       | 前缀查表的核心：拿块哈希来查有没有可复用的物理块             |
| Request（请求 → 块 ID 映射）     | 记录每个请求当前持有哪些块                                   |

复用发生在调度器为新请求调用 `get_computed_blocks()` 时：把 Prompt 逐块哈希，去 Cache 表里查命中；命中的块被 `touch()`（引用计数 +1，并从空闲链表摘出以防被淘汰），随后 `allocate_slots()` 只为未命中的部分申请新块。

**关键点**：V1 的块表是**只追加（append-only）**的——同哈希的重复满块可能短暂共存（因为块表不能改写），这些冗余”会在请求释放时被消除”。这是 V1 为了调度简洁性做的一个务实取舍，读源码时看到重复块不必惊讶。

> **vLLM 的 Prefix Cache 只能复用当前 Request 已经确定的 Token 前缀，不能提前复用其他 Request 生成的未来 Token。**
>
> 即使生成相同的后面token也不会替换，正常执行过程中不会因为发现重复就随意替换已有物理 Block ID，所以这里append-only设计会导致重复，然后这里hash_block会覆盖，也就是一个hash只对应一个block_id，然后req完了后会调用deallocate释放

淘汰则是第 4 节讲的 LRU：需要腾地方时，从空闲链表**头部**弹出最久未用的块，把它的哈希从 Cache 表里删掉。而释放请求时块**逆序入队尾**，让哈希了最多 Token 的尾块（最难命中）优先被淘汰。

**前缀缓存收益大的场景**：

- 长 System Prompt / 长 Few-shot 且被大量请求共享
- 多轮对话（历史不断累积复用）
- 同一 Prompt 并行采样多个输出（`n > 1`）
- RAG 中固定的指令模板 + 变化的检索内容（固定部分放前面才能命中）

**收益有限甚至无收益的场景**：

- 每个请求前缀都不同（如无模板的自由问答）
- 共享部分放在了 Prompt**中间或末尾**——前缀缓存只能复用**从头开始连续相同**的部分，共享内容一旦不在开头就无法命中

**关键点**：想吃到前缀缓存的红利，**Prompt 工程上要有意识地”把公共内容前置”**——固定不变的指令、示例、上下文放最前面，变化的用户输入放最后。这是一条几乎零成本却常被忽视的优化。

**注意**：前缀缓存**不改变模型输出**（复用的是数学上完全等价的 KV），可以放心开启。但要留意多租户环境下的隔离——vLLM 提供 `cache_salt` 等机制，避免不同租户/权限的请求意外命中彼此的缓存造成信息泄露。

# day 6 Chunked Prefill & Token Budget 

**Chunked Prefill 用”把大块 Prefill 摊到多步”换来”Decode 的平稳”**，本质是把一次性的计算尖峰削平成多个可控的小峰，从而稳住 TPOT 尾延迟

切块会让这个长 Prompt 的 **TTFT 略微变大**（它要跨多步才 Prefill 完，首 Token 出得稍晚）。这是一个刻意的权衡——**牺牲少量单请求 TTFT，换取整批 Decode 的 TPOT 稳定**。在追求平稳出字体验的服务里，这笔交易通常很划算。

Chunked Prefill 要落地，调度器需要一个统一的”度量衡”来决定每一步塞多少活。这个度量衡就是 **Token Budget（Token 预算）**。

核心洞察是：**无论 Prefill 还是 Decode，对 GPU 来说本质都是”处理若干个 Token 的前向计算”**。

- 一个做 Decode 的请求，这一步贡献 **1 个 Token**（生成下一个）。
- 一个做 Prefill 的请求（或它的一个 Chunk），这一步贡献 **N 个 Token**（这一块的长度）。

于是调度器给每一步设定一个总的 Token 预算上限（比如”这一步最多处理 2048 个 Token”），然后往里面装请求：先保证正在生成的 Decode 请求（每个占 1 个 Token 额度），剩下的预算再分给 Prefill 的 Chunk。预算用完，这一步就发车。
$$
[
\underbrace{\sum{\text{Decode 请求}}1}_{\text{每个请求 1 Token}}
+
\underbrace{\sum{\text{Prefill Chunk}}N_i}_{\text{每个 Chunk }N_i\text{ Tokens}}
\leq
\text{Token Budget}
]
$$
用 Token 预算这把统一的尺子，“Chunk 该切多大""一步能塞几个 Decode""长 Prefill 和 Decode 怎么共处一步”这些问题就都归结成了同一个装箱问题——**在预算内尽量把 GPU 装满，同时优先保证 Decode 不断流**。

#### V1 统一调度

```python
# V1 调度器每一步的决策，本质是这样一个映射
{
    "req_A": 1,      # 正在 Decode，处理 1 个新 token
    "req_B": 1,      # 正在 Decode，处理 1 个新 token
    "req_C": 512,    # 长 Prompt 的一个 Prefill chunk
    "req_D": 128,    # 短 Prompt，一步就能 Prefill 完
}
# 约束：所有 value 之和 ≤ token_budget
```

**V1 用 `{request_id: num_tokens}` 这一个统一表示，同时涵盖了 Chunked Prefill、Prefix Cache 和 Speculative Decoding。** 因为它们本质都是”某个请求这一步要处理多少 Token”的不同取值——Prefill chunk 是一个较大的值，Decode 是 1，投机解码验证是若干个候选 Token。一套调度框架，优雅地统一了所有情况。

这也是为什么 V1 能把 Chunked Prefill 做成**默认行为**：既然调度天然以 Token 为单位、不区分阶段，长 Prompt 自然就被预算约束切成了 Chunk，无需额外的特殊逻辑。

```python
# 伪代码：还原 schedule() 里"预算截断即切块"的逻辑（非逐字源码）
def schedule(self):
    token_budget = self.max_num_batched_tokens
    scheduled = {}

    # 正在运行的请求：Decode 每个只要 1 个 token，也可能是上一步没填完的 prefill 续块
    for req in self.running:
        if token_budget == 0:
            break
        # 这个请求还差多少 token 才算完（prefill 未完成部分 或 decode 的 1 个）
        need = req.num_tokens_with_spec - req.num_computed_tokens
        # 关键：被剩余预算截断 → 长 prefill 在这里被"切"出一个 chunk
        num_tokens = min(need, token_budget)
        blocks = self.kv_cache_manager.allocate_slots(req, num_tokens)
        if blocks is None:
            ...                                  # 显存不足：抢占并重试，细节见 2.2 节 6.2
            continue
        scheduled[req.id] = num_tokens          # {request_id: num_tokens}
        token_budget -= num_tokens

    # 预算有剩，再从 waiting 拉新请求（同样会被预算截断成首个 chunk）
    # 源码里本步若发生过抢占，就跳过这一段，不接新请求
    while self.waiting and token_budget > 0 and self._can_admit():
        req = self.waiting.peek_front()
        # 先查前缀缓存：命中的 token 不用再算（见 2.3 节）
        _, num_cached = self.kv_cache_manager.get_computed_blocks(req)
        need = req.num_tokens - num_cached
        num_tokens = min(need, token_budget)     # 长 prompt 首个 chunk 也在此截断
        if self.kv_cache_manager.allocate_slots(req, num_tokens) is None:
            break                                # 显存不够就不再接新请求
        self.waiting.pop_front()
        self.running.append(req)
        scheduled[req.id] = num_tokens
        token_budget -= num_tokens

    return SchedulerOutput(scheduled)

```

Eg：

预算 2048：8 个 Decode 请求各取 1 个（蓝色）；running 里的长请求 R 还差 1440 个 Token 没 Prefill，一次拿完（黄色）；剩下的 600 分给 waiting 里 900 Token 的新请求 W1，它被截成首个 Chunk（绿色）；W2 没分到预算，留在队列里（灰色虚线）。

1. **`min(need, token_budget)` 就是切块的本质**：一个 4000 Token 的 Prompt，若这一步只剩 512 预算，就只 Prefill 512，剩下 3488 留到后续步——它自动成了一个 Chunk，没有任何专门的”分块器”。
2. **续块靠 `num_computed_tokens` 记账**：请求的”已计算到第几个 Token”被持久记录，下一步从断点继续，直到 `num_computed_tokens` 追上 `num_prompt_tokens`（Prefill 完成）后自然转入每步 +1 的 Decode。这正是 2.2 节”不分 Prefill/Decode 阶段”在数据上的体现。

**关键点**：理解 V1 调度的精髓就一句话——**`num_computed_tokens` 一路推向 `num_tokens_with_spec`，每步推进多少由 `min(剩余需求, 剩余预算)` 决定**。Chunked Prefill、Decode、投机解码验证，都只是这个式子在不同请求上取了不同的值。

V1 在条件允许时默认开启 Chunked Prefill，并且组批时优先安排 Decode 请求，一般不需要手动打开：

| 参数                           | 含义                                                         | 调优方向                                                     |
| :----------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `max_num_batched_tokens`       | 一步能处理的最大 Token 数（即 Token Budget）                 | 调大 → 吞吐↑、单步耗时↑（TPOT 可能↑）；调小 → TPOT 更稳、吞吐可能↓ |
| `max_num_seqs`                 | 一步最多容纳的请求数（批次上限）                             | 受显存/KV Block 约束，与上者共同决定实际批大小               |
| `long_prefill_token_threshold` | 超过这个长度的 Prompt 被视为”长 Prefill”，单步处理的 Token 数被限制在该值以内 | 默认 0（不限制）；混跑大量长 Prompt 时设一个值，防止单个长请求吃掉整步预算。批里只有它一个请求时不生效 |

调优的核心是围绕 `max_num_batched_tokens` 做取舍：

- **偏吞吐**：把预算调大，一步塞更多 Token，GPU 装得更满，系统吞吐更高，但长 Prefill 占的比重上升，Decode 的 TPOT 稳定性会下降。
- **偏延迟稳定**：把预算调小，每步计算量更可控，Decode 出字更平稳、TPOT 尾延迟更低，但单步能干的活变少，峰值吞吐可能受限。

vLLM 官方调优文档给出了一个参考刻度：较小的值（如 2048）能得到更好的 ITL（Inter-Token Latency，相邻两个输出 Token 的间隔，与 TPOT 基本同义），因为拖慢 Decode 的 Prefill 更少；较大的值能缩短 TTFT，因为一步能 Prefill 更多 Token；追求吞吐时建议设到 8192 以上，小模型跑在大显存 GPU 上时尤其如此。

**提示**：没有放之四海皆准的最优值。正确做法是**在你的目标负载（真实的 Prompt 长度分布、并发量）下压测**，观察 TTFT、TPOT P95/P99 和吞吐三条曲线，找到满足 SLO 的最大吞吐点。具体调参会在第 3 章”vLLM 关键配置调优”里结合实例展开。

**Chunked Prefill 收益明显的场景**：

- Prompt 长度差异大、且有超长 Prompt 混入（长文档问答、长上下文 RAG）
- 对出字流畅度敏感、要压 TPOT 尾延迟的在线服务（如实时对话）
- Prefill 和 Decode 请求混合的高并发在线负载

**收益有限的场景**：

- 纯离线批量推理、只追求总吞吐而不在乎单请求延迟抖动——此时切块带来的调度开销未必划算
- Prompt 都很短、本来就不会有长 Prefill 独占一步的情况

**关键点**：Chunked Prefill 优化的是 **Prefill 与 Decode 混跑时的相互干扰**，它是”平稳性”优化而非”峰值吞吐”优化。判断要不要用、怎么调，关键看你的服务**是否在乎 TPOT 的稳定性**，以及负载里**有没有长 Prompt**。

**注意**：Chunked Prefill 与 Prefix Cache 是天然搭档——前缀命中先砍掉一部分 Prefill 计算，剩下未命中的部分再切块掺入 Decode。二者叠加，长 Prompt 场景下的 TTFT 和 TPOT 能同时得到照顾。

# Attention 后端与图优化

FlashAttention/FlashInfer/FlashMLA/Triton

图优化：用 CUDA Graph 和 torch.compile 消除 Decode 阶段的 CPU 启动开销

**同样一批请求的一次前向计算，怎么让它跑得更快？**

* 一是选对高效的 **Attention 后端 Kernel**，
* 二是用 **CUDA Graph + torch.compile** 消除 Decode 阶段那些看不见却致命的 CPU 开销。这也是 vLLM V1 相比 V0 提速的重要来源

但当模型不大、Batch 也不大时，还有一个更隐蔽的瓶颈会冒出来：**CPU 发射 Kernel 的开销**。

GPU 自己不会主动干活，它每做一个操作（一次矩阵乘、一次 LayerNorm、一次激活……），都要由 CPU 通过一次”启动（launch）“命令告诉它。跑一遍 Transformer，有成百上千个这样的小操作，就对应成百上千次 CPU→GPU 的启动调用。每次启动本身有固定的 CPU 开销（构造参数、驱动调度、提交队列），量级在微秒级。

平时这点开销不算什么——因为 Prefill 一步要处理成千 Token，每个 GPU Kernel 一跑就是几百微秒甚至毫秒，CPU 那点启动开销被淹没了。但 **Decode 每步只处理 1 个（或很少几个）Token**，每个 Kernel 的实际计算可能就几微秒，结果**CPU 发射这个 Kernel 的时间比 GPU 执行它的时间还长**。于是 GPU 干几微秒就停下来，眼巴巴等 CPU 发下一条命令——算力再次闲置，只不过这次不是等数据，是**等指令**。

Kernel越短空等占比越高

Decode 阶段有两重”等待”——**等数据**（Memory Bound，靠 Batching/量化缓解）和**等指令**（CPU launch 开销，靠本节的图优化缓解）。小模型、小 Batch 时，后者尤其突出。这就是为什么图优化对 Decode 加速格外有效。

| 后端                     | 特点                                                         | 典型适用                                             |
| :----------------------- | :----------------------------------------------------------- | :--------------------------------------------------- |
| **FlashAttention**       | IO 感知的经典高性能实现，vLLM 按架构选用 FA2（Ampere）、FA3（Hopper）或 FA4（Blackwell） | Ampere、Hopper 上的默认后端                          |
| **FlashInfer**           | 面向推理优化，KV Cache 处理灵活，支持多种数据类型            | Blackwell 上的默认后端；高吞吐服务、需要灵活 KV 布局 |
| **FlashMLA** 等 MLA 后端 | 针对 MLA（Multi-head Latent Attention）优化                  | DeepSeek 等采用 MLA 结构的模型                       |
| **Triton**               | 用 Triton 语言实现，可移植性好                               | 非 NVIDIA 硬件或需自定义时的兜底                     |

**提示**：多数情况下不用手动指定，vLLM 会根据 GPU 架构、模型类型和数据精度**自动挑选**合适的后端。v0.32.0 在 Ampere 和 Hopper 上默认选 `FLASH_ATTN`，在 Blackwell 上默认选 `FLASHINFER`；MLA 模型会走专门的 MLA 后端。需要显式干预时，用 `--attention-backend` 指定后端，用 `--attention-config.flash_attn_version` 指定 FlashAttention 版本（2、3 或 4）。后端支持矩阵随版本演进较快，具体某张卡、某个模型用哪个后端，以你所用版本的官方文档为准。

- **FlashAttention：通常是 Prefill-heavy 场景值得优先测试的选择。**
- **FlashInfer：通常是 Decode-heavy、长 Context、高并发 Serving 场景值得优先测试的选择。**

| Workload              | 优先测试       | 原因                                    |
| --------------------- | -------------- | --------------------------------------- |
| 长 Prompt、大 Prefill | FlashAttention | 大 Query Tile，Tensor Core 利用率高     |
| 短 Prompt、长 Decode  | FlashInfer     | 专门优化 Paged KV 和 Decode             |
| 高并发 Decode         | FlashInfer     | 可利用针对 Batch Decode 的专用 Kernel   |
| 长 Context Decode     | FlashInfer     | Split-KV 等策略可能更有效               |
| Prefill + Decode 混合 | 两者 Benchmark | 取决于具体 GPU 和 Batch 形状            |
| Training / Backward   | FlashAttention | FlashAttention 提供成熟的前向、反向实现 |

#### 后端如何处理 Prefill/Decode 混合批次

这里有个上一节埋下的关键联系。vLLM V1 的统一调度器会把 **Prefill 的 Chunk 和 Decode 请求塞进同一步**——那么这一步的 Attention 该怎么算？毕竟 Prefill 部分要算”一整段新 Token 之间 + 对历史的注意力”，Decode 部分只算”1 个新 Token 对全部历史的注意力”，两者的计算形态不一样。

这就要求 Attention 后端**原生支持混合批次（mixed prefill/decode batch）**：在一次 Kernel 调用里，同时正确处理批中既有 Prefill 又有 Decode 的请求。FlashAttention 等现代后端提供的变长（varlen）接口正是为此设计的——它把批里所有请求的 Query 首尾拼成一个变长序列，配合每个请求的起止偏移，在一次 Kernel 调用里算完，不需要把 Prefill 和 Decode 拆成两次调用。

**关键点**：**“统一调度器”（软件层抹平 Prefill/Decode）和”支持混合批次的 Attention 后端”（Kernel 层抹平 Prefill/Decode）是配套的。** 没有后者，V1 那套优雅的 Token 预算调度就落不了地。这也是为什么 V1 对 Attention 后端的能力有要求，早期不是所有后端都能支持。

#### CUDA Graph

但 CUDA Graph 有个硬性前提：**录制的操作序列和张量形状必须固定**。这对 Decode 是天作之合——每步都是”处理固定 Batch 个 Token”，形状稳定，可以针对不同的 Batch Size 分别录制好图，运行时按当前 Batch Size 选对应的图重放。

**注意**：正因为要求形状固定，vLLM 只为**一组预设的 Batch Size** 捕获 CUDA Graph。v0.32.0 的默认捕获尺寸是 `[1, 2, 4]`，加上 8 到 248 之间每隔 8 取一个，再加上 256 到 `max_cudagraph_capture_size` 之间每隔 16 取一个；`max_cudagraph_capture_size` 默认 512（数据中心级 Blackwell GPU 上为 1024），两者都可以通过 `compilation_config` 调整。运行时实际 Batch 会被**向上 Padding 到最近的已捕获尺寸**再重放，比如 13 个 Token 的批次按 16 的图跑，多出的 3 个位置是空算；只有超过最大捕获尺寸的批次才回退到逐 Kernel 启动。这也意味着 CUDA Graph 的收益主要体现在 Decode 这类小而规整的批次，大块的 Prefill 较难直接套用。

#### torch.compile

CUDA Graph 解决了”启动开销”

torch.compile 把模型的前向计算**编译**成优化后的图，通过算子融合（fusion）等手段减少 Kernel 数量

问题：

CUDA Graph 要求整段操作形状固定，而 **Attention 恰恰是最难被 CUDA Graph 兼容的部分**——它涉及变长的 KV、复杂的 Mask，形状不那么规整。如果因为 Attention 不好处理就放弃整张图的 CUDA Graph，就太可惜了。

vLLM V1 的巧解是 **Piecewise CUDA Graph（分段 CUDA Graph）**：

- 把整个前向图**在 Attention 算子处切开**（Attention 被包装成一个”不透明”的自定义算子，编译器不去动它）。
- **Attention 之外的部分**（LayerNorm、各种 GEMM、激活、残差……这些都是规整的 token-wise 操作）用 CUDA Graph 捕获、重放。
- **Attention 本身保持 eager（即时）执行**，保留它处理变长 KV 的灵活性。

**核心概念**：**Piecewise CUDA Graph = 对规整的部分用图优化榨干 CPU 开销，对灵活的 Attention 网开一面保留 eager。** 二者各取所长：CPU launch 开销大头（那成百上千个小算子）被 CUDA Graph 消灭，而 Attention 的灵活性（支持混合批次、变长 KV）不受损。这正是 V1 相比 V0 在 Decode 上提速的关键工程之一。

这里decode由于形状固定，可以直接full cuda graph，prefill计算形状更动态

batch向上padding到最近的捕获尺寸再重放，超过max_cudagraph_capture_size的批次不走 cuda graph

#### 为什么要把 Attention 包装成黑盒？

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
        # 只为这些 batch size 捕获 CUDA Graph；批次向上 Padding 到最近的尺寸重放，
        # 超过最大值（这里是 32）的批次不走 CUDA Graph
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
        # 可选：为特定静态形状做 Inductor 自动调优（首次较慢，默认关闭）
        compile_sizes=[1, 2, 4, 8],
    ),
)
```

- `cudagraph_capture_sizes`：控制为哪些 Batch Size 捕获图。不设时用第 4 节讲的默认列表；列表越长，覆盖越细、Padding 浪费越少，但捕获耗时和显存占用越多。
- `compile_sizes`：让 Inductor 为指定静态形状做自动调优，找更快的 Triton 配置——收益是运行更快，代价是首次编译更慢，所以默认关闭。

编译产物会写入**编译缓存**，默认位于 `~/.cache/vllm/torch_compile_cache`（缓存键包含所有相关配置与被追踪的源文件）。缓存目录可在部署间**直接拷贝**以省掉重复编译时间；调试时若想强制重编，可用 `VLLM_DISABLE_COMPILE_CACHE=1` 关闭缓存

这里编译的是**Graph（计算图），但最终生成的是 Kernel（GPU 可执行代码）**



- **收益**：Decode 阶段 CPU 开销大幅下降，小模型/小 Batch 下的吞吐和延迟明显改善；`torch.compile` 的算子融合进一步减少 Kernel 数量。
- **代价**：
  - **启动变慢**：服务启动时要做编译和 CUDA Graph 捕获，首次拉起耗时增加（vLLM 用编译缓存缓解，缓存目录可复制以加速后续启动）。
  - **显存占用**：捕获多个 Batch Size 的 CUDA Graph 需要额外显存。
  - **调试复杂度**：编译后的图不如 eager 模式好调试，排查数值问题时可能需要临时关掉。

`--enforce-eager` 启动，它会同时关闭 `torch.compile` 和 CUDA Graph，用纯 eager 模式对比。定位完再去掉——生产环境务必保持开启，否则会白白损失 Decode 性能

# Nano-Vllm

**第一层：接口层（`llm.py` + `sampling_params.py` + `config.py`）**

用户只接触 `LLM`（提供 `generate`）和 `SamplingParams`（温度、最大生成长度等）。`Config` 在引擎初始化时把显存比例、最大批大小、张量并行度、KV 块大小等全局约束固定下来。这一层的对外 API 很小，不懂调度也能跑通推理。

**第二层：引擎层（`engine/llm_engine.py`）**

`LLMEngine` 持有调度器、模型执行器和分词器，跑一个循环：**从调度器拿一批序列 → 交给执行器做一次前向并采样出下一个 token → 把结果写回序列状态**，直到所有请求结束。

**第三层：调度层（`scheduler.py` + `sequence.py` + `block_manager.py`）**

这一层决定「这一步跑谁、它们的 KV 放哪儿」。`Scheduler` 决定本轮是 prefill 还是 decode、把哪些序列拼成一个 batch；`Sequence` 跟踪每条请求的 token、已缓存长度和状态；`BlockManager` 按 PagedAttention 的思路给 KV 分配物理块。这一层**只管资源分配与生命周期，不碰矩阵乘法**，所以 `BlockManager` 放在 `engine/` 而不是 `layers/`。

**第四层：执行层（`model_runner.py` + `models/qwen3.py` + `layers/\*`）**

`ModelRunner` 把调度器选出的序列转成张量：构造 `input_ids`、`positions`、`slot_mapping`、`block_tables` 等，设置全局 `Context`，然后调用 `Qwen3` 前向。`layers/` 是算子级实现，其中 `attention.py` 有两项任务：把新的 K/V 写进 Cache，以及调用 FlashAttention 计算注意力。