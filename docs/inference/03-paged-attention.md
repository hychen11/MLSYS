# PagedAttention：块管理与抢占

## 核心思路

[第 1 章](01-basics.md#kv-cache-的碎片化)提到，按最大长度预留连续显存会产生内部碎片和外部碎片，而且多个请求的相同前缀很难在连续布局里共享。PagedAttention 借用操作系统虚拟内存的分页思想：

1. **分块存储**：KV Cache 不再要求连续，而是切成固定大小的 Block（例如每块 16 个 Token）。
2. **按需分配**：当前 Block 写满后才分配下一个，浪费最多只有最后一块没填满的部分。
3. **块表映射**：每个请求维护一张 Block Table，记录逻辑块到物理块的映射，就像页表。
4. **内存共享**：相同前缀（如 System Prompt）的请求，块表可以指向同一组物理块。

| 操作系统 | PagedAttention |
| --- | --- |
| 虚拟页 | 逻辑 KV 块 |
| 物理页帧 | 物理 KV 块 |
| 页表 | Block Table |
| 共享页 + 写时复制 | 共享前缀块 + Copy-on-Write |
| 换出到磁盘 | 换出到 CPU 内存 / 丢弃后重算 |

## Kernel 如何读取分散的 KV

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

代价是每次访问都多一次查表和地址计算，以及块粒度带来的访存不完全连续。实践中块足够大（16 个 Token 以上）时，这部分开销远小于省下的显存所带来的并发收益。

!!! note
    vLLM 官方这篇 Kernel 走读文档明确标注为「基于原始论文的历史文档，不再反映 vLLM 当前代码」。上面的指针公式请当作**理解原理的示意**；想读当前实现，要去看 V1 引擎的 Attention 后端。

## 内存共享与 Copy-on-Write

两种共享场景：

- **跨请求共享前缀**：一批请求用了相同的 System Prompt，前缀 KV 完全相同，让它们的块表指向同一批物理块即可。这种共享是**只读**的。
- **同一请求内的分叉**：从同一个 Prompt 并行采样多个回答（`n > 1`）或做 Beam Search 时，多个候选序列共享 Prompt 的 KV，但会在某个块**内部**开始分叉。这才需要 Copy-on-Write。

共享时用**引用计数（`ref_cnt`）**记录有几个序列在用这个块。当某个序列要往一个 `ref_cnt > 1` 的块里写新内容时，先把这个块复制一份作为私有块，在副本上写，再把自己的块表指向副本，其他序列不受影响。

CoW 通常不是为了修改已有 Token 的 KV（历史 KV 不会变），而是为了**向尚未填满的共享块追加新 Token**。

## 显存不足时的抢占

某一步 Decode 需要为若干请求追加新块，但空闲物理块不够。调度器会挑选部分请求「踢出」，释放它们占用的物理块。被踢出请求已经算好的 KV Cache 有两种处理方式：

| 恢复方式                    | 做法                                                         | 优点                    | 代价                                    |
| :-------------------------- | :----------------------------------------------------------- | :---------------------- | :-------------------------------------- |
| **换出（Swapping）**        | 把 KV Cache 从 GPU 显存拷贝到 CPU 内存暂存，恢复时再拷回来   | 不用重算，省算力        | 占用 CPU 内存，来回拷贝有 PCIe 传输开销 |
| **重计算（Recomputation）** | 直接丢弃 KV，恢复时把该请求已生成的 Token 当成新 Prompt 重新 Prefill 一遍 | 不占 CPU 内存，实现简单 | 要重跑一次 Prefill，浪费算力            |

上表是 PagedAttention 论文和早期实现讨论的两种恢复思路。vLLM V1 的常规抢占路径采用重计算：释放块并重置已计算 Token 数，恢复时重新推进 Prefill。若前缀缓存中的满块仍可复用，实际重算量会减少。

抢占可以缓解 KV Cache 显存不足，但会增加重计算和延迟，不能保证所有情况下都不发生 OOM 或请求失败。若抢占频繁，应结合并发、KV Cache 空间和显存余量调参。

通常先观察 `gpu_memory_utilization`、并发和前缀缓存命中率，再考虑块大小等底层参数；默认值随版本和硬件而变。

## vLLM V1 实现：KVCacheManager 与 BlockPool

调用关系是 `Scheduler → KVCacheManager → BlockPool`：

- **`KVCacheManager`**（`kv_cache_manager.py`）：**请求视角**的管理者。调度器每一步都通过它给请求申请、释放块。核心方法和前面讲的概念一一对应：
  - `get_computed_blocks`：前缀缓存查询，返回该请求能命中的、已经算好的块，以及命中了多少 Token。这是第 5 节前缀复用的入口。
  - `allocate_slots`：为请求新增的 Token 申请槽位/块，顺带释放不再需要的块，并检查空闲块是否够用；不够就返回 `None`（交给调度器触发抢占）。它对应本节的按需分块。
  - `free`：请求结束时**逆序**释放它的块——让尾部的块先被回收，因为头部的块更可能被前缀缓存复用，留得越久越好。
- **`BlockPool`**（`block_pool.py`）：**全局视角**的物理块仓库，不关心是哪个请求，只管「哪些块空着、哪些块缓存着有效内容」。

### 空闲块为什么用双向链表

`BlockPool` 里所有块存在一个普通列表 `self.blocks: list[KVCacheBlock]`（按 block id 索引），**空闲块**则单独用 `FreeKVCacheBlockQueue` 管理。它内部是一条**双向链表**，而不是栈或 `deque`，因为空闲块要同时支持两种操作：

- **分配时**从头部快速取走 N 个（`popleft_n`）；
- **前缀缓存命中时**，一个原本「空闲但仍缓存着有效内容」的块要被重新激活——需要从链表**中间任意位置**把它摘出来（`remove`）

持有节点引用时，双向链表能在 O(1) 时间从中间摘除节点。

注意这里的「空闲」只表示没有运行中的请求在用它。一个 Block 可以同时满足：

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

块对象本身带 `prev` / `next` 指针，自己就是链表节点，不需要额外的包装对象。

### 释放与复用

1. **释放不等于清空**：`ref_cnt` 归零的块只是回到空闲链表，**内容还在**。如果后续请求的前缀哈希命中它，`touch()` 直接把它复活复用，这就是 Automatic Prefix Caching 省掉重复 Prefill 的底层机制。
2. **带哈希的块留得更久**：释放时，vLLM 把「没缓存价值的块」放链表头（优先被再分配覆盖，源码注释说这样还能让刚释放的块被立即复用，GPU 局部性更好），「带哈希、有复用潜力的块」放链表尾（晚点才被淘汰）。分配总从链表头取，于是带哈希的块按释放先后被淘汰，效果就是 LRU。

### V1 为什么不需要 CoW

**V1 的 BlockPool 里没有 CoW。** 下面伪代码里的几个函数只做 `ref_cnt` 的增减和空闲链表的摘挂，没有任何「复制块」的逻辑。V1 能绕开 CoW，靠的是两点设计：

- **只共享写满的块**：前缀缓存只对已写满的块计算哈希并共享，未写满的尾块不参与共享；新生成的 Token 一律追加到请求自己的新块里。因此跨请求的前缀共享始终是只读的，没有「往共享块里写」的时刻。
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
