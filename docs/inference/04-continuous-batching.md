# Continuous Batching：迭代级调度

## 静态 Batching 的问题

传统的静态 Batching 把一批请求凑齐后一起跑，**整批中最长的那个请求结束，整批才算结束**。先结束的请求占着 Batch 里的位置空转，新到的请求只能在外面排队，GPU 利用率和延迟都很差。

Continuous Batching（也叫迭代级调度，iteration-level scheduling，最早由 Orca 提出）把调度粒度从「一批请求」降到「一次前向」：**每一步结束后都重新决定下一步跑哪些请求**，结束的请求立即离开，新请求立即补位。

## 调度循环

1. **收新请求**：把新到达的请求放进 Waiting（等待）队列。
2. **调度决策**：调度器决定这一步谁上车。它要同时兼顾两拨请求——正在生成的 Running 队列请求（做 Decode），和等待队列里能塞进来的新请求（做 Prefill）。是否有空间，取决于显存里还有多少空闲 KV Block（这正是 PagedAttention 提供的信息）。
3. **前向计算一步**：对选中的这批请求跑一次模型前向，每个 Decode 请求产出 1 个新 Token，Prefill 请求产出首 Token。
4. **处理产出**：把新 Token 追加到各请求的 KV Cache；检查哪些请求触发了 EOS 或长度上限，将其标记完成、**释放其占用的 KV Block** 归还给空闲池。
5. **循环**：回到第 1 步。

**提示**：当显存不够、Running 队列里请求太多时，调度器还会做**抢占（Preemption）**：暂停部分请求、释放它们的 KV Block，腾出显存保证其他请求推进。vLLM V1 被抢占的请求一律丢弃 KV、恢复时重算，见[第 3 章](03-paged-attention.md#显存不足时的抢占)。

| 技术                | 解决的问题         | 提供的能力                   |
| :------------------ | :----------------- | :--------------------------- |
| PagedAttention      | 显存碎片、利用率低 | 细粒度、可即时回收的显存分配 |
| Continuous Batching | GPU 空转、排队延迟 | 迭代级调度、动态批次构成     |

**PagedAttention 管「显存怎么放」，Continuous Batching 管「请求怎么调度」，二者结合才有 vLLM 的高吞吐。**

V1 相比 V0 最大的设计变化是：**调度器眼里不再有「Prefill 阶段」和「Decode 阶段」之分**。源码注释里明确写了 "no 'decoding phase' nor 'prefill phase'"：每个请求只是不停地把 `num_computed_tokens`（已计算 Token 数）推向 `num_tokens_with_spec`（含投机 Token 的总目标数）。这种统一视角，正是 [Chunked Prefill](06-chunked-prefill.md) 能把 Prefill 和 Decode 混在一步里跑的前提。

## Scheduler 的两条队列
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

「调度 → 前向 → 处理产出」循环：

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
            req = self.waiting.peek_front()
            if not self._can_admit(req):                  # 检查 running 上限、KV 块是否够等
                break
            self.waiting.pop_front()
            self.running.append(req)
            scheduled.append(req)

    return SchedulerOutput(scheduled)
```

1. **在途请求优先于新请求**：先保证已经在跑的请求继续推进，剩余预算才用于接纳新请求。这避免了「新请求不断插队、老请求迟迟跑不完」的饥饿。
2. **抢占发生在显存耗尽时**：当 `allocate_slots` 返回 `None`（申请不到 KV 块），调度器就挑一个「受害者」抢占，释放它的块后**重试分配**，直到成功或者**踢到当前请求自己**为止。FCFS 策略下踢掉 `running` 里的最后一个（最晚加入的），PRIORITY 策略下踢 `(priority, arrival_time)` 最大的那个，即优先级最低、同优先级里来得最晚的请求。

被抢占的请求在 `_preempt_request()` 里经历一套清理：释放它的 KV 块和编码器缓存 → 状态置为 `PREEMPTED` → **`num_computed_tokens` 归零** → 重新塞回 `waiting` 队列的**队头**，等有资源了优先恢复。

**关键点**：注意 `num_computed_tokens` 归零这一步——它意味着 V1 的抢占恢复策略是**重计算（Recompute）**：被抢占的请求下次调度时，从头把 Prompt 和已生成的 Token 重新 Prefill 一遍，而不是从 CPU 换回 KV（与第 3 节的两种恢复方式呼应）。官方调优文档说明 V1 默认用重计算而非换出，因为在 V1 架构下重计算开销更低；开着 Prefix Cache 时，被释放的满块若还没被淘汰，恢复时可以直接命中。

**提示**：读调度器源码时抓住这条主线——**`waiting`/`running` 两队列 + `token_budget` 预算 + `allocate_slots` 返回值驱动抢占**。前文的调度循环，落到代码就是这三者的相互作用。更深入的调度细节（Chunked Prefill 如何切分）见第 6 节；第 8 节会用 nano-vllm 的精简 Scheduler 再走一遍同样的流程。

Continuous Batching 虽好，但调度器面对的是一组相互冲突的目标，工程上要清楚这些权衡：

- **吞吐 vs 延迟**：一步里塞进的请求越多，系统吞吐越高，但每步的前向计算耗时也越长，单请求的 TPOT 会略微变差。批次大小需要在两者间取平衡。
- **Prefill vs Decode 的争抢**：新请求的 Prefill 是 Compute Bound 的重活，如果某一步塞进一个超长 Prompt 的 Prefill，会明显拖慢同批 Decode 请求的出字速度（TPOT 抖动）。这个问题正是下一节 **Chunked Prefill** 要解决的——把长 Prefill 切块，避免它一次性霸占整个 step。
- **公平性 vs 效率**：先来的请求是否优先？长请求会不会被短请求反复插队饿死？调度策略需要在整体效率和请求间公平之间做取舍。

**注意**：Continuous Batching 让延迟指标变得**对负载敏感**。低负载时批次小、延迟低；高负载时批次变大、排队变长，P95/P99 尾延迟会明显上升。做容量规划时，一定要在**目标并发下**压测延迟，而不是看空载数据。

## 队列的数据结构

| 队列 | 结构 | 操作 |
| --- | --- | --- |
| `waiting` | 双端队列，左边老、右边新 | 新请求 `append` 到右边；调度时 `peek` / `popleft` 左边；被抢占的请求 `appendleft` 放回队头，优先恢复 |
| `running` | 普通 list | 新加入的请求 `append` 到尾部；抢占时从尾部 `pop` |

`running` 没有用链表做 O(1) 删除，因为调度时要按下标遍历、抢占时只从尾部弹出，而且 running 里的请求数本来就不多，list 写起来简单，缓存局部性也好。

## 两种调度器的对比

一个基本事实：**Prefill 影响首字延迟（TTFT），Decode 影响出字间隔（TPOT）**。调度器每一步都要在两者之间分配资源。

### nano-vllm：一步处理一种阶段

每一步都要做选择：这一步给新请求做 prefill，还是给老请求做 decode？

若始终优先 Decode，`running` 中有请求时新请求就难以进入，TTFT 可能很长，Batch 也会逐渐变小。

这个简化实现通常先处理新请求的 Prefill，再让它加入 `running` 一起 Decode；代价是已有请求的出字可能短暂停顿。这是该实现的调度取舍。

### vLLM V1：Prefill 与 Decode 可以混排

既然一步里两种都能放，问题就变成了：token 预算有限时，先满足谁？

decode 很便宜：每条序列只要 1 个 token。running 里有 50 条序列，也只用掉 50 个预算。prefill 很贵：一个 prompt 可能就要几千个 token。

所以 V1 先给 running 里每条序列扣掉 1 个 Token 的预算，保证老请求每步都能出一个字；剩下的大部分预算再给新请求做 Prefill，放不下就分块。

结果是两边都照顾到了：老请求每步都前进，新请求也在同一步里开始 Prefill。这里「先 Decode」的意思不是让新请求等着，而是先扣掉 Decode 那一小部分预算。
