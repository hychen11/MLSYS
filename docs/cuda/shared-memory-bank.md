# Shared Memory Bank Conflict

## Bank 是什么

Shared Memory 是位于 SM 内部的高速片上存储（A100 每个 SM 最多 164 KB）。它不是一整块一次只能服务一个访问的存储，而是被划分成 **32 个 Bank**，不同 Bank 可以在同一周期并行服务同一条 Warp 访存指令。

在默认的 32-bit 模式下，每个 Bank 宽 4 Bytes。相邻的 4 字节字依次落到 Bank 0、1、…、31，然后回到 Bank 0 循环。

## 地址到 Bank 的映射

$$
\boxed{\text{Bank ID}=\left\lfloor\frac{\text{Byte Offset}}{4}\right\rfloor \bmod 32}
$$

Byte Offset 是相对 Shared Memory 起始位置的字节偏移。对 `float` 数组，就是**下标 mod 32**：

```c++
__shared__ float smem[128];
// smem[0]   offset 0    → Bank 0
// smem[1]   offset 4    → Bank 1
// smem[31]  offset 124  → Bank 31
// smem[32]  offset 128  → Bank 0   ← 每 32 个 float 回到 Bank 0
// smem[33]  offset 132  → Bank 1
```

可以把 Shared Memory 想成一个 32 列的二维表，每一列是一个 Bank：

```text
Bank 0: smem[0], smem[32], smem[64], ...
Bank 1: smem[1], smem[33], smem[65], ...
...
Bank 31: smem[31], smem[63], smem[95], ...
```

例如 `__shared__ float smem[1024]` 共 4096 Bytes，每个 Bank 存 32 个 float（128 Bytes）。

## 什么情况会冲突

同一 Warp 的一条访存指令里：

| 情况 | 结果 |
| --- | --- |
| 32 个线程访问 32 个不同 Bank | 一次完成，无冲突 |
| 多个线程访问**同一 Bank 的不同地址** | Bank Conflict，访问被串行化 |
| 多个线程访问**同一个地址** | 广播（Broadcast），无冲突 |

```c++
float x = smem[threadIdx.x];        // 线程 i → Bank i，无冲突
float y = smem[threadIdx.x * 32];   // 所有线程 → Bank 0 的不同地址，32-way 冲突，串行 32 次
float z = smem[0];                  // 所有线程读同一地址，广播，无冲突
```

$k$ 个线程落在同一个 Bank 的不同地址上，叫 $k$-way bank conflict，这条指令要拆成 $k$ 次完成。

## 怎么避免

### Padding

最典型的例子是矩阵转置：用 `smem[32][32]` 按列读时，同一列的 32 个元素间隔 32 个 float，全部落在同一个 Bank 上。每行多补一个元素就能错开：

```c++
__shared__ float tile[32][32 + 1];   // 每行 33 个 float
// tile[r][c] 的下标是 r*33 + c，Bank = (r*33 + c) % 32 = (r + c) % 32
// 按列读（c 固定、r 从 0 到 31）时，32 个线程落在 32 个不同 Bank
```

代价是浪费少量 Shared Memory。

### Swizzling

不浪费空间，而是用异或等方式重排地址到 Bank 的映射，例如把列下标换成 `c ^ r`。CUTLASS 和 FlashAttention 等高性能 Kernel 普遍使用 Swizzle 布局。

### 换访问模式

像 [Reduce V2](reduce.md) 那样调整线程与数据的对应关系，让连续线程访问连续地址。

## Bank Conflict vs Memory Coalescing

两个概念经常被混淆：

| | Bank Conflict | Memory Coalescing |
| --- | --- | --- |
| 针对 | Shared Memory | Global Memory（HBM） |
| 关注 | 线程是否争抢同一个 Bank | 线程的访问地址能否合并成尽量少的内存事务 |
| 典型问题 | 32 个线程访问同一列 | 32 个线程以很大的步长访问 HBM |
| 优化方法 | Padding、Swizzling | 连续访问、合理的数据布局 |

总结：

- 不同 Bank：并行访问。
- 同 Bank、不同地址：Bank Conflict。
- 同 Bank、同一地址：读取时广播，无冲突。
- Padding 或 Swizzling 通过改变地址映射来减少冲突。
