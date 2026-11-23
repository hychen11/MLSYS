# Reduce 优化：从 V0 到 V7

## 为什么 Reduce 是访存密集型

用 [Roofline](../performance/roofline.md) 分析 Sum Reduce：每个 `float` 元素读一次（4 Bytes）、做一次加法（1 FLOP），算术强度只有 0.25 FLOP/Byte，远低于任何 GPU 的屋脊点。

所以 Reduce 是典型的 **Memory-bound** 操作，优化的核心是**提升内存带宽利用率**，而不是减少计算次数。衡量指标就是「实测带宽 / 峰值带宽」。

## 实验环境

测试卡是 NVIDIA A100 80GB（sm_80，Ampere）：

| 指标 | 数值 |
| --- | ---: |
| 理论显存带宽 | 2,039 GB/s |
| L2 Cache | 40 MB |
| SM 数量 | 108 |
| 每 SM Shared Memory | 最大 164 KB |
| 每 SM 最大驻留线程 | 2048（64 个 Warp） |
| 显存 | 80 GB HBM2e |
| Compute Capability | 8.0 |

数据规模：$2^{27}$ 个 `float`（128M 个，共 512 MB），远大于 40 MB 的 L2，保证测的是 HBM 带宽。Block 大小默认 256。下文的带宽利用率都相对 2039 GB/s 的理论峰值计算。

## V0：朴素交错寻址

每轮让 `tid % (2*s) == 0` 的线程把相距 `s` 的两个元素相加。

```c++
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0) {          // 同一 Warp 内一半线程进分支、一半不进
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

问题：

- **Warp Divergence**：同一个 Warp 里的线程有的满足条件、有的不满足，两条路径要串行执行。
- `%` 取模运算本身也比较慢。

带宽利用率约 **15%**（~300 GB/s）。

## V1：消除 Warp Divergence

换一种线程到数据的映射：让**连续的线程**干活，把「哪个线程工作」和「处理哪个数据」解耦。

```c++
for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {              // 活跃线程集中在前面，整 Warp 要么全进要么全不进
        smem[index] += smem[index + s];
    }
    __syncthreads();
}
```

Divergence 基本消失了，但引入了新问题：`index = 2*s*tid` 让同一 Warp 的线程以 $2s$ 为步长访问 Shared Memory，会产生 [Bank Conflict](shared-memory-bank.md)。例如 $s=16$ 时步长是 32 个 float，整个 Warp 都落在同一个 Bank 上。

带宽利用率约 **32%**（~653 GB/s），相比 V0 提升约 2.1 倍。

## V2：顺序寻址，消除 Bank Conflict

把步长反过来：从 `blockDim.x / 2` 开始逐轮减半，每轮让前 `s` 个线程把后半段加到前半段。

```c++
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];    // 连续线程访问连续地址
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

同一 Warp 的 32 个线程访问 `smem[tid]` 和 `smem[tid + s]`，都是连续地址，分别落在 32 个不同的 Bank 上；活跃线程又集中在前面，同时解决了 Divergence 和 Bank Conflict。

带宽利用率约 **40%**（~816 GB/s），相比 V1 提升约 1.25 倍。

## V3：加载时先做一次加法

V2 第一轮（$s$ = blockDim.x/2）就有一半线程闲置，它们唯一的贡献是把数据从全局内存搬进 Shared Memory。

改成每个线程在加载阶段就读 2 个元素并先求和。同样 256 个线程可以处理 512 个数据，Block 数量减半（从 524288 个降到 262144 个），而规约阶段的工作量不变：

- Grid 更小，调度开销降低
- 每个线程在加载阶段就做了「有用功」，不再是纯搬运

```c++
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

注意 Grid 配置要跟着变：每个 Block 处理 `blockDim.x * 2` 个元素，Block 数为 `(n + blockDim.x * 2 - 1) / (blockDim.x * 2)`。

带宽利用率约 **45%**（~918 GB/s），相比 V2 提升约 1.12 倍。

## V4：展开最后一个 Warp

当 $s \le 32$ 时，只剩一个 Warp（tid 0–31）在工作。此时每轮的 `__syncthreads()` 是整个 Block 的屏障，开销偏大，而只需要 Warp 内同步就够了。

以 blockDim = 256 为例，规约有 8 轮（$s$ = 128, 64, 32, …, 1），把最后 6 轮单独展开成 Warp 内规约，可以省掉 6 次 `__syncthreads()`。

```c++
__device__ void warpReduce(volatile float* smem, int tid) {
    float v = smem[tid];
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += smem[tid + offset];
        __syncwarp();          // 保证本轮所有 lane 都读完，再写
        smem[tid] = v;
        __syncwarp();          // 保证本轮所有 lane 都写完，再进入下一轮读
    }
}

__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {   // 只循环到 s > 32
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < 32) warpReduce(smem, tid);                       // 最后一个 Warp 单独处理

    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

!!! danger "老写法在 Volta 及之后的架构上不安全"
    很多教程里的 `warpReduce` 只有 `volatile` 和一串 `smem[tid] += smem[tid + k]`，没有 `__syncwarp()`。这依赖「Warp 内 32 个线程天然锁步执行」的假设。从 Volta（sm_70）开始引入了独立线程调度（Independent Thread Scheduling），这个假设不再成立，可能读到其他 lane 还没写完的值。

    - `volatile` 只保证每次都真的读写 Shared Memory、不被缓存到寄存器，**不提供同步**。
    - 正确做法是像上面那样在读写之间加 `__syncwarp()`，或者直接用 V6 的 Warp Shuffle。

带宽利用率约 **52%**（~1060 GB/s），相比 V3 提升约 1.15 倍。

## V5：模板参数完全展开

Block 大小在编译期已知时（如固定 256），可以用模板参数让编译器把整个规约循环展开：

```c++
template <int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    // BLOCK_SIZE 是编译期常量，不满足的分支会被直接删除
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    if (tid < 32) warpReduce(smem, tid);    // 同 V4
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

调用时按 Block 大小选择模板实例：

```c++
switch (block_size) {
    case 512: reduce_v5<512><<<grid, 512, 512 * sizeof(float)>>>(d_in, d_out, n); break;
    case 256: reduce_v5<256><<<grid, 256, 256 * sizeof(float)>>>(d_in, d_out, n); break;
    case 128: reduce_v5<128><<<grid, 128, 128 * sizeof(float)>>>(d_in, d_out, n); break;
}
```

以 `BLOCK_SIZE = 256` 为例，`BLOCK_SIZE >= 512` 在编译期为假，整个 `if` 块不生成任何指令。最终机器码是一段紧凑的直线代码，没有循环计数、跳转和死代码：

```c++
if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads();
// 然后是展开后的 Warp 内规约
```

带宽利用率约 **62%**（~1265 GB/s），相比 V4 提升约 1.19 倍。

## V6：Warp Shuffle 替代 Shared Memory

Warp 内的 32 个线程可以**直接交换寄存器**，不经过 Shared Memory：

```c++
float v = __shfl_down_sync(0xffffffff, var, delta);
// 每个 lane 读取 lane_id + delta 号 lane 的 var
```

- `0xffffffff`：参与的线程掩码，表示 Warp 内全部 32 个线程
- `var`：每个线程要交出去的寄存器值
- `delta`：偏移量，读取 `lane_id + delta` 处的值（越界时返回自己的值）

它通过 SM 内部专用的数据通路把源 lane 的寄存器值传到目标 lane 的寄存器，自带 Warp 内同步，比 Shared Memory 快：

1. 不需要地址计算
2. 没有 Bank Conflict
3. 延迟更低

限制是只能在一个 Warp 内通信，而且**掩码里的所有线程都必须执行这条指令**，不能放在只有部分 lane 进入的分支里。

对 256 线程的 Block，需要两级规约：

1. **Warp 内规约**：每个 Warp 把 32 个值规约成 1 个（8 个 Warp 得到 8 个值）
2. **Warp 间规约**：各 Warp 的结果写入 Shared Memory，再由 Warp 0 规约一次

```c++
__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;                                    // lane 0 持有结果
}

__global__ void reduce_v6(float* input, float* output, int n) {
    __shared__ float warp_results[32];             // 最多 1024/32 = 32 个 Warp
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    val = warpReduceSum(val);                      // 第一级：Warp 内
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;               // 要求 blockDim.x 是 32 的倍数
    if (wid == 0) {                                // 第二级：Warp 间，整个 Warp 0 都参与
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    if (tid == 0) output[blockIdx.x] = val;
}
```

带宽利用率约 **72%**，相比 V5 再提升约 1.17 倍。

## V7：向量化加载 + Grid Stride Loop

### 向量化加载（`float4`）

全局内存以事务为粒度传输。每个线程一次只读 4 Bytes 时，一个 Warp 正好 128 Bytes、一个事务，带宽本身不浪费；但每条加载指令都有固定的发射开销，要靠很多条指令才能喂饱访存流水线。

改用 `float4`（16 Bytes = 128 bit），每个线程一条 `ld.global.v4.f32` 指令读 4 个 float：

- 同样的数据量，加载指令数降到 1/4
- 每个线程手上同时有更多在途的访存请求，提升指令级并行（ILP），访存流水线更饱和

`float4` 是 CUDA 内置类型（无需自己定义），布局等价于：

```c++
struct __align__(16) float4 { float x, y, z, w; };
```

注意 `reinterpret_cast<float4*>` 要求地址 16 字节对齐；`cudaMalloc` 返回的指针满足这一点，但对一个偏移过的子数组做向量化时要小心。

### Grid Stride Loop

V0–V6 中每个 Block 处理固定数量的数据，Grid 大小随数据规模线性增长。更灵活的模式是**固定 Grid 大小，让每个线程以 `gridDim.x * blockDim.x` 为步长循环**，直到覆盖整个数组：

- Grid 可以设成正好填满 GPU（如 SM 数 × 4），避免尾部 Block 浪费
- 对超大数组（超出 Grid 上限）同样适用
- 每个线程处理更多数据，摊薄了 Block 内规约和 Kernel 启动的固定开销

**向量化和 Grid Stride Loop 通常一起用**：每次读一个 `float4`，总迭代空间是 `n4 = n / 4`。

```c++
__global__ void reduce_v7(const float* input, float* output, int n) {
    __shared__ float warp_results[32];
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;
    int stride = gridDim.x * blockDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    int n4 = n / 4;
    float val = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
        float4 d = input4[i];
        val += d.x + d.y + d.z + d.w;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        val += input[i];                           // n 不是 4 的倍数时的尾部
    }

    val = warpReduceSum(val);
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    if (tid == 0) output[blockIdx.x] = val;
}

// 启动配置
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
int block_size = 256;                  // 8 个 Warp
int grid_size  = num_sms * 4;          // A100: 108 × 4 = 432
reduce_v7<<<grid_size, block_size>>>(d_input, d_partial, n);
reduce_v7<<<1, 1024>>>(d_partial, d_output, grid_size);   // 第二趟：把各 Block 的部分和再规约一次
```

每个 SM 驻留 4 个 Block × 8 个 Warp = 32 个 Warp。A100 每 SM 最多 64 个 Warp，Occupancy 约 50%。对 Reduce 这种每个线程有大量独立访存的 Kernel，这个 Occupancy 已经足以掩盖访存延迟。

每个 Block 只输出一个部分和，所以需要第二趟（如上面用 1 个 Block 再规约），或者在 Block 末尾用 `atomicAdd` 累加到同一个结果上。

带宽利用率约 **85%**（~1733 GB/s）。

## 总结

| 版本 | 核心优化点 | 带宽利用率 | 相对 V0 |
| :-- | :-- | :-- | :-- |
| V0 交错寻址 | 无 | ~15% | 1.0× |
| V1 连续线程 | 消除 Warp Divergence | ~32% | 2.1× |
| V2 顺序寻址 | 消除 Bank Conflict | ~40% | 2.7× |
| V3 加载时相加 | 减少空闲线程，Grid 减半 | ~45% | 3.0× |
| V4 展开最后 Warp | 省去 Warp 内多余的 `__syncthreads()` | ~52% | 3.5× |
| V5 完全展开 | 模板参数，编译期消除循环 | ~62% | 4.1× |
| V6 Warp Shuffle | 寄存器直接通信，省去 Shared Memory | ~72% | 4.8× |
| V7 向量化 + Stride Loop | 提升访存效率，Grid 正好填满 GPU | ~85% | 5.7× |

85%–90% 已经接近 Reduce 实际可达的上限（受 DRAM 刷新、ECC、时钟波动等影响）。生产环境优先用 CUB 的 `cub::DeviceReduce::Sum`，它针对各代架构做了调优，通常能达到 90% 以上的带宽利用率，且无需自己维护。

每一步解决的问题：

1. **Warp Divergence**：让连续线程干活，整 Warp 同进同出（V1）
2. **Bank Conflict**：反转步长，连续线程访问连续地址（V2）
3. **空闲线程**：加载时每线程处理 2 个元素（V3）
4. **多余同步**：最后一个 Warp 只需 Warp 级同步（V4）
5. **循环开销**：模板参数让编译器生成直线代码（V5）
6. **访存层次**：Warp Shuffle 在寄存器间直接通信（V6）
7. **带宽效率**：`float4` 减少指令数，Grid Stride Loop 让 Grid 填满 GPU（V7）
