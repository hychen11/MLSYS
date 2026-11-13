GPU Kernel 的性能上限由**计算量**和**访存量**共同决定，可以用 Roofline 模型来定位瓶颈。对于 Sum Reduce

* 元素读一次 1 load

* 加法一次 1 flop

* 算数强度 1 flop / 4bytes, 1 float 4 bytes

这表明 Reduce 是典型的**访存密集型**（Memory-Bound）操作，优化的核心在于**提升内存带宽利用率**，而不是减少计算次数。

测试环境 3090ti sm_86 amphere 架构

| 指标                    |         数值 |
| ----------------------- | -----------: |
| 理论显存带宽            |   1,008 GB/s |
| L2 Cache 容量           |         6 MB |
| SM 数量                 |           84 |
| 每 SM Shared Memory     |  最大 100 KB |
| 显存容量                | 24 GB GDDR6X |
| CUDA Compute Capability |          8.6 |

测试数据规模：2^27（128M 个 `float`，共 512 MB）

注意一个reduce有很多优化，比如baseline **约 15%（~300 GB/s）**

防止branch divergence**约 32%（~653 GB/s）**，相比 V0 提升约 2.1 倍

防止bank conflict**约 40%（~816 GB/s）**，相比 V1 提升约 1.25 倍

### V2

```c++
__global__ void reduce(float *input, float* output, int n){
    extern __shared__ float smem[];
    int tid=threadIdx.x;
    int gid=blockIdx.x * blockDim.x +threadIdx.x;
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

### v3 解决idle线程

规约的第一轮（step = blockDim.x/2）迭代中，只有前 128 个线程参与后续计算，后 128 个线程就彻底闲置了——它们的唯一贡献就是把数据从全局内存搬到了 Shared Memory

换个角度看：如果每个线程在加载阶段就负责 2 个元素并预先求和，那么用同样 256 个线程可以处理 512 个数据——Block 数量减半（262144 个），而规约阶段的工作量不变

- Grid 更小，调度开销降低
- 每个线程在加载阶段就已经做了"有用功"，不再是纯搬运工

注意 Grid 配置的变化——因为每个 Block 现在处理 `blockDim.x * 2` 个元素，Block 数量应设为 `(n + blockDim.x * 2 - 1) / (blockDim.x * 2)`

优化本质上是在**不增加规约轮数**的前提下，让每个线程承担更多工作

**约 45%（~918 GB/s）**，相比 V2 提升约 1.12 倍

```c++
__global__ void reduce(float *input, float* output, int n){
    extern __shared__ float smem[];
    int tid=threadIdx.x;
    int gid=blockIdx.x * (blockDim.x *2) +threadIdx.x;
    float val=0.0f;
    if(gid<n) 					 val+=input[gid];
    if(gid+blockDim.x<n) val+=input[gid+blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

### v4

V3 在规约阶段仍然使用完整的 for 循环，当 `step <= 32` 时，只有 1 个 Warp 的线程（tid 0~31）还在工作。此时循环里的 `__syncthreads()` 完全是多余的，因为同一 Warp 内的线程在执行相同指令路径时本身就是同步的，不需要额外的屏障。这些多余的 `__syncthreads()` 带来了不必要的开销。

```c++
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

// V4: 展开最后一个 Warp（在 V3 基础上）
__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每线程处理 2 个元素（继承自 V3）
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 规约循环仅执行到 step > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

`warpReduce` 中的 `smem` 必须声明为 `volatile`，防止编译器将中间结果缓存到寄存器，导致其他线程读取到旧值。这是保证 Warp 内线程间数据可见性的关键

以 blockDim=256 为例，V3 的规约循环有 8 轮（step = 128, 64, 32, 16, 8, 4, 2, 1），每轮都执行 `__syncthreads()`。V4 将后 5 轮（step ≤ 32）展开为直线代码，省去 5 次 `__syncthreads()` 调用。

**V4 实测带宽利用率：约 52%（~1060 GB/s）**，相比 V3 提升约 1.15 倍。

### v5

对于 Block 大小在编译期已知的情况（如固定为 256），这些开销完全可以消除。

使用模板参数实现编译期展开

通过将 Block 大小作为模板参数传入，编译器可以在编译时确定所有循环的迭代次数，直接生成展开后的代码：

```
// V5: 完全循环展开（模板参数）
template <int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    // 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    // 最后 Warp 内展开
    if (tid < 32) {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

调用时需要根据 Block 大小选择对应的模板实例化：

```
switch (block_size) {
    case 512: reduce_v5<512><<<grid, 512, 512*sizeof(float)>>>(d_in, d_out, n); break;
    case 256: reduce_v5<256><<<grid, 256, 256*sizeof(float)>>>(d_in, d_out, n); break;
    case 128: reduce_v5<128><<<grid, 128, 128*sizeof(float)>>>(d_in, d_out, n); break;
}
```

为什么模板展开比运行时循环更快

💡 **提示**：以 `BLOCK_SIZE=256` 为例，编译器生成的代码等价于：

```
// 编译后实际生成的指令序列（无循环、无分支判断）：
if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads();
// 然后直接展开 Warp 内规约...
```

`BLOCK_SIZE >= 512` 的条件在编译期为假，整个 `if` 块被删除，不生成任何指令。最终的机器码是一段紧凑的直线代码，没有循环跳转、没有死代码。

**V5 实测带宽利用率：约 62%（~1265 GB/s）**，相比 V4 提升约 1.19 倍。

### V6 Warp Shuffle 替代 Shared Memory

`__shfl_down_sync` 是warp level prim，底层依赖 GPU SM 内部的专用 Warp Shuffle 数据交换机制，不需要同步，不会data race，数据并不是完全不复制，而是通过硬件数据通路，把源 Lane 的寄存器值传递到目标 Lane 的寄存器中

Warp 内的 32 个线程有一种特殊的通信方式：**寄存器直接交换**（Warp Shuffle），无需经过 Shared Memory。

```c++
float val=__shfl_down_sync(0xffffffff, var, delta);
// 每个寄存器读取 lane_id+delta处线程的寄存器，往后读
```

- `0xffffffff`：表示 Warp 内全部 32 个线程参与
- `var`：每个线程的寄存器值
- `delta`：偏移量（要读取的是 `lane_id + delta` 的值）

这比 Shared Memory 访问快很多，因为：

1. 不需要地址计算
2. 不会有 Bank Conflict
3. 延迟更低（寄存器级互联）

缺点是只能在一个warp 32线程 里进行

对于 256 或 512 个线程的 Block，需要**两级规约**：

1. **Warp 内规约**：每个 Warp 内通过 `__shfl_down_sync` 将 32 个值规约为 1 个值
2. **Warp 间规约**：将各 Warp 的结果写入 Shared Memory，再做一轮规约

8*32 reduce -> 8 reduce

注意这里warp里的所有threads都要参与调用`__shlf_down_sync()`!

```c++
__device__ float warpReduceSum(float val){
  	for(int offset=16;offset>0;offset>>=1){
      	val+=__shfl_down_sync(0xffffffff,val,offset);
    }
  	return val;
}
__global__ void reduce_v6(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;      // 线程在 Warp 内的编号（0~31）
    int wid  = tid / 32;      // 该线程属于哪个 Warp

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：Warp 内规约
    val = warpReduceSum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32];  // 最多 32 个 Warp（1024/32）
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    // 第二级：Warp 间规约（用 Warp 0 处理）
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}
```

### V7 向量化加载（float4）

GPU 的内存系统以**事务（Transaction）**为粒度传输数据，每次事务通常为 128 字节。如果每个线程每次只加载 4 字节（1 个 float），则：

- 一个 Warp 32 线程 × 4 字节 = 128 字节，恰好一个事务
- 但每条加载指令的调度开销是固定的

改为使用 `float4`（16 字节），每个线程每次加载 4 个 float：

- 每条 `ld.global.v4.f32` 指令的数据吞吐是 `ld.global.f32` 的 4 倍
- 在相同的循环迭代次数下，处理的数据量翻 4 倍，等效地减少了循环次数
- 提升指令级并行（ILP），让访存流水线更饱和

128bits = 16bytes = 4*float = float4

### **Grid Stride Loop**

V0-V6 中每个 Block 处理固定数量的数据。更灵活的模式是 **Grid Stride Loop**：固定 Grid 大小，让每个线程循环处理多段数据，直到覆盖整个数组。

好处：

- Grid 大小可以设置为恰好填满 GPU，避免尾部 Block 浪费，**固定 Grid 大小为 SM 数 × 4**（最大化 GPU 利用率）
- 对超大数组（超出最大 Grid 限制）同样适用
- 每个线程处理更多数据，充分摊销 Kernel 启动和规约的固定开销

```c++
struct float4{
    float x;
    float y;
    float z;
    float w;
}
```

```c++
// V7: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
__global__ void reduce_v7(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    // float4 加载：每线程每次处理 4 个 float
    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = n / 4;  // float4 的元素数量

    float val = 0.0f;

    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 处理 n 不是 4 的倍数时的尾部元素
    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    // Warp 内规约
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    if (tid == 0) output[blockIdx.x] = val;
}

// 固定 Grid 大小为 SM 数 × 4（最大化 GPU 利用率）
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
int grid_size  = num_sms * 4;   // 432 for A100
int block_size = 256;

reduce_v7<<<grid_size, block_size>>>(d_input, d_partial, n);
```

blocksize 256 8个warp

gridsize sm*4 相当于一个sm 32个warp，一般一个warp可以跑48/64，所以一般Occupancy会高点



**然后这里精髓就是 grid stride loop！** 也就是固定grid 和block，然后每个线程循环读！stride就是block_size * grid_size，然后每次读都是float4，总数也是n4=n/4，一般都vector+grid stride loop一起用的



## bank

Shared Memory 被划分为 32 个 **Bank**（默认 32-bit 模式，每个 Bank 宽度为 4 字节）。相邻的 4 字节地址依次落到 Bank 0、Bank 1、…、Bank 31、Bank 0、Bank 1、……如此循环。

理想情况下，同一 Warp 内的 32 个线程访问 32 个不同 Bank，可以**同时**完成——这叫**无冲突访问**。但如果一个 Warp 内有两个或多个线程访问**同一个 Bank 的不同地址**，访问就会**串行化**，这就是 **Bank Conflict**。

**注意**：同一 Warp 内多个线程访问**同一地址**不会产生冲突（硬件可以广播），只有访问同一 Bank 的**不同地址**才会冲突。



注意一个bank宽度4bytes，一般32个bank，一个bank可以很多，比如n*32 这么多块，其中32个bank，每个bank就n块，

**4 Bytes 指的是典型情况下 Bank 的访问粒度**，一个 SM 的 Shared Memory 总容量通常是几十到几百 KiB，3090ti就大约100kb

```
__shared__ float smem[1024];
```

就是1024*4bytes=4096bytes，然后32个bank，128bytes，每个bank 32块，存128bytes



Shared Memory 被划分成多个 Bank，目的是让 GPU 的多个线程能够并行访问 Shared Memory。但如果多个线程同时访问同一个 Bank 中的不同地址，就可能产生 Bank Conflict，导致访问被串行化

在 NVIDIA GPU 中，Shared Memory 是位于 SM 内部的高速片上存储。

它并不是一整块只能同时处理一次访问的存储，而是划分成多个可以并行访问的 Bank。

比如shared mem里有32个banks，每个bank **4bytes**，32bits，不同bank可以并行处理同一条warp内存指令的访问

假设 Warp 有 32 个线程：

- Thread 0 访问 Bank 0
- Thread 1 访问 Bank 1
- ...
- Thread 31 访问 Bank 31

那么这些访问可以并行完成，不会产生 Bank Conflict。

#### 数据如何映射到 Bank

对于典型的 32-Bank、每 Bank 4 Bytes 的配置，映射公式为：

\[ \boxed{\text{Bank ID}=\left(\frac{\text{Byte Address}}{4}\right)\bmod32} \]

很重要，就是(address/4) mod 32 !

这里的 Address 是相对于 Shared Memory 起始位置的字节偏移。

```
__shared__ float smem[128];
smem[0] offset 0 bank 0
smem[1] offset 4 bank 1
smem[32] offset 128 bank 31
smem[33] offset 132 bank 32
```

**Bank ID 是循环映射的，每隔 32 个 float 就回到 Bank 0**

```
Bank 0: smem[0], smem[32], smem[64], ...
Bank 1: smem[1], smem[33], smem[65], ...
Bank 2: smem[2], smem[34], smem[66], ...
...
```

### Bank Conflict

```
float x = smem[threadIdx.x];
```

Thread 0 访问 smem[0]，Thread 1 访问 smem[1]，依此类推。

每个线程访问不同的 Bank，因此没有 Bank Conflict。

但如果执行：

```
float x = smem[threadIdx.x * 32];
```

所有线程都访问bank 0，但是读取的不同地址，造成32way bank conflict

如果 32 个线程访问的是完全相同的地址，Shared Memory 可以通过 Broadcast 机制将数据提供给多个线程，通常不会产生 Bank Conflict。

所以不是所有访问同一个 Bank 的情况都算冲突，重点在于是否访问该 Bank 中的不同位置。

|          | Bank Conflict           | Memory Coalescing                                |
| -------- | ----------------------- | ------------------------------------------------ |
| 主要针对 | Shared Memory           | Global Memory                                    |
| 关注什么 | 线程是否竞争同一个 Bank | 线程访问地址能否合并为较少的 Memory Transactions |
| 典型问题 | 32 个线程访问同一列     | 32 个线程跨很大步长访问 HBM                      |
| 优化方法 | Padding、Swizzling      | 连续访问、合理的数据布局                         |

在典型的 32-bit Shared Memory 访问情况下：

- 不同 Bank：可以并行访问。
- 同 Bank、不同地址：可能产生 Bank Conflict。
- 同 Bank、相同地址：读取时可通过 Broadcast 避免冲突。
- Padding 或 Swizzling：通过改变地址映射，减少 Bank Conflict。

## 总结

| 版本                    | 核心优化点                      | 带宽利用率 | 相对速度 |
| :---------------------- | :------------------------------ | :--------- | :------- |
| V0 朴素树形             | 无                              | ~15%       | 1.0x     |
| V1 交错寻址             | 减少 Warp Divergence            | ~32%       | 2.1x     |
| V2 步长反转             | 消除 Bank Conflict + Divergence | ~40%       | 2.7x     |
| V3 双元素处理           | 减少空闲线程，Grid 减半         | ~45%       | 3.0x     |
| V4 展开最后 Warp        | 省去 Warp 内多余 syncthreads    | ~52%       | 3.5x     |
| V5 完全循环展开         | 模板参数，编译期消除所有循环    | ~62%       | 4.1x     |
| V6 Warp Shuffle         | 寄存器直通，省去 Shared Memory  | ~72%       | 4.8x     |
| V7 向量化 + Stride Loop | 提升访存效率 + 完整覆盖 GPU     | ~85%       | 5.7x     |

理论峰值带宽 2039 GB/s，V7 达到 ~1733 GB/s（约 85%），已接近实际可达上限（受 ECC、时钟波动等影响，85-90% 是 Reduce 的合理目标）

生产环境中优先使用 NVIDIA CUB 库中的 `cub::DeviceReduce::Sum`，它在各种 GPU 架构上做了针对性优化，通常能达到 90% 以上的带宽利用率，且维护成本为零

1. **Warp Divergence（初步）**：用 strided index 替换 `tid % (2*s) == 0` 判断，让完整 Warp 进入/跳过分支（V1）
2. **Bank Conflict + Warp Divergence（彻底）**：反转步长方向，从 `blockDim/2` 逐步减半，一次性解决两类问题（V2）
3. **空闲线程**：每线程处理 2 个元素，减少 Block 数量，提升线程利用率（V3）
4. **多余同步**：Warp 内天然同步，展开最后 5 轮可以省去 `__syncthreads()`（V4）
5. **循环开销**：模板参数让编译器删除无用分支，生成紧凑的直线代码（V5）
6. **访存层次**：Warp Shuffle 直接在寄存器间通信，比 Shared Memory 更快（V6）
7. **带宽效率**：`float4` 减少指令数，Grid Stride Loop 最大化 GPU 占用率（V7）