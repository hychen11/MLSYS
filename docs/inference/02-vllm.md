# vLLM：使用与核心能力

| 特性 | 说明 |
| :-- | :-- |
| **PagedAttention** | 借鉴分页思想按块管理 KV Cache，减少预留浪费（[第 3 章](03-paged-attention.md)） |
| **Continuous Batching** | 每一步都可以把新请求插入正在运行的批次，而不是等一批全部完成（[第 4 章](04-continuous-batching.md)） |
| **Prefix Caching** | 自动复用相同前缀的 KV Cache（[第 5 章](05-prefix-cache.md)） |
| **OpenAI 兼容 API** | 提供兼容的服务端接口，现有 OpenAI 客户端改 `base_url` 即可调用 |
| **模型、硬件与量化** | 支持多种模型架构、硬件和量化格式，具体组合取决于版本与后端 |

核心关系：PagedAttention 提高 KV Cache 利用率；Continuous Batching 提高请求调度效率；高效 Kernel 与图优化降低单步执行开销。

## 离线推理

离线推理适合**不需要实时响应**的场景，如批量文本生成、数据标注、评测跑分等。入口是 `LLM` 类：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

prompts = ["什么是 KV Cache？", "用一句话介绍 PagedAttention。"]
outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print(out.prompt, "->", out.outputs[0].text)
```

`generate` 一次接收整批 Prompt，由引擎内部调度，不需要自己手动组 Batch。

## Chat Template

对话模型在 SFT（Supervised Fine-Tuning）阶段见过大量按 Chat Template 构造的数据，推理时也要按同样的格式拼 Prompt，否则效果会明显变差。以 Qwen 为例，模板展开后是这样：

```text
<|im_start|>user
1 + 1 = ?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

两种用法：

```python
messages = [{"role": "user", "content": "1 + 1 = ?"}]

# 1) 让 vLLM 自动套模板
outputs = llm.chat(messages, sampling_params)

# 2) 手动套模板，再走 generate
tokenizer = llm.get_tokenizer()
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,   # 末尾补上 "<|im_start|>assistant\n"，提示模型开始回答
)
outputs = llm.generate([text], sampling_params)
```

## 在线服务

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 2
```

服务提供 OpenAI 兼容接口：

| 接口 | 用途 |
| --- | --- |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/chat/completions` | 对话补全（服务端自动套 Chat Template） |
| `GET /v1/models` | 列出已加载的模型 |

用 OpenAI 客户端调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "什么是 Continuous Batching？"}],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

## 采样参数

| 参数 | 说明 | 默认值 |
| :-- | :-- | :-- |
| `temperature` | 控制随机性，越高越随机；0 表示贪心 | 1.0 |
| `top_p` | 核采样，只从累计概率前 p 的 Token 中采样 | 1.0 |
| `top_k` | 只从概率最高的 k 个 Token 中采样 | 随版本而定；0 或 -1 常表示不限制 |
| `max_tokens` | 最大生成 Token 数 | 随版本及调用方式而定 |
| `repetition_penalty` | 重复惩罚系数，>1 减少重复 | 1.0 |
| `stop` | 遇到指定字符串时停止生成 | None |
| `n` | 每个 Prompt 生成的序列数 | 1 |
| `seed` | 随机种子，设置后可复现结果 | None |

采样参数应按任务评估。需要更稳定的输出时可以从 `temperature=0` 开始，并显式设置 `max_tokens`。注意模型目录里的 `generation_config.json` 也可能改变默认采样值。

## 常用启动参数

| 参数 | 说明 | 设置建议 |
| :-- | :-- | :-- |
| `--model` | HuggingFace 模型标识或本地路径 | - |
| `--tensor-parallel-size` | 张量并行 GPU 数 | 根据模型大小决定 |
| `--gpu-memory-utilization` | vLLM 可使用的显存比例，扣除权重和激活后剩下的都给 KV Cache | 按显存余量和 KV 需求调整 |
| `--max-model-len` | 最大序列长度 | 按业务需要设置；越大，单请求可能占用的 KV 越多 |
| `--dtype` | 权重精度 | auto |
| `--quantization` | 量化方式 | None |
| `--max-num-seqs` | 单步最大并发序列数 | 按目标负载压测 |
| `--max-num-batched-tokens` | 单步 Token 预算（[第 6 章](06-chunked-prefill.md)） | 在 TTFT 与 TPOT 之间权衡 |
| `--enforce-eager` | 禁用 CUDA Graph 与编译（[第 7 章](07-graph-optimization.md)） | 调试时使用 |

## 张量并行

模型单卡放不下，或单卡性能达不到目标时，可以考虑张量并行（TP）：把每层的矩阵乘切到多张卡上，每层需要做跨卡的 `all_reduce`。TP 同时把权重和 KV Cache 分摊到多卡，留给 KV 的空间更大；代价是通信开销，收益取决于卡间互联带宽（NVLink 远好于 PCIe）。切分细节见 [nano-vllm 的并行 Linear](nano-vllm/02-model.md#并行-linear-族与张量并行)。
