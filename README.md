# MLSys 笔记

LLM 推理系统、GPU 性能分析、CUDA 算子与模型量化的学习笔记。

在线阅读：<https://hychen11.github.io/MLSYS/>

## 目录

- **LLM 推理**（`docs/inference/`）：KV Cache 与性能指标、vLLM、PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill、图优化、投机解码，以及 nano-vllm 源码走读
- **性能分析**（`docs/performance/`）：Roofline 模型与 Nsight Compute
- **CUDA 算子**（`docs/cuda/`）：Reduce 从 V0 到 V7 的优化、Shared Memory Bank Conflict
- **量化**（`docs/quantization/`）：数值格式、量化基础、AWQ 与 SmoothQuant

## 本地预览

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve          # 打开 http://127.0.0.1:8000
```

推送到 `main` 后，GitHub Actions（`.github/workflows/deploy.yml`）会自动构建并发布到 GitHub Pages。首次使用需要在仓库 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。

## 写作约定

- 每个主题一个 Markdown 文件，放在 `docs/` 对应目录下，并在 `mkdocs.yml` 的 `nav` 中登记。
- 公式用 `$...$`（行内）和 `$$...$$`（块级）；流程图用 ```` ```mermaid ```` 代码块。
- 图片放在 `docs/assets/`，用相对路径引用。
