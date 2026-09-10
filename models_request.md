# Open-Source Model Serving Inventory for 2x NVIDIA RTX 6000 Pro Blackwell (192 GB VRAM)

## 1. Hardware Environment & Quantization Guidelines

### Server Hardware Specifications
- **GPUs**: 2x NVIDIA RTX 6000 Pro Blackwell (96 GB GDDR7 VRAM each)
- **Total Combined VRAM**: **192 GB**
- **Interconnect & Topology**: Tensor Parallelism ($TP=2$) across PCIe Gen 5 / NVLink
- **Native Execution Precisions**: Native NVFP4 (NVIDIA 4-bit floating point micro-scaling), FP8 (W8A8 / W8A16), FP16/BF16, AWQ, GPTQ, and GGUF

### Quantization Sizing Rules
1. **Models $\le 140$B parameters**: Fit comfortably within the 192 GB VRAM budget in **8-bit / Q8 / FP8** precision with ample headroom for KV cache (128K–262K context). For these models, **both Q8/FP8 and Q4/NVFP4** variants are provided.
2. **Models $> 200$B parameters** (e.g., DeepSeek-V4-Flash 284B, Qwen3-235B): Weights alone in 8-bit require 240 GB–300 GB VRAM (exceeding 192 GB). These models fit in **4-bit / Q4 / NVFP4** (~130 GB–155 GB weights, leaving 37 GB–62 GB for KV cache across the two GPUs). Therefore, **only the Q4 / NVFP4 variant** is specified.

---

## 2. Hardware Sizing & Compatibility Matrix

| Model Family | Model Name | Architecture | Parameters (Total / Active) | Q8 / FP8 Weight VRAM | Q4 / NVFP4 Weight VRAM | Fits on 2x 96GB? |
|---|---|---|---|---|---|---|
| **DeepSeek** | DeepSeek-V4-Flash | Sparse MoE + MLA | 284B / 13B | *~290 GB (Exceeds)* | ~145 GB | **Yes (NVFP4 only)** |
| **Qwen** | Qwen3.8-Flash-Next | Hybrid MoE + QSA | 125B / 6B | ~130 GB | ~68 GB | **Yes (Both Q8 & NVFP4)** |
| **Qwen** | Qwen3.8-27B | Dense Multimodal | 27B / 27B | ~29 GB | ~15 GB | **Yes (Both Q8 & NVFP4)** |
| **Qwen** | Qwen3.6-35B-A3B | Sparse MoE | 35B / 3B | ~37 GB | ~19 GB | **Yes (Both Q8 & NVFP4)** |
| **Qwen** | Qwen3.6-27B | Dense Multimodal | 27B / 27B | ~29 GB | ~15 GB | **Yes (Both Q8 & NVFP4)** |
| **Qwen** | Qwen3.5-122B-A10B | Sparse MoE | 122B / 10B | ~128 GB | ~65 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3.5-35B-A3B | Sparse MoE | 35B / 3B | ~37 GB | ~19 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3.5-27B | Dense | 27B / 27B | ~29 GB | ~15 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3.5-9B | Dense | 9B / 9B | ~10 GB | ~5.5 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-235B-A22B | MoE (Instruct / Thinking) | 235B / 22B | *~245 GB (Exceeds)* | ~132 GB | **Yes (Q4/NVFP4 only)** |
| **Qwen** | Qwen3-Next-80B-A3B | MoE (Instruct / Thinking) | 80B / 3B | ~84 GB | ~43 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-32B | Dense (Instruct / Thinking) | 32B / 32B | ~34 GB | ~18 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-30B-A3B | MoE (Instruct / Thinking) | 30B / 3B | ~33 GB | ~17 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-14B | Dense (Instruct / Thinking) | 14B / 14B | ~15 GB | ~8 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-8B | Dense (Instruct / Thinking) | 8B / 8B | ~9 GB | ~4.8 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-4B | Dense (Instruct / Thinking) | 4B / 4B | ~4.5 GB | ~2.5 GB | **Yes (Both Q8 & Q4)** |
| **Qwen** | Qwen3-1.7B | Dense | 1.7B / 1.7B | ~2.0 GB | ~1.1 GB | **Yes (Both Q8 & Q4)** |
| **poolside** | Laguna-S-2.1 | MoE (Agentic Coding) | 117.6B / 8.5B | ~122 GB | ~62 GB | **Yes (Both Q8 & NVFP4)** |
| **Gemma** | Gemma 4 31B IT | Dense Multimodal | 31B / 31B | ~33 GB | ~17 GB | **Yes (Both Q8 & NVFP4)** |
| **Gemma** | Gemma 4 26B-A4B IT | Sparse MoE Multimodal | 26B / 4B | ~28 GB | ~14 GB | **Yes (Both Q8 & NVFP4)** |
| **Gemma** | Gemma 3 27B IT | Dense | 27B / 27B | ~29 GB | ~15 GB | **Yes (Both Q8 & Q4)** |
| **Gemma** | Gemma 3 12B IT | Dense | 12B / 12B | ~13 GB | ~7 GB | **Yes (Both Q8 & Q4)** |
| **Gemma** | Gemma 3 4B IT | Dense | 4B / 4B | ~4.5 GB | ~2.5 GB | **Yes (Both Q8 & Q4)** |
| **Llama** | Llama 3.3 70B Instruct | Dense | 70B / 70B | ~75 GB | ~39 GB | **Yes (Both Q8 & NVFP4)** |
| **OLMo** | OLMo-3 32B Think | Dense Reasoning | 32B / 32B | ~34 GB | ~18 GB | **Yes (Both Q8 & Q4)** |
| **OpenAI** | GPT-OSS-120B | Sparse MoE | 120B / 15B | ~126 GB | ~64 GB | **Yes (Both Q8 & Q4)** |
| **GLM** | GLM-4.5-Air | Sparse MoE | 40B / 12B | ~42 GB | ~22 GB | **Yes (Both Q8 & Q4)** |

---

## 3. Detailed Model Inventory by Family

### 3.1 DeepSeek Family

#### DeepSeek-V4-Flash (284B MoE, 13B Active)
- **Architecture**: 4th-Gen Sparse Mixture-of-Experts with Multi-head Latent Attention (MLA) and integrated DeepSeek-Spark speculative decoding module.
- **Context Length**: 128K tokens.
- **Quantization Details**:
  - **Q8 / FP8**: Exceeds 192 GB total VRAM (~290 GB required). **Do not deploy in 8-bit.**
  - **Q4 / NVFP4**: Fits in ~145 GB VRAM across $TP=2$, leaving ~47 GB for KV cache and runtime overhead.
- **Repository Links & Identifiers**:
  - **Hugging Face (NVFP4 Official - NVIDIA)**: [nvidia/DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
  - **Hugging Face (vLLM FP8/NVFP4 Optimized)**: [RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8)
  - **Hugging Face (DSpark Speculative Checkpoint)**: [nvidia/DeepSeek-V4-Flash-nvfp4-DSpark](https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark)
- **vLLM Deployment Command**:
  ```bash
  vllm serve nvidia/DeepSeek-V4-Flash-NVFP4 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92
