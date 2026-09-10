# Non-NVFP4 registration matrix — 2026-09-09

The local server was started on port `8002` and the submitted registration jobs
download and hardware-validate asynchronously.  A submitted profile is not
deployable until its job reports `COMPLETED / AVAILABLE`; use
`./llmctl models list` or `./llmctl models review NICKNAME` to inspect it.

No new NVFP4 profile was submitted.  `FP8` is the Q8 experimental arm; the
`INT4`, `AWQ`, `GPTQ`, and `compressed-tensors` sources are the Q4 arm.

| Requested family | Submitted non-NVFP4 profiles | Deferred / not submitted |
| --- | --- | --- |
| DeepSeek-V4-Flash | — | The only compatible Q4 distribution found was GGUF-only (`ggml-org/DeepSeek-V4-Flash-GGUF`). LLM-RIO cannot select a GGUF file from such a repository and it has no `config.json`; the full repository is 355 GiB. |
| Qwen3.8-Flash-Next | — | FP8 metadata is 172.8 GiB and the available INT4 source is 175.4 GiB. Either exceeds the server's conservative 176-GiB usable two-GPU budget after the 15% validation margin. |
| Qwen3.8-27B | `qwen3.8-27b-fp8`, `qwen3.8-27b-int4` | — |
| Qwen3.6-35B-A3B | `qwen3.6-35b-a3b-fp8`, `qwen3.6-35b-a3b-int4` | — |
| Qwen3.6-27B | `qwen3.6-27b-fp8`, `qwen3.6-27b-int4` | — |
| Qwen3.5-122B-A10B | `qwen3.5-122b-a10b-fp8`, `qwen3.5-122b-a10b-int4` | — |
| Qwen3.5-35B-A3B | `qwen3.5-35b-a3b-fp8`, `qwen3.5-35b-a3b-int4` | — |
| Qwen3.5-27B | `qwen3.5-27b-fp8`, `qwen3.5-27b-int4` | — |
| Qwen3.5-9B | `qwen3.5-9b-int4` | No Q8/FP8 source with suitable publisher support was found. The ordinary Qwen source is BF16, not Q8. |
| Qwen3-235B-A22B | `qwen3-235b-a22b-int4` | Q8/FP8 is too large; the official GPTQ-INT4 source is the Q4-only arm. |
| Qwen3-Next-80B-A3B | `qwen3-next-80b-a3b-fp8` | The publisher supplies Q4 as GGUF only, which the current LLM-RIO registration path cannot launch safely. |
| Qwen3-32B | `qwen3-32b-fp8`, `qwen3-32b-awq` | — |
| Qwen3-30B-A3B | `qwen3-30b-a3b-fp8`, `qwen3-30b-a3b-int4` | — |
| Qwen3-14B | `qwen3-14b-fp8`, `qwen3-14b-awq` | — |
| Qwen3-8B | `qwen3-8b-fp8`, `qwen3-8b-awq` | — |
| Qwen3-4B | `qwen3-4b-fp8`, `qwen3-4b-awq` | — |
| Qwen3-1.7B | `qwen3-1.7b-fp8`, `qwen3-1.7b-int4` | — |
| poolside Laguna-S-2.1 | `laguna-s-2.1-fp8`, `laguna-s-2.1-int4` | — |
| Gemma 4 31B IT | `gemma-4-31b-it-int4` | Google publishes an INT4 compressed-tensor QAT source, but no corresponding Q8/FP8 source. |
| Gemma 4 26B-A4B IT | — | No non-NVFP4 Q4 source suitable for direct vLLM registration was found. The official Q4 source is unquantized weights or GGUF; no Q8/FP8 source is published. |
| Gemma 3 27B/12B/4B IT | — | All official sources are gated and the Q4 distributions are GGUF or QAT-unquantized rather than directly launchable Q4 weights. See the gated-model procedure below. |
| Llama 3.3 70B Instruct | `llama-3.3-70b-instruct-awq` | The official base model is gated and BF16; no publisher Q8 checkpoint was selected. |
| OLMo-3 32B Think | `olmo-3-32b-think-8bit`, `olmo-3-32b-think-4bit` | — |
| GPT-OSS-120B | — | The available native MXFP4 snapshot is 182.3 GiB; its 15% validated footprint exceeds the usable two-GPU budget. It has no separate Q8 arm. |
| GLM-4.5-Air | `glm-4.5-air-fp8`, `glm-4.5-air-awq` | — |

## Gated-model procedure

1. Log into Hugging Face and accept the license on each model page you intend
   to use: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`,
   `google/gemma-3-27b-it`, and/or `meta-llama/Llama-3.3-70B-Instruct`.
2. Create a Hugging Face **read** token after access has been approved.
3. Stop the service, set `LLMRIO_HF_TOKEN` for the service process (or set
   `hf_token` in the non-committed local `config.toml`), then restart
   `./llmctl serve`. Do not put the token in this repository.
4. Submit the now-authorized source, for example:

   ```bash
   ./llmctl models add gemma-3-27b-it google/gemma-3-27b-it
   ./llmctl models add llama-3.3-70b-instruct meta-llama/Llama-3.3-70B-Instruct
   ```

These commands register the official high-precision source. They are not Q8
substitutes; keep them out of the Q8-vs-Q4 experiment until a suitable direct
Q8 source is chosen and validated.
