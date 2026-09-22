import os

if os.environ.get("LLM_RIO_KVCACHED_VLLM026_SHIM") == "1":
    from llm_rio.modes.kv_cached.kvcached_vllm_compat import install

    install()
