import os

if os.environ.get("LLM_RIO_KVCACHED_VLLM026_SHIM") == "1":
    from llm_rio.kvcached_vllm_compat import install

    install()
