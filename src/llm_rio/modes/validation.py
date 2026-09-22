def memory_budgets(mode: str, utilization: float) -> list[float]:
    if mode == "queue":
        from llm_rio.modes.queue.validation import memory_budgets as budgets
    elif mode == "vllm-sleep":
        from llm_rio.modes.vllm_sleep.validation import memory_budgets as budgets
    elif mode == "kv-cached":
        from llm_rio.modes.kv_cached.validation import memory_budgets as budgets
    else:
        raise ValueError(f"Unknown serving mode: {mode}")
    return budgets(utilization)
