# Choosing a mode

There is no supported-model list or supported-hardware list maintained by LLM-RIO.
The engine determines compatibility. Validation measures the operator's actual
configuration; a result on one placement is not evidence for another placement.

| Tradeoff | Queue | vLLM sleep | kv-cached |
| --- | --- | --- | --- |
| Switching | Full cold launch | Prefer cached worker; wake from host RAM | Experimental elastic KV sharing |
| GPU overhead | Active engines only | Sleeping engines retain measured residual VRAM | Version-sensitive shared allocation |
| Host RAM | Normal engine needs | Cached weights plus transition peaks; limits enforced | Cached weights and compatibility-layer needs |
| External GPU pressure | Live admission, defer if unavailable | Reclaim sleepers or defer; measure wake peaks | Experimental results reported separately |
| Large single-model backlog | Preferred starting point | Cold-launch cost still applies | Additional integration complexity |
| Multiple frequent model switches | Repeated load cost | Often useful if RAM and residual VRAM fit | Not production-qualified |
| llama.cpp | Optional and explicitly selected | Unavailable | Unavailable |

Queue drains admitted work before stopping a worker, verifies teardown, prioritizes
oldest backlog, and preserves tenant fairness, TP fallback and independent replicas.
Sleep uses native level-1 sleep/wake and budgets live GPU memory, host RAM and swap.
Missing telemetry defers admission or forces conservative cleanup; it does not mean
zero memory usage. Preloading is configured within the selected cache mode.

Use [independent examples](../examples/config/queue.toml) as a starting point; the
[sleep example](../examples/config/vllm-sleep.toml) and
[experimental example](../examples/config/kv-cached.toml) have their own mode sections.
Switch by draining and stopping the service, changing `serving_mode`, restarting, and
validating profiles for the new mode. A queue profile cannot silently become a sleep
profile. Keep only one service on a managed set of GPUs.
