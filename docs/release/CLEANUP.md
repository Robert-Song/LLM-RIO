# Final cleanup and release qualification

## Agreed scope
One application, isolated queue/vllm-sleep/kv-cached policies. Queue and native sleep
are production candidates; kv-cached remains experimental. llama.cpp is queue-only.
Local model artifacts are supported. Beta interfaces and schemas need no compatibility.
Original lab databases, credentials, configuration, and model files must remain intact.

## Baseline
Before changes: 320 tests pass; Ruff lint and mypy pass; eight files need formatting.
Hardware: two RTX PRO 6000 Blackwell Max-Q GPUs, 97887 MiB each. No LLM-RIO service
was running during initial inspection. Existing external allocations are not ours to stop.
Local beta databases were archived under ignored `db_backup/final-cleanup-baseline/`.

## Stages and acceptance evidence
The [follow-up implementation review](IMPLEMENTATION_REVIEW.md) found six
reproducible defects and additional unfinished paths. The earlier checked stages
described landed code, not completed acceptance; the checklist below corrects that.

- [x] Fix process/service ownership and document reviewed shared/client boundaries
- [x] Fix queue TP fallback and selected-profile llama.cpp revalidation
- [x] Fix native sleep cache serialization and evidence validation
- [x] Experimental dependency/bootstrap isolation
- [x] Fix cloned/edited profile jobs, actual engine identity and trust eligibility
- [ ] Prove all supported HTTP/CLI/TUI workflows and capability/role restrictions
- [x] Tracked documentation, reproducible install, CI and package checks
- [x] Design exhaustive final native acceptance and public-surface traceability
- [x] Implement the real-GPU driver, independent oracles and manual evidence ledger
- [ ] Execute the complete final native acceptance kit
- [ ] Full hardware workflow qualification, including llama.cpp/GGUF and TP
- [ ] 100 transitions and one-hour mixed-load soak per production mode

Current software checks and the limited two-mode hardware smoke are recorded in
[acceptance evidence](EVIDENCE.md). The [final acceptance design](FINAL_ACCEPTANCE.md)
specifies the remaining execution work. Design coverage is not a passing result.

## Defects established during audit
- TUI imports private CLI helpers and duplicates administration behavior.
- CLI default config argument overrides the environment's config selector.
- Validate buttons set flags without probing; multiple trust actions overlap.
- llama.cpp validator is disconnected from registration, uses a fixed port,
  and does not retain GPU reservations on unverified teardown.
- Queue eligibility depends on a vLLM-only sleep argument even for llama.cpp.
- Mode policy spans planner, workers, validation, profiles, and HTTP handlers.
- All docs and most acceptance scripts are ignored, so fresh checkouts lack them.

Passing unit tests does not qualify GPU compatibility or deployment readiness.
