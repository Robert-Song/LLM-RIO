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
- [ ] Shared contracts, client, persistence ownership, launch specifications
- [ ] Independent queue mode, including llama.cpp registration and validation
- [ ] Independent native sleep mode
- [ ] Experimental dependency/bootstrap isolation
- [ ] Registration sources, profile identity, audited advanced trust
- [ ] Interface cleanup and capability enforcement
- [ ] Tracked documentation, reproducible install, CI and package checks
- [ ] Hardware baselines and production model qualification
- [ ] 100 transitions and one-hour mixed-load soak per production mode

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
