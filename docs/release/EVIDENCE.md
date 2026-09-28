# Release acceptance evidence

Status: **release gate pending**. This records observations, not a model or hardware
allowlist. Every deployment configuration must validate its own artifacts and GPU
placements through the selected engine.

The 2026-09-23 UTC [implementation review](IMPLEMENTATION_REVIEW.md) supersedes
the earlier broad implementation-complete claim. Six defects reproduced in the
isolated audit script; those defects now have fixes and the physical test kit is
implemented. Physical and operator evidence is still pending. The
[172-case final acceptance design](FINAL_ACCEPTANCE.md) has not been executed.

## Software

The 2026-09-22 final full regression run passed 312 tests after registration and
qualification-harness hardening. Ruff lint/format, mypy (99 source files),
documentation links/four configs/CLI help, and lock checks passed. The wheel
built and installed in a fresh Python 3.11 environment; CLI help imported there
without optional kv-cached. A focused rerun of
39 passing tests covered validation, the qualification harness, and TUI before
the final full-suite rerun. The final wheel build and fresh native reinstall
passed after the selected-mode worker-policy change.

## Current software verification (2026-09-23)

- 373 tests passed after the launcher and mode-selection repairs; the 14 Torch deprecation warnings come from isolated experimental
  compatibility tests, not native engine activation.
- Ruff lint and formatting passed; mypy passed for 102 source files.
- Documentation links, four configurations, CLI/tool help and all example JSON passed.
- Acceptance mapping covers 172 cases / 296 applicable mode-case entries, 32 HTTP
  routes, 33 CLI commands, 43 TUI controls, 46 settings and 92 native modules.
- All six standalone audit reproductions now report `confirmed=false`.
- Lock check and source/wheel builds passed. A fresh Python 3.11.5 environment
  installed the wheel with native lockfile constraints; CLI and kit help passed.
  Neither Torch nor kv-cached was installed. Installed application source matches
  the wheel's source fingerprint.
- Wheel SHA-256: `248fd35595b8b54677508a19c501e70a5cd1ac7808e745c5f8814486f45cef26`.
- Lock SHA-256: `8062c8b5d3732498b1e953498b566c60f15f65677e17bfdab5a94ed1af6f76cc`.

These are software checks of the working tree and an isolated wheel installation,
not a claim that a clean-checkout hardware gate has passed.

## Launcher repair and operator startup smoke (2026-09-23)

The default checkout still selected the preserved beta `config.toml`, so plain
`./llmctl` encountered removed `prism_*` settings and a missing explicit mode.
Earlier isolated configuration tests had missed this operator entry point. The
local `.env` now selects a separate `config.release.toml` using native sleep, matching
the prior native RAM-cache mode, with `state/release/llm-rio.db`. Original beta
configuration/database hashes remain unchanged; the 59-model catalog, matching vault
and configuration are archived under ignored `db_backup/startup-repair-20260923-verified/`.
The archive's integrity, hashes and credential restoration were checked. Backup
verification now recognizes fully matching deleted-key tombstones instead of treating
their intentionally non-authenticating hashes as broken live credentials.

CLI and TUI startup now share the connection settings resolver. Invalid settings
produce concise errors without input values; missing explicit files fail. The TUI
checks settings before exiting and retains the start form on failure. Both Doctor
and Start service honor a valid `.env` config selector. CFG-03/04 and the manual
operator lane now explicitly cover the bare launcher with a beta default present.

Verification used the real checkout launcher and host inventory: two `./llmctl serve`
starts, followed by a pseudo-terminal run of bare `./llmctl` with keyboard navigation
through Diagnostics → Start service. Each service returned HTTP 200 for health and
authenticated capabilities in native sleep mode. CLI status passed, the original
new administrator credential survived restart, and Doctor reported no errors with
two GPUs. All owned smoke processes exited cleanly; port 8002 was unbound afterward.
At the time of that startup smoke, the new release database contained one administrator
and zero model registrations; see [the later data migration record](DATA_MIGRATION.md)
for its current contents.
No engine/model inference was run in this startup smoke.

The 71 focused tests and full 361-test suite passed, as did Ruff, mypy, documentation
and acceptance mapping checks. The rebuilt wheel was reinstalled in the existing
isolated Python 3.11.5 native environment with its previously lock-constrained
dependencies; CLI help and release configuration loading passed without Torch or
kv-cached. These checks do not replace the pending physical inference gates below.

## Mode selection follow-up (2026-09-23)

The operator selected queue while retaining a valid `[modes.vllm_sleep]` section.
The inactive-section rejection made this ordinary CLI/TUI override fail. Earlier
mode-switch tests used minimal configurations and missed populated mode sections.
Valid named sections now coexist; only the selected mode is used, and queue's cache
accessor cannot inherit dormant sleep policy. Unknown fields, invalid values and
incompatible active engine settings still fail. The user's queue selection and
configuration bytes were preserved.

Both queue and vllm-sleep were selected through the actual bare `./llmctl` start
form in a pseudo-terminal. Each launched successfully and returned HTTP 200 from
health and authenticated capabilities with the selected mode. Both processes exited
cleanly and port 8002 was free afterward. These were startup checks with an empty
catalog, not inference or hardware qualification. The physical CFG-05 procedure now
covers mode changes with populated sections left in the same file.

96 focused tests and all 373 tests passed; Ruff, formatting, mypy (102 files),
documentation and acceptance mapping checks passed. The rebuilt native wheel was
reinstalled in the isolated lock-constrained environment; its source matches the
checkout, both mode selections resolve correctly, and Torch/kv-cached remain absent.

## User and model state migration (2026-09-23)

The configured beta data was migrated into release schema v1 from its archived,
integrity-checked source. API key values, quota accounts, model grants, catalog
metadata, and profile evidence were checked after commit. The release target contains
59 models and 247 profile rows; inference and other operational history was omitted.
Ninety-five active queue configurations retain validated coverage of all 54 available
models. Forty-seven complete sleep profiles remain eligible and inactive. The 75
equivalent queue rows collapsed by release profile uniqueness and 29 ineligible or
unclassified rows are detailed in [DATA_MIGRATION.md](DATA_MIGRATION.md). A verified
pre-migration release snapshot is retained under ignored `db_backup/`.

## Isolated hardware smoke

Both runs used a fresh qualification database and one RTX PRO 6000 Blackwell
Max-Q GPU, UUID `GPU-7d5d200c-1bb6-32d6-6870-8451ad82fa21`. The local
Qwen3-4B-AWQ snapshot was a representative probe artifact, not a supported-model
list. Original lab databases and artifacts were untouched. Each mode completed
native validation, one non-streaming and one streaming chat request, and a
maintenance drain. Each request settled once with 32 completion tokens.

| Mode | Cold first request | Second streaming request | State observed | Drain |
| --- | ---: | ---: | --- | --- |
| queue | ~63.4 s | ~0.21 s | ready worker | ready, worker stopped and PID cleared |
| vllm-sleep | ~75.4 s | ~0.57 s | sleeping worker with host-resident weights, then wake | ready, worker stopped and PID cleared |

Detailed generated reports are deliberately ignored under `docs/release/results/`.
The LLM-RIO qualification services were stopped after their drains; an unrelated
external process on the other GPU was left untouched. A post-check showed no
qualification worker on the tested GPU and no LLM-RIO service process.

## Open release gates

- Execute the audited fixes through their mapped physical/operator acceptance cases.
  The final test kit is implemented; `scripts/qualify.py` is supplementary only.
- Run the automated gate from a fresh checkout after final changes.
- Complete context/concurrency, tenant fairness, mixed-model switching, TP,
  cancellation, maintenance and restart operator workflows on actual hardware.
- Qualify queue's optional llama.cpp/GGUF path when `llama-server` and an artifact
  are available. Neither was available for this smoke.
- Record at least 100 observed residency transitions and a one-hour mixed-load
  soak for **each** production mode with `scripts/native_acceptance.py`; compare accounting,
  telemetry, worker PIDs, leases, and ports after drain.
- Capture a request-level performance baseline from the archived previous
  code/environment. Only the pre-change GPU inventory and software test baseline
  were captured, so no throughput regression claim is possible yet. Investigate
  any regression and close release-blocking findings before deployment.

## Audit-fix follow-up

The six standalone audit findings now return `confirmed=false`. Regression fixes
cover pending local jobs, fitting TP fallback, sleep serialization, complete finite
evidence, actual engine identity, clone/selected-profile revalidation, owner locking,
and matching-vault archives. The real-GPU driver and manual ledger are available in
[the execution kit](MANUAL_GPU_TEST.md). Their offline oracle tests are synthetic.
The 100-cycle and one-hour gates have **not** been run against this changed build.
No release approval is inferred from earlier two-request smoke results.
