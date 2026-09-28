# Final native GPU test: operator execution kit

Run this against an isolated acceptance installation, once in **queue** and once
in **vllm-sleep**, before production deployment. This kit uses actual validation
processes and actual `/v1/chat/completions` calls. It does not synthesize measured
profiles. Model and GPU choices are inputs, not a product compatibility list.

The [172-case catalog](ACCEPTANCE_CASES.md) remains the complete checklist.
The driver automates the sustained physical traffic lane; operator, fault,
CLI/TUI, TP/replica, and restoration cases require the additional work below.
A passing traffic report alone is **not release approval**.

## 1. Freeze and install the build

From the reviewed checkout:

```sh
uv sync --locked --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts/native_acceptance.py scripts/record_native_acceptance.py scripts/qualify.py scripts/audit_native_release.py scripts/check_acceptance_plan.py scripts/check_docs.py
uv run --no-sync ruff format --check src tests scripts/native_acceptance.py scripts/record_native_acceptance.py scripts/qualify.py scripts/audit_native_release.py scripts/check_acceptance_plan.py scripts/check_docs.py
uv run --no-sync mypy src/llm_rio
uv run --no-sync python scripts/check_docs.py
uv run --no-sync python scripts/audit_native_release.py
uv build
```

Keep that wheel and `uv.lock`. Install the wheel into a new native environment:

```sh
uv export --locked --no-dev --no-emit-project --no-hashes --format requirements-txt --output-file /absolute/acceptance/native-dependencies.txt
uv venv /absolute/acceptance/native-env
uv pip install --python /absolute/acceptance/native-env/bin/python --constraint /absolute/acceptance/native-dependencies.txt /absolute/checkout/dist/llm_rio-0.1.0-py3-none-any.whl
/absolute/acceptance/native-env/bin/python -m llm_rio --help
```

Use a separately pinned vLLM executable, or install the engine extra into this
native environment from the frozen package. Record its dependency versions in
operator evidence. Native installation does not need kv-cached. The runner checks
that the running router source matches the supplied wheel, and records both hashes.
Rebuild/reinstall/restart after any source change before collecting final evidence.

## 2. Select inputs and prepare two independent configurations

Select at least two distinct engine-supported chat artifacts A and B. Choose their
measured placements so they compete for at least one GPU. To test all supported
workflows also select the fixtures in [FINAL_ACCEPTANCE.md](FINAL_ACCEPTANCE.md):
a pinned HF revision, a local directory, a disposable local copy, a third artifact
for ordering/eviction, feature-capable fixtures, and physical TP/replica placements.
Select a GGUF file if you enable queue's optional llama.cpp backend.

Start from [queue.toml](../../examples/config/queue.toml) and
[vllm-sleep.toml](../../examples/config/vllm-sleep.toml). For each mode set:

- Absolute, new `database_path`, `model_store`, and `log_dir` under your acceptance
  directory. Give each mode its own database/vault and port ranges. Never use the
  lab database. Existing local artifact directories are reused in place.
- Explicit `managed_gpu_uuids`, obtained with `nvidia-smi --query-gpu=uuid,name --format=csv`.
  Use dedicated, otherwise idle GPUs for the healthy lane. Run the modes sequentially.
- An absolute `engines.vllm_executable`; also an absolute `llama_cpp_executable`
  and `enable_llama_cpp=true` only for a queue GGUF lane.
- Initial `engines.max_model_len` and `max_num_seqs` small enough for both first
  registration probes. The manifest's per-model `validation_overrides` then probe
  the final intended limits. No measurements are copied from beta state.
- Finite startup, drain, request and stream-idle deadlines sufficient for measured
  cold starts and queue wait. Keep production memory/reservation settings. Disable
  preloading during controlled idle cycles; test preloading separately.

Idle timers affect test duration: 100 queue cycles each include full cold loading
and teardown; 100 sleep cycles each wait for a real sleep and verify a same-PID
wake. The one-hour soak is **additional**. Keep the intended production timers for
final evidence; a shortened timer or reduced cycle/soak run is diagnostic only.

Copy [gpu-run.example.json](../../examples/acceptance/gpu-run.example.json) into
`queue-run.json` and `sleep-run.json` outside tracked source. Replace every path,
GPU UUID, source revision, and model name. Set each mode's URL and config path.
`build_files.wheel` identifies the installed wheel; `build_files.lock` identifies
its dependency lock. Add the exported `native-dependencies.txt` to `build_files`
as `native_constraints` so the installed transitive dependencies are recorded too.
Use the complete [deployment manifest](../../examples/acceptance/native-manifest.example.json)
to record host/driver/engine inventory, topology, thresholds, and scoped limitations.

Paste applicable [feature request examples](../../examples/acceptance/feature-requests.example.json)
into `feature_requests`, replacing model names with registered fixture names.
JSON output and full tool-result continuation run through real inference. Add
engine-supported reasoning, multimodal, long-context, stop, and multiple-choice
payloads as required by REQ-01–22. Do not silently omit claimed features: document
engine-specific applicability in the manual ledger before execution.

## 3. Start, validate, and run

In one terminal, from an empty working directory without a beta `.env`:

```sh
/absolute/acceptance/native-env/bin/python -m llm_rio serve --config /absolute/acceptance/queue.toml
```

Take the new admin key from this isolated service's first-start log. In a second
terminal set `LLMRIO_API_KEY` without putting it into the manifest or report. Then:

```sh
/absolute/acceptance/native-env/bin/python /absolute/checkout/scripts/native_acceptance.py --manifest /absolute/acceptance/queue-run.json --check
/absolute/acceptance/native-env/bin/python /absolute/checkout/scripts/native_acceptance.py --manifest /absolute/acceptance/queue-run.json --output /absolute/acceptance/evidence/queue
```

`--check` only validates inputs. The actual run:

1. Checks the running mode, source build, database path, GPU UUIDs and charging policy.
2. Drains existing test workers and checks independent GPU telemetry for idle devices.
3. Registers missing fixtures through the public API, waits for real probes, then
   revalidates the requested final launch limits. Existing failed jobs are retried;
   active jobs finish first. Failure aborts the gate with its diagnostic.
4. Creates Alice and Bob with independent, finite quota accounts and grants.
   Checks denied administration, invalid credentials, invalid inputs, and model lookup.
5. Calls each model with streaming/nonstreaming, multi-turn, system, Unicode and
   longer prompts; executes configured feature payloads.
6. Runs 100 controlled residency cycles. Queue requires actual worker teardown and
   empty independent GPU process telemetry. Sleep requires the request to use a
   retained worker ID/PID, then return that worker to SLEEPING.
7. Runs at least one hour of seeded mixed-artifact load from concurrent clients
   on both accounts, varying input/output lengths and streaming.
8. Drains and checks worker PIDs, queues, leases, validation reservations and ports.
   Reconciles every successful request ID, prompt/completion usage, settlement
   count, reservation charge, ledger debit/refund, and both account balances.

The runner writes `requests.jsonl`, `telemetry.jsonl`, `report.json`, and
`evidence-hashes.json`. It records TTFT, queue wait, latency percentiles, soak
throughput, artifacts/profiles, NVML-backed `nvidia-smi` samples and host memory.
Credentials and raw configuration are excluded. Failure or incomplete gates exit 1;
short diagnostic runs cannot produce `traffic_gate_passed=true`.

If it aborts during registration, use `models review NAME` and `maintenance status`;
registration may still be running when a client deadline expires. Stop the service
cleanly before another run. Use a new output directory for each attempt. Keep failed
reports; do not edit them to turn a failure into a pass.

Stop queue with Ctrl-C. Independently verify its GPU processes are gone, then start
sleep with its separate config and new admin credential. Repeat both commands using
`sleep-run.json` and an empty `evidence/sleep` directory.

## 4. Complete the operator and fault lanes

Use the exact procedures/expected outcomes in [the case catalog](ACCEPTANCE_CASES.md)
and [execution design](FINAL_ACCEPTANCE.md). These groups require separate evidence:

| Lane | Concrete execution and required oracle |
| --- | --- |
| Launcher / configuration | Leave a beta `config.toml` in a disposable checkout; select a separate release config with `.env`, then repeat with a shell selector. Keep valid queue and sleep sections in the release file. Run bare `./llmctl`, select each mode in Diagnostics → Start service, query health and authenticated capabilities, stop and restart. Confirm selected mode/database, isolated mode settings and stable admin credential. Repeat selection through TOML, environment and CLI. Submit an invalid path and beta settings in the start form: it must remain open with values retained and a clear error without credential values. Correct the config and submit again. |
| Registration / selected profiles | Register HF and local sources; fail a job deliberately, correct it and retry. Clone with changed context, then `models validate CLONE --profile ID`. Edit and revalidate a selected vLLM and queue/llama.cpp profile. Inspect the resulting engine arguments, context, UUID placement and binding. Repeat via HTTP and TUI. |
| Local artifact drift | Register a **disposable copy**. Change a content file after validation. Both model listing and inference must report it non-callable; no quota/lease is admitted. Revalidate. Also change it during a probe: publication must fail. Never edit original weights for this test. |
| Queue pressure / fairness | Warm A, hold its GPU with long A requests, then enqueue B and C in known order from different accounts. Record submission/admission times and worker IDs. Verify oldest backlog, within-account fairness, independent replicas, queue-full refunds, TP reservations and the TP1-busy/TP2-free case. Test available physical TP/replica topologies and the 1/2/4/8 simulated suite separately. |
| Sleep cache pressure | Warm A/B/C; wait for sleep; call a cached fixture again and verify its PID. In a separate configuration set a cache budget that requires eviction. Verify LRU eviction and a cold load afterward. Exercise reserved RAM headroom, cgroup limits, swap policy and sleeping residual/wake peaks. Healthy-lane limits are not changed midway. |
| External GPU pressure | Use a separate, explicitly owned GPU allocator on the selected test GPU. Start pressure before a queued cold/wake request. Record allocator PID and bytes. RIO must defer/evict safely; release only your allocator and observe progress. Never kill unrelated GPU jobs. Missing telemetry and uncertain teardown must retain reservations. |
| Disconnect / failure | Start a long streaming call with `curl -N` and a unique `X-Test-Run-ID`; interrupt it before assignment and midstream in separate runs. Kill only an identified acceptance worker in a separate fault run. Confirm terminal request state, one settlement/refund, zero stuck leases, healthy next request and cleanup. Exercise start/sleep/wake interruption and finite watchdogs as specified in REQ/SLP/OPS cases. |
| Limits / engine features | Send prompts near the validated context limit and one over it, cross the measured concurrency limit, then test requested choices and output ceilings. Exercise full tool/reasoning/JSON/multimodal conversations on supported fixtures. Engine errors must propagate without quota leaks. Record unsupported features as scoped limitations, not passes. |
| Roles / quotas | Admin, TA, Alice, Bob and a second Alice key: grants/revoke, rotate, restore, shared-account simultaneous reservations, quota exhaustion, zero usage and compaction. Reconcile account balances and per-ID rows before/after. Repeat supported admin actions via CLI and TUI, checking preserved input, disabled prerequisites, stable selection and duplicate submission. |
| Ownership / restart | Start a second **test** service with the same DB, then a different DB using the same GPU UUIDs. Both must fail before reconciliation affects the first. Gracefully restart the first, then interrupt it during startup/validation in separate runs. Check process groups, ports, recovery and idempotent settlement. |
| Backup / rollback | Stop the test service, archive database+vault+config, verify all hashes and credential decryption, restore into a new isolated location with the matching wheel/environment. Authenticate with a restored key, list profiles, run a real model call, and drain. Keep old code away from the new schema. |
| Optional llama.cpp | In queue only, validate a real GGUF, stream and nonstream it, revalidate the selected profile after context/layer changes, exercise concurrent validation port allocation, mixed vLLM/GGUF switching, limits and teardown. Missing GGUF/engine blocks qualification of an enabled backend. |

For example, the HTTP selected-profile operation is:

```sh
curl --fail-with-body -H "Authorization: Bearer $LLMRIO_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"profile_id":"PROFILE_ID","validation_overrides":{"max_model_len":4096}}' \
  http://127.0.0.1:18880/staff/model-jobs/JOB_ID/retry
```

The equivalent CLI is `llm-rio models validate NAME --profile PROFILE_ID`;
the TUI location is Models → Profiles → Revalidate. All run probes. Trust saved
measurements is a separate advanced action and cannot repair invalidated evidence.

## 5. Record each case and evaluate the complete release evidence

Predeclare applicability before running. A scope file is a JSON object such as
`{"vllm-sleep:LCP-01":"Optional llama.cpp is queue-only"}`; only include actual
mode/case pairs in the catalog. Missing hardware is BLOCKED, not a convenient
NOT_APPLICABLE. Run narrower configurations only with explicit qualification limits.

```sh
python /absolute/checkout/scripts/record_native_acceptance.py init --output /absolute/acceptance/manual.json
python /absolute/checkout/scripts/record_native_acceptance.py record --ledger /absolute/acceptance/manual.json --mode queue --case REG-05 --status PASS --evidence /absolute/acceptance/evidence/reg-05.json --notes 'Pending local registration returned model_unavailable; balance unchanged.'
python /absolute/checkout/scripts/record_native_acceptance.py report --ledger /absolute/acceptance/manual.json --traffic /absolute/acceptance/evidence/queue/report.json /absolute/acceptance/evidence/sleep/report.json
```

Use the native environment's Python in these commands. `init --scope FILE` accepts
predeclared exclusions. Record PASS only after checking every expected outcome of
that case across all required interfaces, topologies and variants. Use an evidence
bundle with one result record per variant, following the result schema in the design.
The tool does not infer a pass from a filename. Record FAIL/BLOCKED with
the actual evidence when an outcome cannot be demonstrated. The ledger retains
history and hashes evidence. The report refuses incomplete cases, changed evidence,
failed traffic gates, and differing builds. Your additional personal tests can be
recorded alongside the official case evidence.

Review throughput/latency against a controlled pre-change baseline with the same
artifacts, placement, limits and load. The historical smoke is not that baseline.
Investigate regressions before deployment. Final production approval requires both
native traffic gates, all applicable manual gates, no unresolved release defects,
and matching installation/operations documentation.
