# Release qualification

A deployment manifest records host/GPU UUID inventory, dependency versions, source artifact
revisions, effective profile placements, and verified request limits. It describes the
installation tested, **not** an application model/hardware allowlist. LLM-RIO is a general
router; underlying engine support and actual registration validation determine eligibility.

## Automated gate

From a fresh checkout, install the locked native router/dev dependencies, run pytest, Ruff
lint/format, mypy, documentation/config/CLI checks, build a wheel, install that wheel in a
fresh environment, and exercise CLI help/import without kv-cached. Optional compatibility
tests run separately when Torch is available. Preserve accounting, streaming/disconnect,
maintenance, startup cancellation and teardown regression tests. Test each mode independently,
including simulated GPU counts 1/2/4/8 and blocked placement when telemetry is unavailable.

## Hardware gate

Use representative locally available artifacts that exercise router behavior: single-GPU
placement, TP when available, independent replicas, switching, native sleep/wake, local
sources, and queue's optional llama.cpp/GGUF path. Larger/smaller deployment environments
must validate their own configurations. No finite catalog is claimed to prove all engine
compatibility.

For each native mode:

1. Register and validate the selected artifacts against a fresh release database.
2. Run streaming/non-streaming inference and verify response protocol and accounting.
3. Exercise single-model saturation, mixed-model switching, context/concurrency limits,
   tenant fairness, maintenance, cancellation and restart recovery.
4. Record at least 100 **observed residency transitions** and a one-hour mixed-load soak.
   Requests alone do not count as transitions. Capture errors, queue wait, throughput,
   latency, GPU/host memory, worker PIDs and ownership/accounting snapshots.
5. Drain at the end. Require no unexplained request failures, leaked workers/reservations,
   stuck leases, or accounting mismatches. Investigate regressions against the pre-change
   hardware baseline before release.

The generic `scripts/qualify.py` tool collects traffic/telemetry/accounting evidence and
checks its observed transitions and duration; it does not replace context/TP/operator
acceptance. Select any callable models using `--models`, or use the service's current
callable catalog. Supply credentials through environment variables; reports contain no keys.
Store generated reports under ignored `docs/release/results/`. Missing evidence is a
pending gate, never a pass. Experimental results are reported separately.

## Deployment decision

Release only when the production workflow matrix passes, all release-blocking defects are
closed, docs match the installed interfaces, and hardware reports meet the gates. Keep the
previous environment and archived state available. Rollback must stop the new service first;
never run two owners on the same managed GPUs or old code against the release schema.
