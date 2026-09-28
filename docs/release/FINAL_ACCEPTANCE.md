# Final native production acceptance design

Status: **KIT IMPLEMENTED; PHYSICAL ACCEPTANCE NOT EXECUTED**. Scope: queue/vLLM, vLLM sleep, and a separate
queue/llama.cpp lane when that optional backend is shipped. kv-cached is excluded.
The [implementation review](IMPLEMENTATION_REVIEW.md) records the audited fixes.
Start with [the runnable operator kit](MANUAL_GPU_TEST.md); all physical/operator
cases must pass before production approval.

This is the final test specification: [172 concrete cases](ACCEPTANCE_CASES.md),
[machine-readable cases and surface mappings](native-acceptance.json), real-user
journeys, fault cases, and physical release gates. It covers every currently
exposed native API route, CLI command, TUI button and configuration field, with
traceability for all 92 native source modules. That is
a finite, auditable coverage claim; it cannot prove every possible model, driver,
request payload or interleaving.

## 1. Compatibility and selection of test inputs

LLM-RIO has no fixed supported-model or supported-GPU list. vLLM/llama.cpp determine
engine compatibility; LLM-RIO validates the exact artifact, effective launch and
placement. Test inputs are selected by the behaviors needed below. Record their
identities in a run manifest without turning that manifest into a product allowlist.

| Fixture | Selection rule and purpose |
| --- | --- |
| A, B | Two distinct, engine-supported model artifacts that can each serve chat. Use placements that force contention on at least one shared GPU. Different aliases of the same artifact are useful for clone tests but do not replace this mixed-artifact test. |
| C | A third independently routable artifact for ordering and cache eviction; it may reuse A's architecture but must have a separately validated identity. |
| H | A pinned immutable Hugging Face revision, including a private/gated variant if that deployment claims authenticated registration. |
| L | A disposable local directory copy used for change/race tests. Never mutate original lab artifacts. |
| T | A measured TP placement needing at least two GPUs, plus a profile set exposing TP1-busy/TP2-free fallback. Larger simulated inventories cover scheduling beyond available physical devices. |
| R | At least one model validated on more than one independent single-GPU placement to exercise replicas. |
| F | Engine-supported tool, reasoning, structured-output and multimodal fixtures as applicable. The same artifact may cover several features. Unsupported engine features are explicitly scoped, not silently marked passed. |
| G | Operator-selected GGUF and pinned `llama-server` for the optional queue backend. Missing G blocks qualification of that enabled backend, not a vLLM-only configuration with the backend disabled. |

Run scheduling/ownership contract cases on 1/2/4/8 simulated GPUs. Separately run
available physical single-GPU, multi-GPU TP and replica cases. A simulator result
is never recorded as physical acceptance. Missing hardware yields BLOCKED for
that physical case and a precise qualification limitation; it does not imply the
router has a permanent hardware restriction.

## 2. Freeze and isolate the run

Use [the manifest example](../../examples/acceptance/native-manifest.example.json)
as a required input template. Replace placeholders, record resolved revisions and
engine identities, and fix the seed, workload, thresholds and fault schedule
**before** collecting acceptance results.

- Freeze the reviewed source revision, any patch hash, wheel hash, dependency
  lock, actual engine environment/version, effective config and artifact hashes.
  A commit ID alone does not identify a dirty working tree.
- Use a new per-run directory, for example
  `state/release/acceptance/RUN_ID/{queue,vllm-sleep}/`. Keep database, vault,
  logs and ports distinct for each mode. Preserve lab state separately.
- Confirm exclusive test GPU ownership before starting. Never terminate an
  unrelated service/process. Multiple-owner tests run only against instances
  created by this suite and must fail safely before any existing worker is killed.
- Keep credentials in process environment or a protected local vault; replace
  them with key IDs in reports. Do not capture key-creation stdout unredacted.
- Configure finite startup, request, stream-idle, drain and transition timeouts
  for failure cases. Derive healthy cold/wake deadlines from measured evidence,
  then record explicit numeric values in the manifest.
- Test one mode at a time against shared physical GPUs. Stop and independently
  verify cleanup before launching the next mode.

Mandatory preflight commands from the frozen checkout:

```sh
uv sync --locked --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts/check_docs.py scripts/qualify.py scripts/check_acceptance_plan.py scripts/audit_native_release.py scripts/native_acceptance.py scripts/record_native_acceptance.py
uv run --no-sync ruff format --check src tests scripts/check_docs.py scripts/qualify.py scripts/check_acceptance_plan.py scripts/audit_native_release.py scripts/native_acceptance.py scripts/record_native_acceptance.py
uv run --no-sync mypy src/llm_rio
uv run --no-sync python scripts/check_docs.py
uv run --no-sync python scripts/audit_native_release.py
uv build
```

The six-defect audit now passes after fixes. It is a synthetic regression check,
not a hardware gate. Validate installation/help/native isolation from the wheel
in a fresh environment and run from outside the source checkout.

## 3. Test personas and traffic

Create a host operator, one admin, one staff key, users Alice and Bob on separate
quota accounts, and a second Alice key sharing her account. Add a revoked key,
an invalid credential and an unauthenticated client. Grants differ by user and
change during explicit access cases. Server permission enforcement is tested
independently of whether a control is visible in the TUI.

Use independent clients rather than one synchronous caller:

| Persona | Behavior |
| --- | --- |
| Interactive Alice | Multi-turn chats, 16–128 output tokens, 1–5 second think times, frequent streaming. |
| Batch Bob | Concurrent 256–1024 token requests, bursts crossing worker capacity and queue limits, A/B model mix. |
| Same-account Alice client | Competes with the first Alice key to verify quota-account fairness and atomic shared credit. |
| Tool/reasoning client | Function declaration → model tool call → tool result → continuation; reasoning and structured output on supported fixtures. |
| Impatient client | Cancels before assignment, before headers and midstream; slow reads and TCP reset in separate labeled cases. |
| Staff/admin | Registration, review, revalidation, grants, quota/profile changes, refresh, drain and resume at scheduled points. |

The healthy soak uses valid requests only. Invalid requests and injected failures
run in separately labeled cases so they cannot conceal unexplained failures in
the healthy workload.

## 4. Real interface execution

API requests use actual HTTP to a real service process. CLI actions use the
installed `llm-rio` executable in a subprocess with explicit connection/config
environment. TUI actions use Textual Pilot against that same real API; do not
replace `admin_client`, registration, supervisor, quota storage or engine probes
with mocks in this lane. Capture redacted screenshots/transcripts and inspect
the resulting state independently via API and read-only storage observation.

For every mapped management action, run its positive case through every advertised
surface in [native-acceptance.json](native-acceptance.json). Reset or namespace
fixtures between surface runs. Also apply these modifiers to each surface:

1. Missing/invalid credentials, allowed role and denied role.
2. Missing object, invalid field, unmet capability/prerequisite.
3. One normal submission, concurrent duplicate submission, refresh while pending.
4. Network failure before submission and after server acceptance; retry safely.
5. Confirmation cancel and confirm where supported; no mutation on cancel.

Unsupported combinations are explicit expected 4xx outcomes, not omitted cases.
`POST .../profiles/{profile_id}/{action}` expands to `enable`, `disable`, and an
unknown action. Destructive key actions use disposable keys; never delete the
only run administrator. TUI coverage includes keyboard-only navigation and
80×24/120×40 terminals, not just button text assertions.

`scripts/check_acceptance_plan.py` checks that the 32 API routes, 33 CLI command
leaves, 43 static TUI controls, 44 native config fields and 89 source modules still
have cases. It also detects direct cross-mode imports. **That checker proves design coverage,
not execution or correctness.** New dynamic controls or workflow semantics still
require reviewer attention.

## 5. Canonical end-to-end journeys

These journeys connect the catalog cases into realistic sequences. Execute each
from a clean mode database, then repeat against restarted persisted state.

### Journey A: first installation to first user response

1. Operator starts the selected mode with the isolated configuration. Verify
   health, capabilities and the bootstrap credential. No model is callable yet.
2. Admin creates staff, Alice and Bob; staff registers H and L and grants only H
   to Alice. Review progress and try Alice's request while validation is pending.
3. Observe native validation waiting for maintenance. Admin drains, waits for
   full teardown, and watches real probes complete. Resume.
4. Alice lists models, sends nonstream and stream calls, then a tool/reasoning
   turn where supported. Bob's ungranted request fails without engine execution.
5. Staff grants Bob; Bob's call succeeds. Reconcile each account and request ID.

Representative commands, with values supplied by the manifest:

```sh
export LLMRIO_CONFIG_FILE=/absolute/test/run/queue/config.toml
export LLMRIO_API_URL=http://127.0.0.1:18880
llm-rio serve --config "$LLMRIO_CONFIG_FILE" --mode queue
# In a separate operator terminal, after secure credential setup:
llm-rio capabilities
llm-rio keys create alice --role user --limit 1000000
llm-rio models add acceptance-a "$HF_REPO" --revision "$HF_REVISION" --grant-to alice
llm-rio models add acceptance-b --local-path "$LOCAL_MODEL_DIR" --engine vllm
llm-rio models review acceptance-a
llm-rio maintenance drain
llm-rio maintenance status
# Wait for completed probes and maintenance readiness; do not use a fixed sleep.
llm-rio maintenance resume
```

Send a user request with unique correlation headers; repeat with `stream: true`
and a real SSE parser rather than collecting the whole body before examination:

```json
{
  "model": "acceptance-a",
  "messages": [
    {"role": "system", "content": "Reply briefly."},
    {"role": "user", "content": "Return the marker RIO-CASE-REQ01 and say hello."}
  ],
  "max_tokens": 64,
  "temperature": 0,
  "stream": false
}
```

Use `X-Test-Run-ID`, `X-Request-ID`, `X-Client-Worker` and `Idempotency-Key` from
the case journal. A language-model marker is diagnostic only; correctness relies
on protocol, routing identity and usage, not exact generated prose.

### Journey B: busy shared service and switching

Alice starts a long A stream. Bob sends a B batch, then Alice queues C. Verify
oldest-backlog and tenant progress while an admin refreshes the dashboard. Queue
must drain admitted A work and verify teardown before GPU reuse. Sleep mode must
drain, level-1 sleep, account for residual memory, admit B safely, and wake cached
A with the same PID when eligible. Add replicas/TP cases, then cancel a queued
and an active request. Inspect account balances and resource ownership after
each phase.

### Journey C: correcting a model and recovering from a failed edit

Admin changes a launch-affecting profile value. Confirm inference is blocked
pending real measurement, while already admitted work follows its drain policy.
Validate the selected profile and check the actual launch arguments. Clone with
generation defaults, then clone with a changed context/YaRN setting and validate
that clone independently. Modify disposable local content and verify all cold,
warm, preload and wake routes block detected drift. Test eligible fingerprint
trust and every forbidden trust variant. A manual override cannot rescue missing
or invalidated measurements.

### Journey D: maintenance, crash, restore and rollback

Drain during mixed traffic, confirm queued refunds and completed active work, run
revalidation, then resume. In separate cases kill only the test router/engine at
each lifecycle boundary, restart and verify ownership reconciliation. Back up
the stopped test database with its matching vault/config and restore into a new
isolated directory; authenticate using restored credentials. Rehearse rollback
with the old code and its old schema after stopping the new instance.

## 6. Oracles independent of the action under test

Every case records a before/after snapshot and a monotonic timestamped journal.
An HTTP 200, a matching UI label, or a successful worker command alone is not a
pass. Use at least two independent observations for resource safety.

### Request protocol and accounting

- Correlate by exact request ID, account/key ID, logical model, worker ID and run
  ID. Do not compare unordered model/token totals: two requests can exchange
  charges and still have matching totals.
- Record request admission, engine assignment, first token, last token and
  completion times. Separate queue wait, cold load/wake, TTFT and generation
  latency. Parse streams incrementally, including Unicode boundaries and DONE.
- For successful actual-usage charging: persisted prompt/completion usage equals
  authoritative response usage, including zero; net charge equals the documented
  engine total. For requested-maximum mode: charge equals the reserved maximum
  while actual prompt/completion telemetry remains accurate.
- For failure/cancellation without authoritative usage: check the documented
  fallback estimator against bytes/events actually observed. Such a result is
  not proof of engine-exact token accounting. Keep the charging-policy decision
  visible; never silently treat an estimate as authoritative evidence.
- Inspect reservation and ledger state through a read-only observer after drain.
  Rejected-before-reservation requests have no charge; admitted/terminal requests
  have one terminal completion and one effective settlement. Replaying cleanup,
  request IDs or compacted idempotency records cannot create another charge.
- At quiescence, for a finite account:
  `final balance = initial balance + administrative adjustments - net charges`.
  During traffic subtract outstanding reservations as well. Reconcile current
  period and lifetime totals before/after reset and compaction.

### Ownership and residency

- Observe API snapshots, persisted worker/events, exact process-group membership,
  NVML process usage and port listeners. Record GPU UUIDs rather than relying on
  unstable numeric device indices.
- Queue: no overlapping conflicting GPU owners through LOADING, READY, DRAINING
  or STOPPING; no reuse until process-group exit and GPU-context disappearance.
- Sleep: at most the allowed active placement per GPU, with retained sleepers
  explicitly measured. Live free memory must cover the launch/wake requirement;
  credit only verified residual usage of the actual target. RAM/swap eviction
  must not target active requests or unrelated processes.
- At final drain: no queue entries, request leases, live quota reservations,
  validation GPU ownership, reserved ports, owned live workers/children or GPU
  contexts. COLD/STOPPED rows have no PID. Verify externally even if the service
  reports maintenance ready. Historical terminal rows are not leaks.

### Limits and fairness

Use exact tokenizer/engine-tokenized fixtures for context boundaries; the router's
rough text estimator is not an independent oracle. Include tools and multimodal
inputs, whose token cost differs from plain text. Record effective verified
context/concurrency rather than assuming config values prove support.

Tenant fairness uses quota-account identity and charged/reserved work, not equal
request counts. Record service opportunities and oldest backlog over time. Fix
a bounded-progress deadline in the manifest from residency, drain, load and tick
limits; a tenant with eligible capacity may not starve indefinitely. For strict
ordering/fairness assertions, use controlled finite backlogs; random soak is
supplementary evidence.

## 7. Fault injection lanes

Use real engines for healthy execution and recovery from owned-process death.
Use a separate explicitly labeled protocol fixture/transport proxy to generate
otherwise nondeterministic malformed streams, zero usage and stalled endpoints.
Those fixture results prove router handling, not engine compatibility.

Inject faults only into a run-owned process, disposable database or private
transport. GPU/RAM pressure helpers have a recorded PID, maximum allocation and
timeout. Test host/cgroup/swap pressure inside an isolated resource boundary;
never exhaust the host globally or disable system telemetry for other services.
Storage faults operate on test state, never the original lab files. Record the
exact injection, expected affected requests, cleanup and a successful recovery
request. Unexplained errors outside that affected set fail the case.

### State-boundary coverage

Use these paths as explicit assertions within the linked cases. Observe initial
worker creation separately from an already persisted COLD row. Repeated calls
may be idempotent; they may not fabricate a transition or erase ownership.

| Native path | Required cases and causal assertion |
| --- | --- |
| Creation → LOADING → READY | REQ-01/02, QUE-01; health succeeds before first admission, with one process/port/placement owner. |
| READY → DRAINING → STOPPING → COLD | QUE-02/03, OPS-01; last admitted completion precedes teardown, which precedes GPU/port reuse. |
| READY → OFFLOADING → SLEEPING | SLP-01; level-1 acknowledgement and live residual observation precede cached state. |
| READY → DRAINING → OFFLOADING → SLEEPING | SLP-02; active requests finish before sleep; later arrivals cannot enter the draining worker. |
| SLEEPING → WAKING → READY | SLP-03/04/06; live admission covers every GPU, PID persists, wake completes before new admission. |
| SLEEPING → STOPPING → COLD | SLP-07/09/18; cache eviction/maintenance reason recorded; full process/GPU teardown verified. |
| LOADING/OFFLOADING/WAKING/READY → failed teardown or COLD | QUE-13/15, SLP-14/15, OPS-09/10; uncertainty stays owned/unroutable, successful cleanup releases exactly once. |
| ACTIVE → DRAINING → MAINTENANCE_READY → ACTIVE | OPS-01/02, REG-07/08; pending requests rejected/refunded, workers fully unloaded, resume refused during probe ownership. |
| Crash during any path → startup recovery | REG-12, OPS-04/05, SLP-20; reconcile persisted/actual processes before accepting traffic; no assumed recoverable RAM cache. |

Forbid queue OFFLOADING/SLEEPING/WAKING, admission outside READY, partial-TP
ownership, a second active owner during STOPPING, READY after failed wake, and
maintenance-ready while owned processes or validation reservations remain.

### Error assertions

Pin exact code/status pairs in the frozen runner. The current baseline below may
be deliberately changed while closing review findings; document any replacement
before the final run. A test that accepts any error status is too weak.

| Condition | Current status / code or contract |
| --- | --- |
| Missing/invalid credential; denied role | 401 `invalid_api_key`; 403 `permission_denied` |
| Missing model; ungranted model | 404 `model_not_found`; 403 `model_not_allowed` |
| Registered but unavailable; detected local drift | 409 `model_unavailable`; 409 `artifact_changed` |
| No eligible measured profile | 503 `model_verification_required` or `model_backend_verification_required`, according to absence versus incompatibility |
| Duplicate nickname/state; duplicate request/idempotency key | 409 `state_conflict`, `request_id_conflict`, or `idempotency_conflict` |
| Quota or queue full | 429 `quota_exceeded` or `queue_full`, with no retained rejected-request reservation |
| Native maintenance; premature resume | 503 `service_maintenance`; 409 `validation_in_progress` / `maintenance_not_ready` |
| Invalid schema; unsupported engine; invalid enable/trust | 422 `invalid_request_error` / `unsupported_engine`; 409 `profile_ineligible` / `measurements_incompatible` |
| Limit violation | 400 `context_length_exceeded`, `max_tokens_exceeded`, or `n_exceeded` |
| Nonstream timeout/transport failure | 504 `worker_request_timeout`; 503 `worker_unavailable` |
| Failure after streaming headers | Terminal SSE error with specific `worker_stream_*` / `worker_protocol_error` / `worker_transport_error` code; HTTP status alone is not the oracle |

Admin/staff missing-object paths currently also use FastAPI's `detail` envelope.
Record that envelope explicitly or normalize it before freezing the interface.
Inference errors must preserve the documented OpenAI-compatible envelope.

## 8. Controlled transitions and one-hour soaks

Run each production mode separately:

1. Execute **100 complete controlled A/B residency cycles** (a stronger target
   than merely counting 100 state changes). Issue a valid request on both sides
   of each switch. Count cycles from ordered lifecycle events plus PID/NVML
   observations, not from request count or polling alone. Every counted queue
   unload proves teardown; every counted cached sleep/wake proves PID continuity.
2. Run a separate **60-minute healthy mixed-load soak** after warmup. Use at least
   two distinct artifacts, two accounts, mixed sizes, streaming/nonstreaming,
   seeded arrivals and occasional bursts above serving capacity. Choose demand
   that actually causes switching; fitting all models permanently does not test
   transitions. Count duration only while load and telemetry collectors run.
3. Run the failure/recovery schedule separately, then a healthy recovery phase
   and complete drain. A collector crash/cancellation leaves a failed/incomplete
   run, executes cleanup and records surviving owned resources for recovery.

Record errors by expected/unexpected class, per-model/account request and token
throughput, queue wait/TTFT/end-to-end p50/p95/p99/max, RSS/PSS/swap, GPU memory,
worker/child PIDs, port/reservation counts, and transition/cold/wake latencies.
Telemetry sampling must be frequent enough for memory peaks; lifecycle events
are still the authoritative transition sequence. Detect event/telemetry gaps.

Capture comparable previous-version baselines with the same artifacts, placement,
dependency environment, limits, seed and request trace. Proposed default review
thresholds are >10% throughput loss or >10% p95 latency increase over baseline
variability; thresholds are fixed in the manifest before execution. Crossing one
requires investigation and a recorded disposition, never silent acceptance.
Any unexplained error, leak, mismatch or starvation fails regardless of speed.

## 9. Execution tools and manual lanes

The older `scripts/qualify.py` is a supplementary collector. The real-GPU
`scripts/native_acceptance.py` implements registration, traffic, controlled cycles,
soak, telemetry and exact accounting. `scripts/record_native_acceptance.py` tracks
manual outcomes and checks final evidence. The following responsibilities are
split between those tools and the operator according to [the kit](MANUAL_GPU_TEST.md):

| Component | Required behavior |
| --- | --- |
| Run manifest/preflight | Validate explicit native mode, real artifact/engine identities, isolated state, ownership, budgets and required fixtures. Never auto-select a beta database. |
| Fixture lifecycle | Public registration/validation, key/grant creation, secure credential handling, stable IDs, isolated case namespaces and cleanup. |
| HTTP traffic client | Unique request IDs, incremental SSE parsing, timestamps, cancellations and exact captured usage. No retry of inference without recorded idempotency semantics. |
| CLI/Pilot drivers | Actual installed commands and live TUI clients; capture exit code, redacted output and public postcondition. |
| Observers | Event/request ledger, readonly DB, process groups, NVML and host memory with gap detection and clock alignment. |
| Fault controller | Whitelisted run-owned targets, bounded pressure, named fault schedules, guaranteed cleanup and recovery probes. |
| Oracle/result writer | Per-case assertions and per-request reconciliation; append evidence continuously; distinguish PASS/FAIL/BLOCKED/NOT_RUN/approved N/A. |
| Release aggregator | Expand mode/surface/topology/capability matrix, reject missing results and unresolved findings, verify tested artifact hashes and explicit optional-backend scope. |

CLI/Pilot, fault injection, restore rehearsal and topology variants remain operator
lanes. The tools never mark them passed from internal planner tests. Unit/fault contracts complement the public-flow lane.
Do not reuse historical fixture helpers that automatically recreate measurement
bindings when a profile is edited.

## 10. Result and release decision

The result key is `(run_id, case_id, mode, interface, topology, variant)`. Record
timestamps, artifact/config/wheel hashes, seed, expected outcome, actual outcome,
assertions, evidence paths and any linked defect. Keep full secrets outside the
report. Use the [result example](../../examples/acceptance/native-result.example.json)
as the minimum result shape.

PASS requires all required assertions and evidence. FAIL means an assertion or
unexpected error. BLOCKED means missing prerequisite. NOT_RUN and XFAIL are not
passes. N/A is allowed only for a predeclared unsupported engine feature or
explicitly disabled optional backend, with reason and scope recorded; it cannot
excuse an exposed native operation or replace unavailable physical evidence.

Release requires:

- All P1/blocking review findings closed, plus documented disposition of other
  findings and any approved architecture adjustments.
- Full required case expansion passes through advertised surfaces; expected
  failures match their reason/status and preserve invariants.
- Both native modes complete controlled transitions and their one-hour healthy
  soak; required physical TP/replica/recovery/operator evidence exists.
- No unexplained errors, stuck leases, leaked processes/reservations/ports,
  accounting mismatches or uninvestigated performance regressions.
- Fresh-install/docs/help/config/lock checks pass for the exact tested wheel;
  rollback archive and matching credential vault have been restored successfully.
- After any safety, scheduling, profile, accounting or engine change, rerun its
  affected cases and both final healthy gates on the final package. Evidence
  from an older binary cannot certify the replacement.

Archive the manifest, results and redacted observations under ignored
`docs/release/results/RUN_ID/`, with a concise tracked decision record and hashes.
An operator may then deploy that qualified package within its recorded scope.
