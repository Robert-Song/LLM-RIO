# LLM service acceptance tests

This is the test contract for the intended service behavior. It deliberately tests externally visible results and required observability, not a particular scheduler implementation. A passing test must not depend on sleeps chosen to make a race *probably* disappear.

The tests below are the fast acceptance suite. The long fair-share rotation scenario is specified separately and is **not** part of this suite.

After you confirm all tests passed, I will also run all tests with watching nvidia-smi and actual terminal output, too.
During this development phase, the machine we will test have only 2 GPUs and 3 models. Tests below are written based on this testing environment, but this should not be hard-coded in the app, and this app is meant to be scalable (not in a way of security, but in terms of performance)

## Improvement conventions

This app is not released yet, there are no users, so backward compatability is not needed.
If debug or changes are needed, take the best fix. Do not consider preserving legacy behavior.
Database inside state/ can be wiped, too (not a production database).

No GUI is needed.
The machine is already well-secured so it is guranteed that only person who will access this directory is admin.
Reaching CLI or calling ./llmctl is guranteed to be an admin.
CLI should be interactive.

## Test conventions

Configure these values once for the target machine. Do not hard-code GPU numbers: the service discovers local GPUs and uses the validated placement profiles.

```bash
export BASE_URL=http://0.0.0.0:8002
export ADMIN_KEY='rio_43e22ad6dc6d_0ZKev_Uydo3GsuC2hzhzjA3UQk54FsdvTBA7D-cfois'
export TEAM_A_KEY='rio_71a98c79a30f_55URMzyvMQ5gM8oYYANnNJ8Nm_L-HeRFc_CyGzFyJm4'
export TEAM_B_KEY='rio_3075269f169a__vYXUH3A86ckkjA7qY3WlNj-rEW6gre8-EFHSRUQaaM'
export QWEN_MODEL='qwen3.6-27b-nvfp4'
export GEMMA_MODEL='gemma-4-31b-it-nvfp4'
export LAGUNA_MODEL='laguna-s-2.1-nvfp4'
```

Model names above are aliases. The actual catalog entries must point to immutable, validated model artifacts and profiles. In the target configuration:

- Qwen and Gemma each have a single-GPU profile.
- Laguna has a multi-GPU tensor-parallel profile.
- Qwen replicas are separate `tensor_parallel_size=1` vLLM workers, each with its own local HTTP endpoint. They are **not** one tensor-parallel Qwen worker.

Before every test:

1. Ensure service mode is `ACTIVE` and no maintenance drain is in progress.
2. Record the starting balance.
3. Capture one scheduler-state snapshot and one `/v1/models` response.
4. Clear previous test request logs/metrics by using a unique `X-Test-Run-ID` header.

The gateway must expose an admin-only read-only status endpoint (the exact path may differ) with, at minimum:

```json
{
  "mode": "ACTIVE",
  "workers": [
    {
      "worker_id": "...",
      "model": "qwen3.6-27b-nvfp4",
      "state": "READY|DRAINING|LOADING|STOPPED|FAILED",
      "gpu_uuids": ["..."],
      "tensor_parallel_size": 1,
      "active_requests": 0,
      "queued_requests": 0,
      "accepted_requests": 0
    }
  ]
}
```

It also needs per-request structured logs containing the test-run ID, logical model, selected worker ID, admission time, inference-worker acceptance time, completion status, and token usage. These fields make the tests deterministic and make a failed test diagnosable.

For every successful completion request, assert all of the following:

1. HTTP status is `200`.
2. Response model is the requested logical model.
3. At least one non-empty output choice is returned.
4. The request is recorded as accepted and completed exactly once.
5. The stream terminates cleanly when `stream=true`; no chunks may come from two worker IDs.

Use short, deterministic prompts such as `Reply with exactly: OK` and a bounded `max_tokens` (for example 16) unless a test explicitly exercises quota reservation.

## 0. Catalog and model loading health

**Purpose:** prove that every registered, usable model can be discovered, loaded by demand, and used for one successful inference. This is the baseline smoke test for artifact correctness, routing, and the model supervisor.

Command shape:

```bash
uv run scripts/load_test.py --preset 0
```

### Steps

1. Send `GET /v1/models` using a key that is allowed to use Qwen, Gemma, and Laguna.
2. Verify that each expected alias appears exactly once and is marked callable/available. Record the catalog revision/profile identifier returned for each alias.
3. For each model, one at a time, send one non-streaming completion request with the same bounded prompt, in order of Qwen, Gemma, and Laguna.
4. Wait using the request's terminal response or an explicit status event; do not sleep for an assumed load time.
5. After each request, inspect the request log and scheduler status.
6. Between models, let the scheduler make its normal decision.

### Required results

- Every listed model completes one request successfully.
- The worker selected for Qwen/Gemma reports a validated single-GPU placement.
- The worker selected for Laguna reports the validated multi-GPU set and matching tensor-parallel size; all GPUs in that set belong to the same worker.
- No worker is routed requests before its state is `READY`.
- The gateway never returns a raw engine connection error, traceback, or a model-internal port to the client.
- A model that is registered but currently unloaded is loaded automatically. Staff catalog approval must not require a separate manual deploy action.

### Expected full behavior
1) When Qwen is first requested, the application will grab all available GPUs (2 in this case). Recognizing that Qwen can fit on a single GPU, it will create two different subprocesses and call vLLM to load Qwen on both GPUs.
2) Next, it will load-balance the request (we only have one request in this case), route it to one of the GPUs, and complete the request.
3) Once the first request is done, our test will send a request for Gemma. The system will check the request queue, confirm that no remaining requests for Qwen exist, and unload Qwen from both GPUs. Seeing that Gemma can fit on a single GPU, it will create two subprocesses (one on each GPU) and call vLLM to load Gemma on both GPUs. It will then handle the request and return the result.
4) Once the second request is done, our test will send a request for Laguna. The system will check the request queue, see that no remaining requests for Gemma exist, and unload Gemma from both GPUs. It will then notice that Laguna requires both GPUs, so it will create one instance using tp=2 (tensor parallelism) to load Laguna with vLLM across both GPUs and handle the request.


## 1. Qwen heavy parallel replica test

**Purpose:** verify that a sustained single-model load causes the service to use multiple independent Qwen replicas when capacity and the validated Qwen profile allow it, and that the gateway balances new requests across ready replicas.

Command shape:

```bash
uv run scripts/load_test.py --preset 1 --requests-per-worker 10
```

### Exact workload

- Start **128 client workers behind one synchronization barrier**.
- Each worker sends 10 bounded Qwen completion requests sequentially after the barrier.
- Therefore there are at most 128 simultaneous HTTP requests and exactly 1,280 attempted requests, unless a failure causes the runner to stop early.
- Every request includes the same `X-Test-Run-ID` and `X-Client-Worker` identifier.

### Steps

1. Ensure no non-Qwen requests are queued before releasing the barrier.
2. Release all 128 workers at once.
3. Poll scheduler status while the backlog exists and retain each snapshot.
4. Let every worker finish; set a bounded test deadline appropriate for the machine's validated Qwen profile.
5. Group gateway logs by selected Qwen worker ID.

### Required results

- All 1,280 requests succeed. Engine crashes, connection resets, admission rejections, and indefinite waits are failures.
- Ttwo ready replicas each accept a non-zero number of requests. A single busy replica plus an idle ready replica fails the routing check.
- Each worker uses a disjoint GPU set. For the normal one-GPU Qwen profile, each replica owns exactly one distinct GPU.
- A streaming request is pinned to one replica from first byte through terminal chunk.
- Aggregate throughput, p95 time-to-first-token, p95 completion latency, error count, and requests per replica are reported. This test records performance; pass/fail performance thresholds belong in the machine-specific benchmark profile, not in a universal constant.

**Important interpretation:** this checks higher *aggregate* throughput from two or more independent replicas. It does not require one individual completion to become twice as fast.

### Expected full behavior
1) When Qwen is first requested, the application will grab all available GPUs (2 in this case). Recognizing that Qwen can fit on a single GPU, it will create two different subprocesses and call vLLM to load Qwen on both GPUs.
2) Next, when 128 requests enter the queue, the application will load-balance and route them across both GPUs to complete the requests. Ideally, given two identical GPUs and uniform requests, it should distribute the load evenly (50/50). Alternatively, dynamic real-time request dispatching as new requests arrive is also acceptable.


## 2. Dual-model simultaneous load: Qwen plus Gemma

**Purpose:** prove that the scheduler can preserve useful coexistence: when demand exists for two models that each fit in one GPU, it can run them as separate workers rather than treating all GPU capacity as one indivisible pool.

Command shape:

```bash
uv run scripts/load_test.py --preset 2 --requests-per-worker 10
```

### Exact workload

- Create 16 Qwen clients and 16 Gemma clients behind the same synchronization barrier.
- Each client sends a bounded request immediately after barrier release. If a repeat count is supported, use the same repeat count for both model groups.
- The initial wave is exactly 32 concurrent streams: 16 Qwen and 16 Gemma.

### Steps

1. Confirm both test keys are allowed to use their requested model.
2. Release 16 Qwen clients first, then wait 2 seconds, and release Gemma slients.
3. Capture worker states until each request has been accepted by a ready inference worker.
4. Collect per-request worker/model assignment and final timing/token metrics.

### Required results

- Every Qwen request reaches a Qwen worker; every Gemma request reaches a Gemma worker. No cross-model response or fallback occurs.
- When the host has at least two compatible free GPUs, the active worker placements for Qwen and Gemma are disjoint and are both `READY` before their requests are accepted by inference.
- Qwen traffic must not starve Gemma traffic merely because Qwen's initial queue is larger, and vice versa. Each model has at least one admitted request before the other model completes its entire initial wave.
- Report Qwen throughput/latency, Gemma throughput/latency, and combined throughput. Do not calculate combined tokens per second by adding incompatible time windows; use the common interval from first accepted request to last completion.
- If the machine's validated profiles prove the two models cannot coexist, mark this test `SKIPPED (profile incompatibility)` before sending workload. It must not be silently converted into a serial test.

### Expected full behavior
1) When Qwen is first requested, the application will grab all available GPUs (2 in this case). Recognizing that the Qwen model can fit on a single GPU, it will create two different subprocesses and call vLLM to load Qwen on both GPUs.

2) When a Gemma request arrives, the application notices that Gemma requires at least one GPU (as it fits on a single GPU). However, all GPUs are currently in use. The system inspects the GPUs and sees that Qwen is loaded on both GPU 0 and GPU 1. It checks the queue and sees that requests for Qwen are still coming in, so it cannot completely unload Qwen. However, it detects that 2 GPUs are currently allocated to Qwen, whereas Qwen's minimum requirement is only 1 GPU (TP=1). Because 2 (allocated Qwen GPUs) > 1 (minimum GPUs required for Qwen), we have (2 - 1) unloadable GPU from Qwen. Then, we have (1) (maximum (sum) of all GPUs that can be freed) >= 1 (minimum GPUs required for Gemma), the system decides to preemptively free up one GPU. It reroutes all new Qwen requests to the other Qwen-loaded GPUs, waits for the ongoing computation on the target GPU to complete, unloads Qwen from that GPU, loads Gemma onto it, and begins accepting Gemma requests.

3) One GPU will continue serving Qwen, while the other GPU serves Gemma.
Because Qwen is a smaller model and started earlier, it will ideally finish its requests before Gemma. Once Qwen has no queued or in-flight work, the waiting Gemma demand immediately reclaims its idle GPU; `wait_duration` applies only to automatic unloading when no other model needs the GPU.

Finally, the application checks the request queue, sees that Gemma requests are still incoming while one GPU is now idle, and loads Gemma onto that second GPU as well.

## 3. Preemption capacity check: High-demand Qwen deferred by multi-GPU Laguna requirement

Purpose: Prove that the scheduler enforces strict capacity requirements before attempting preemption. When a incoming model requires multiple GPUs (Laguna, TP=2) while another model (Qwen) is active, the scheduler calculates maximum reclaimable capacity. If the maximum reclaimable GPUs cannot satisfy the incoming model's minimum requirement, and the active model cannot be fully evicted (due to incoming requests and active `fair_share_seconds`), the scheduler must queue the new model without triggering partial, useless GPU unloads.

Command shape:

```bash
uv run scripts/load_test.py --preset 3 --requests-per-worker 10

```

### Exact workload

* Saturate both GPUs with a high-volume Qwen request stream (e.g., 32 concurrent requests).
* Issue a Laguna request (requiring TP=2 across 2 GPUs) shortly after Qwen begins processing, while Qwen is still within its `fair_share_seconds` threshold.
* Allow Qwen requests to complete naturally, followed by Laguna execution.

### Steps

1. Release Qwen clients to fill the request queue and occupy both GPUs.
2. Release Laguna client while Qwen requests are still actively streaming.
3. Monitor GPU states and confirm Qwen is NOT preempted or downscaled.
4. Verify Laguna remains queued until 2 GPUs become free simultaneously.
5. Record completion metrics for both model waves.

### Required results

* Qwen continues processing on both GPUs without interruption or unnecessary single-GPU unloads.
* Laguna request is safely held in the queue without timing out or causing worker crashes.
* Once Qwen completes, Laguna is successfully allocated across both GPUs using `tp=2`.

---

### Expected full behavior

1) When Qwen is first requested with high volume, the application grabs all available GPUs (2 in this case). Recognizing that Qwen can fit on a single GPU (TP=1), it creates two subprocesses and loads Qwen on both GPU 0 and GPU 1 to handle the load.

2) When a Laguna request arrives, the system notices that Laguna requires 2 GPUs (TP=2). It inspects the GPUs and sees that 0 GPUs are free because both are currently allocated to Qwen.

3) The system checks if it can free up capacity for Laguna. It sees that Qwen requests are still incoming and Qwen's execution time is less than `fair_share_seconds`, meaning Qwen cannot be completely evicted (unloaded to 0 GPUs). It then calculates the maximum reclaimable capacity from partial release: `2 (allocated Qwen GPUs) - 1 (minimum GPUs required for Qwen) = 1 (maximum GPUs that can be freed)`. It checks whether this freed capacity meets Laguna's requirement: `2 (minimum GPUs required for Laguna) <= 1 (maximum GPUs that can be freed)` is FALSE (`2 > 1`). Because freeing 1 GPU is insufficient to run Laguna, the application decides NOT to unload Qwen from any GPU and holds the Laguna request in the queue.

4) Both GPUs continue serving Qwen uninterrupted until all Qwen requests in the queue are completed.

5) Once Qwen finishes all requests and no new Qwen requests are incoming, the waiting Laguna demand immediately reclaims both idle Qwen GPUs. The application verifies that 2 GPUs can be freed (`2 >= 2`), unloads both Qwen workers, loads Laguna across both GPUs using `tp=2`, and processes the request. `wait_duration` applies only when no queued model needs the idle GPUs.

## 4. Maintenance drain and resume

**Purpose:** prove that administrators can promptly and safely free a shared HPC machine without a race that loads another model after draining has begun.

### Setup

- Start one deliberately slow but bounded streaming completion that stays in flight long enough to observe draining. A test-only prompt/limit is acceptable; do not use an unbounded generation.
- Record its request ID and selected worker.

### Steps

1. While the request is active, submit the single atomic administrator command:

   ```text
   POST /admin/maintenance
   {"mode": "drain"}
   ```

2. Immediately send a new ordinary inference request with a permitted user key.
3. Observe the active request until it finishes normally.
4. Poll admin status until it reports `MAINTENANCE_READY` (or the documented equivalent).
5. Verify every worker is `STOPPED` and GPU-memory use has returned to the configured idle/headroom range.
6. Submit the atomic resume command.
7. Send one permitted inference request and wait for completion.

### Required results

- The maintenance command changes service mode durably to `DRAINING` before its response is returned.
- The new inference request is rejected immediately with `503` and `Retry-After`; it is not queued and cannot trigger a model load.
- The previously accepted request completes successfully. A drain must wait for all in-flight sequences on its worker, not merely one scheduler iteration.
- Draining workers receive no new requests, then stop only after their active request count is zero.
- After `MAINTENANCE_READY`, no model worker remains alive and no automatic scheduler action reloads a model.
- After resume, mode is `ACTIVE`; the final request loads/routes its allowed model automatically and succeeds.

## Separate endurance test: fair-share safety bound

Do **not** shorten this test to seconds and call it equivalent. It verifies behavior only visible when two incompatible, continuously backlogged workloads contend for the same GPU set over the configured fair-share interval (for example, 120 minutes).

Pass condition: a waiting model is eventually selected before the configured starvation bound expires, but model switching is not forced when there is no incompatible queued demand. For every switch, record queue age, policy reason, drain duration, load/warm-up duration, ready time, and productive token time. Evaluate switch cost and aggregate throughput manually after the run.

## Runner requirements

`scripts/load_test.py` should emit JSONL for every request plus one summary JSON file. The summary must include:

```json
{
  "preset": 1,
  "run_id": "uuid",
  "attempted": 1280,
  "succeeded": 1280,
  "failed": 0,
  "by_model": {},
  "by_worker": {},
  "combined_tokens_per_second": 0,
  "p95_ttft_ms": 0,
  "p95_latency_ms": 0,
  "failures": []
}
```

The runner must fail its process exit code if any required assertion fails. It must not report success merely because it received some successful responses.
