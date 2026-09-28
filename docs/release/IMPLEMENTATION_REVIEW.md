# Native implementation review against the original cleanup plan

Review date: 2026-09-23 UTC (2026-09-22 local). Scope: **queue and vLLM sleep**,
including the optional queue-only llama.cpp workflow. kv-cached correctness and
compatibility are excluded. Native isolation from optional bootstrap is included.

**Current disposition: the reproducible code defects below have fixes and regression
coverage. The six standalone reproductions now report no reproduced defect. Physical
release qualification remains pending; the fixes are not evidence that hardware or
operator cases passed.** The original review is retained below as a record of what
failed and why the earlier completion claim was too broad.

## Fix disposition

| Finding | Change | Regression / remaining physical evidence |
| --- | --- | --- |
| AUD-01 | Availability checked before local path inspection; listings detect local drift | Six-defect regression; REG-05 remains a manual physical case |
| AUD-02 | Smallest fitting measured TP shape selected; dead branch removed | Queue and 1/2/4/8 topology tests; QUE-07/08 physical |
| AUD-03 | One persistent supervisor cache-budget lock | Concurrent regression; SLP-13 physical pressure |
| AUD-04 | Complete finite vectors, host/timing evidence and shape checks | Invalid/NaN/infinite/bool/missing evidence regressions; PRO-13 operator |
| AUD-05 | Executed engine identity, wrapper version output and distribution manifest bound; pre/post-probe stability | External wrapper update regression; CFG-10/PRO-15 physical |
| AUD-06 | Clone persists exact-launch validation job | Clone regression; PRO-08 with real probes |
| AUD-07 | Atomic selected-profile job snapshot; shared typed CLI/TUI validation; llama.cpp restriction removed | Persisted override/concurrent-submit tests; PRO-06/LCP-05 physical |
| AUD-08 | Forward structured-output requests to engine, matching tool/reasoning compatibility ownership | Request contract coverage; REQ-07 actual supported feature fixture |
| AUD-09 | Added real-GPU driver, exact request/ledger/balance reconciliation, TTFT, controlled cycles, independent telemetry and manual evidence ledger | Runner oracle tests; complete kit must still be executed |
| AUD-10 | Removed dead policy/client facade, added queue-specific scaling settings; documented shared mechanism boundaries | [Architecture decisions](../ARCHITECTURE.md); cross-mode/import and surface checks |
| AUD-11 | Archive requires config, owner lock, and copied-vault decryption/hash verification | Archive mismatch/owner tests; OPS-11/12 restore rehearsal |
| AUD-12 | Acquire database/GPU locks before recovery; release after teardown | Owner-lock regression; CFG-13 separate live test instances |

Run the [physical kit](MANUAL_GPU_TEST.md) after freezing/reinstalling the final wheel.
The [original acceptance design](FINAL_ACCEPTANCE.md) and case catalog remain gates.

## Review boundary and evidence

Compared the pre-cleanup revision `33af608` with `6e44d2d` plus the working tree.
The reviewed source/test/qualify diff SHA-256 was
`74f9301d5f712df46217af9fe034f67b5d58a625e782faa521f521572c45a6f2`.
This digest identifies the reviewed diff, not a deployable package. The last full
test result was 312 passing tests; this review adds independent reproductions and
does not interpret that historical count as end-to-end acceptance.

Run the six isolated reproductions with:

```sh
uv run python scripts/audit_native_release.py
```

The script uses synthetic profiles and a temporary database, starts no service,
and uses no GPU. Exit 1 means a reviewed defect remains reproducible. It must not
be used to manufacture validation evidence. All six checks reproduced on the
originally reviewed tree; all six now report `confirmed=false` after fixes. A future fix needs a normal regression test as well as clearing
the corresponding real-user case below.

## Original confirmed defects

| ID / severity | Trigger and observed problem | Source | Required correction and acceptance case |
| --- | --- | --- | --- |
| AUD-01 / P1 | A staff/admin request arrives just after local registration, while `artifact_path` is still null. Artifact checking constructs `Path(None)` before checking catalog availability and raises `TypeError`, producing an internal error. | [routes_inference.py](../../src/llm_rio/api/routes_inference.py), lines 69–82 | Check availability/path state before artifact inspection; return a stable unavailable response without reserving quota. REG-05. |
| AUD-02 / P1 | TP1 is measured only on a busy GPU, while a measured TP2 set is completely free. Queue placement first filters to the globally smallest shape and returns no placement. Valid TP fallback is lost; requests can wait or cause unnecessary preemption. | [queue planner](../../src/llm_rio/modes/queue/planner.py), lines 142–156 | Select the smallest **fitting** measured shape while preserving oldest-backlog reservations. QUE-07 and QUE-08. |
| AUD-03 / P1 | Two native sleep cache-budget checks overlap. The lifecycle reads a lock from itself but stores the new lock on the supervisor, so each invocation obtains a different lock. The reproduction observed two simultaneous supposedly serialized checks. | [sleep lifecycle](../../src/llm_rio/modes/vllm_sleep/lifecycle.py), lines 99–107 | Use one persistent owner/lock and test concurrent scheduler/offload/status pressure decisions. SLP-13. |
| AUD-04 / P1 | Mode eligibility accepts a sleep profile with missing host-cache/offload/activation evidence. It also accepts NaN memory vectors because only length and negativity are checked. Advanced trust relies on this eligibility decision. | [profiles.py](../../src/llm_rio/profiles.py), lines 122–157; [profile trust](../../src/llm_rio/services/profile_trust.py) | Define complete, finite, internally consistent evidence for each mode; reject missing/invalid values before routing or trust. PRO-13. |
| AUD-05 / P1 | An external engine changes version while its wrapper executable stays unchanged. The binding hashes the wrapper and reads vLLM distribution metadata from the router interpreter. The reproduction changed external version output without changing the binding. | [engine identity](../../src/llm_rio/engines/identity.py), lines 26–40; [validator](../../src/llm_rio/validation.py), version capture | Bind the actual executed engine environment/version, with deterministic failure when identity cannot be established. CFG-10 and PRO-15. |
| AUD-06 / P1 | Clone with changed context/YaRN creates invalidated profiles but no `model_jobs` row. CLI/TUI Validate locate an existing job, so the clone has no supported way to obtain measurements. Trust correctly cannot repair invalidated evidence. | [clone persistence](../../src/llm_rio/profiles.py), `clone_model`; [TUI job lookup](../../src/llm_rio/ui/models.py), `_model_job_id` | Persist a validation job or provide a model/profile validation service that creates one, retaining the exact cloned launch. PRO-08. |

## Original additional source-confirmed gaps

These are code-path findings; they have not been represented as successful live
hardware reproductions.

| ID / severity | Finding | Original intention and closure criterion |
| --- | --- | --- |
| AUD-07 / P1 | Selected-profile revalidation is not a complete shared workflow. [TUI models](../../src/llm_rio/ui/models.py) explicitly refuses non-vLLM profiles. [CLI validate](../../src/llm_rio/commands/models.py) retries the old job and only optionally changes context. [profile edit](../../src/llm_rio/api/routes_admin.py) can change engine/GGUF/launch fields without updating the job used by [registration](../../src/llm_rio/registration.py). | Validate/Revalidate must always probe the selected effective engine/artifact/launch. PRO-06 and LCP-05 must pass independently through all advertised surfaces. |
| AUD-08 / P2 | [request validation](../../src/llm_rio/api/inference_validation.py) rejects `response_format` unless `structured_output` was validated, but [profile publication](../../src/llm_rio/validation.py) never publishes that capability. Supported engines therefore have no normal qualification path for this exposed request field. | Probe and expose supported structured output, or explicitly remove the unavailable promise. REQ-07 uses an engine-supported fixture and may not be silently skipped. |
| AUD-09 / P1 release gate | [qualify.py](../../scripts/qualify.py) is a limited traffic collector. It repeats one prompt, polls state changes, compares a multiset of model/completion-token pairs rather than request identities, and does not reconcile prompt tokens, ledger debits/refunds or balances. It omits most operator/fault workflows and per-request TTFT. | Implement the final runner/oracles in [the acceptance design](FINAL_ACCEPTANCE.md). Existing smoke or collector output cannot certify the final release. |
| AUD-10 / P2 architecture | Mode extraction is partial. `QueueSettings` is empty; common settings retain policy timers; common `WorkerPlacement` owns sleep state; `SelectedMode.eligibility` delegates to boolean-based common policy; common runtime owns preload/cache orchestration. EngineAdapter covers launch only, and `AdminClient` is an unused class facade over `Any`-based functions. Queue preemption also contains unreachable legacy code after an unconditional return. | Complete or explicitly revise each boundary with a reason. Preserve shared teardown/accounting; place mode policy/state with each mode; replace dead code and misleading typed-contract claims. Add architectural tests beyond direct cross-mode imports. |
| AUD-11 / P2 backup gate | [archive.py](../../src/llm_rio/operations/archive.py) verifies SQLite integrity and file hashes, but accepts absent vault/config, does not verify credential decryption with the copied vault, and trusts a `--server-stopped` assertion. | A rollback archive is complete only after ownership check, matching-vault authentication and restore rehearsal. OPS-11/12. Existing source-preservation evidence remains valid; complete rollback readiness is unproven. |
| AUD-12 / P1 ownership risk | [startup](../../src/llm_rio/api/app.py) opens the database and reconciles recorded workers without first obtaining a database/service or managed-GPU owner lock. A second owner can attempt reconciliation against the first owner's live workers. | Refuse concurrent owners before recovery or allocation. CFG-13 must use dedicated test instances; no destructive test against the lab service. This risk was reviewed, not exercised against live owners. |

## Assessment at the original review (historical)

| Plan item | Assessment | Evidence or remaining work |
| --- | --- | --- |
| Explicit canonical modes; no beta configuration compatibility | Largely implemented | Resolver fixes, strict nested settings and examples are reasonable. Inactive settings tests exist. Some beta API aliases remain; decide/remove them deliberately. |
| Independent mode planning/residency/validation/settings/state | Partial | Separate native planners/lifecycles are useful. Validation policy, state and startup policy remain partly common; AUD-10. |
| One registry; process, GPU and port ownership | Partial | Shared port allocator and verified process-group teardown are the right mechanisms. Sleep serialization and multiple service owners remain gaps; AUD-03/12. |
| Exact launch-bound measured profiles and restricted trust | Partial | Hashing and invalidation are useful. External engine identity and measurement completeness do not satisfy the stated guarantees; AUD-04/05. |
| Persisted registration/revalidation/retry for all sources/engines | Partial | HF/local registration is connected; clone and profile-specific revalidation break the end-to-end workflow; AUD-01/06/07. |
| Queue demand, fairness, TP fallback and replicas | Partial | Native unit coverage exists; queue TP fallback defect and dead policy branch require resolution; AUD-02/10. |
| Native sleep/wake and live GPU/RAM/swap limits | Partial | Probes and shared cleanup exist; small physical smoke succeeded. Serialized enforcement, pressure/failure tests and sustained physical evidence remain. |
| Shared admin client, lean interfaces and reliable TUI | Partial | Private CLI imports removed and workflow controllers extracted. Client is not a fully typed service API; role/capability parity and every error/duplicate-submit path remain to be proven. |
| Fresh release schema; original lab preserved | Implemented mechanism, acceptance partial | Beta schema refusal, domain repositories and consistent source-preserving backup are reasonable. Multi-owner startup and rollback credential verification remain open. |
| Tracked docs, reproducible native install, CI | Implemented foundation | Lockfile, fresh wheel import and docs/help checks exist. Documentation must track the partial implementation and open findings. |
| Final user/operator and hardware qualification | Not complete | Prior two-request-per-mode smoke is narrow. The final 172-case design is new; its cases have not been executed. |

## Reasonable decisions to retain

- Independent native planners/lifecycles, even with some policy duplication.
- One process-group teardown implementation and port allocator; fail closed when
  GPU/process release cannot be established.
- One SQLite transaction boundary around quota reservation/settlement despite
  repository splitting.
- An immutable artifact identity, administrative enablement separate from
  measurements, and a single audited trust action.
- Engine-driven compatibility rather than a product model/hardware allowlist.
- A fresh release schema and unchanged lab databases/model artifacts.

## Original test-confidence assessment

Historical tests preserve useful accounting, cancellation and teardown contracts,
but [release fixtures](../../tests/release_fixtures.py) translate old switches and
automatically recompute launch bindings on replacement. That is convenient for
planner unit tests and can hide stale-evidence regressions if used for acceptance.
Final acceptance must create state through public workflows and real probes;
synthetic evidence belongs only in explicitly labeled fault/contract tests.

Close AUD-01–07 and ownership/evidence defects first, settle AUD-08/10's public
contract decisions, complete backup checks, and implement the final acceptance
runner. Then freeze an exact package and run the full native gate. Findings are
closed only by a regression test plus their linked user/operator acceptance case.
This review intentionally does not silently change product behavior while writing
the test specification.
