# Native acceptance case catalog

Status: **designed, not executed**. The [execution plan](FINAL_ACCEPTANCE.md) defines fixtures, expansion rules, oracles and release gates. Each semicolon in a step is sequential. Both native modes run every shared case; queue-only and sleep-only cases are labeled. Actual HTTP/CLI/TUI bindings and configuration coverage are in [the machine-readable catalog](native-acceptance.json).

## Installation and configuration

Modes: queue, vllm-sleep. Lane: install. Prerequisite: Clean checkout, empty test root, operator-owned ports and devices.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| CFG-01 · operator | Install locked router/dev environment from a clean checkout; build wheel and install it into a second fresh environment; invoke every CLI help command | Install and help succeed without experimental dependencies or bootstrap activation; record revision, wheel hash and lock hash |
| CFG-02 · operator | Start once from CLI mode, once from environment mode, and once from TOML mode | Selected mode and capability response match each source; no implicit mode |
| CFG-03 · operator | Omit mode; use an unknown mode; supply retired prism settings and unknown nested settings; submit through bare ./llmctl → Diagnostics → Start service, correct and resubmit | Actionable failure before engine spawn without credential values or traceback; start form stays open and retains values; corrected configuration starts; database and artifacts unchanged on error |
| CFG-04 · operator | Use conflicting CLI/environment/TOML values for config selector, API port and mode; repeat from a different working directory; keep beta config.toml present and select a release file with .env; start via bare ./llmctl and restart | One documented precedence applies to CLI/TUI/server, Doctor and start form; selected release mode/database respond to health and authenticated capabilities; admin credential survives restart |
| CFG-05 · operator | Keep valid queue and sleep sections together; select each mode through TOML, CLI, environment and TUI; separately enable llama.cpp in sleep mode or provide unknown settings, invalid ranges and reversed port range | Only the selected mode's settings affect execution; queue does not inherit sleep preloads or memory policy; invalid settings and incompatible active engine options fail before partial release state |
| CFG-06 · operator | Start against a fresh release path; stop and restart it | Exactly one bootstrap admin and versioned schema; original credential remains usable after restart; no duplicate accounts |
| CFG-07 · operator | Point new code at a COPY of a beta database and vault; attempt startup | Startup refuses before schema or vault mutation; hashes of copied source remain unchanged |
| CFG-08 · operator | Use missing or nonexecutable engine; invalid config path; unwritable model/log/database paths; occupied API port | Clear bounded failure identifying cause; no leftover service or engine process |
| CFG-09 · operator | Select GPUs by UUID and alternate CUDA visibility ordering; specify missing/duplicate UUIDs and unsupported device mapping | Only intended UUIDs can be allocated; invalid selection is explicit; no implicit reuse of unrelated GPUs |
| CFG-10 · operator | Use external vLLM environment; validate a profile; upgrade its engine while keeping launcher script unchanged; restart | Old profile becomes ineligible until revalidation; recorded engine identity comes from the executed engine |
| CFG-11 · operator | Change measurement-affecting engine environment and launch defaults; then change log level only | Measurement-affecting drift blocks old evidence; unrelated logging changes do not fabricate or erase measurements |
| CFG-12 · operator | Exhaust or occupy private worker ports while sending inference and validation requests; free a port and retry | Allocator never collides; failed work is recoverable; uncertain teardown retains ownership |
| CFG-13 · operator | Start a second service against the SAME test database while first serves; also test separate test databases sharing test GPUs | Second owner is refused or explicitly blocked before it can terminate or allocate against first owner; original stream survives |
| CFG-14 · operator | Run doctor with healthy inventory, missing telemetry, absent engine, and no GPUs; inspect JSON and TUI result | Diagnostics identify prerequisites without loading a model; exit/report semantics agree |
| CFG-15 · operator | Start native mode with experimental activation variables inherited from shell | Actual engine launch environment has no active experimental patch/bootstrap; native import isolation verified |
| CFG-16 · operator | Vary validation idle window, scheduler tick and idle/residency timing within documented ranges | Observed scheduling and validation delays follow active settings; accepted settings are not silently ignored |
## Identity, access and quotas

Modes: queue, vllm-sleep. Lane: real-service. Prerequisite: Admin, staff, two user keys, two quota accounts, A and B validated.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| IAM-01 · unauthenticated | Call every protected route with no bearer, invalid bearer and malformed scheme; call health | Protected routes return 401 without mutation; health returns only its documented public status |
| IAM-02 · staff/user-a | Call every route with each role and attempt each supported CLI/TUI action | Role matrix enforced by server; forbidden actions return 403 and UI explains unavailable controls; no credential disclosure |
| IAM-03 · admin | Create admin/staff/user keys, generated and custom credentials; list/show/copy them through supported interfaces | One created identity and account; credentials work only for assigned role; secrets stay out of acceptance reports |
| IAM-04 · admin | Create duplicate nickname, malformed custom token, missing quota account and nonexistent grant; repeat submission concurrently | Deterministic documented 4xx; no partial key/account/grant; duplicate submission has one effect |
| IAM-05 · admin | Rotate a key while a request is active; use old and new secrets for subsequent requests | New secret authenticates and old secret fails; accepted work completes or follows documented cancellation policy without double settlement |
| IAM-06 · admin | Revoke then restore a user key during queued/active work; retry after each change | New requests denied while revoked and accepted after restore; existing-work policy is explicit and accounting remains correct |
| IAM-07 · admin | Delete a key with prior usage and attempt restore/authentication | Credential is unusable; audit and charged usage remain; missing-key actions return 404 |
| IAM-08 · staff | Grant A, revoke A, replace full model grant set with B; exercise user-side and model-side access forms | User list and requests agree with grants after each action; no access granted implicitly |
| IAM-09 · staff | Replace grants with one valid and one nonexistent model; use a missing key; submit empty replacement | Invalid replacement is atomic and changes nothing; empty replacement removes all grants |
| IAM-10 · user-a | List visible models; request hidden, missing, disabled and pending model names | No unauthorized execution; stable 403/404/409 or documented revalidation error; available-model hints respect visibility |
| IAM-11 · admin | Set finite quota and unlimited quota; change quota while requests are reserved | No lost reservations; limit/unlimited state visible consistently in usage and UI |
| IAM-12 · user-a/user-b | Send simultaneous requests sharing one account with total reserve above remaining quota; repeat with independent accounts | Atomic acceptance permits only funded reservations; rejected requests return quota_exceeded 429; account isolation holds |
| IAM-13 · user-a | Spend exactly remaining reservable credit; request just over limit; complete under requested maximum | Boundary behavior is consistent; unused reservation refunded once; observed actual usage reconciles |
| IAM-14 · admin | Reset usage while prior requests have settled and while one remains active; query lifetime and current-period totals | Reset does not delete lifetime audit or corrupt live reservations; subsequent settlement occurs once |
| IAM-15 · admin | Compact settled usage twice while other calls are queued/streaming; replay an old request identifier afterward | Compaction excludes live work, preserves balances/totals and duplicate protection, and is idempotent |
| IAM-16 · user-a | Read own usage from API and supported CLI; attempt to inspect another user through lower-role paths | Only permitted usage is visible; active reserves, period totals and lifetime totals reconcile |
## Registration and validation jobs

Modes: queue, vllm-sleep. Lane: real-service. Prerequisite: Two distinct engine-supported artifacts A/B, pinned HF source H, disposable local copy L.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| REG-01 · staff | Register pinned HF revision H with grant to user-a; review job until native maintenance prerequisite appears; drain; complete validation; resume and infer | Resolved immutable revision persisted; probes really launch; user can infer only after eligible profile publication |
| REG-02 · staff | Register existing local directory L; validate and infer; compare source inode/path/content before and after | Local files reused without copying or deleting; complete manifest and pinned identity recorded |
| REG-03 · staff | Register missing/relative/empty paths, wrong source type, both sources, neither source, invalid revision, missing grants | Actionable 4xx or persisted failed job as appropriate; no worker/reservation leak or partial grants |
| REG-04 · staff | Register a vLLM-supported artifact with nonstandard metadata or weight extension | Inspection is advisory; actual engine probe decides compatibility; no router architecture allowlist |
| REG-05 · user-a/admin | Request a local model immediately after registration and before artifact_path is populated | Documented model-unavailable response instead of 500; no quota reservation or engine start |
| REG-06 · staff | Submit two identical registrations concurrently and repeat a pending validation request | One logical operation or explicit 409 conflict; no duplicate validation worker or partial catalog row |
| REG-07 · staff | Run validation while ACTIVE with production requests; enter maintenance during a long stream | Native validation waits; admitted production finishes; only verified maintenance-ready GPUs are acquired |
| REG-08 · admin | Try resume while validation owns any GPU; inspect status from API/CLI/TUI | 409 validation_in_progress with prerequisites; no serving launch competes with probes |
| REG-09 · staff | Validate with explicit TP, context, sequence count, batch tokens, GPU utilization, dtype and allowed launch arguments | Persisted job and accepted LaunchSpec preserve exact requested effective values or explicitly report a measured retry adjustment |
| REG-10 · staff | Retry a failed job with only context changed; then explicitly clear/change another override | Unchanged saved fields preserved; clear semantics documented; retries always execute probes |
| REG-11 · staff | Force download/auth/disk/engine-start/generation-probe failure one at a time; inspect and retry | Failure stage, actionable diagnostics and redacted log path saved; one retry recovers without duplicate profile publication |
| REG-12 · operator | Interrupt service during download, launch, probe and final publication; restart and review same job | No orphan engine; persisted job reaches recoverable state; profile publication and catalog availability are atomic |
| REG-13 · staff | Queue validation for A and B; attempt duplicate retries while first is running | Jobs serialize shared validation ownership; disjoint probes stay within their reservations; no stale running job |
| REG-14 · staff | Validate all eligible single-GPU placements; make one fail; validate explicit TP across heterogeneous/ineligible sets | Only measured successful UUID sets published; unsupported or unprobed placement cannot serve |
| REG-15 · operator | Introduce external GPU memory pressure and then missing telemetry during validation; remove each condition | Validation waits/fails closed with reason; baseline drift is rejected; it never treats absent telemetry as free memory |
| REG-16 · staff | Validate same artifact twice without changes and once after source revision/content changes | Probes run every time; changed identity invalidates prior serving eligibility; profile row/key references remain consistent |
| REG-17 · operator | Change, remove, add or replace a local file after validation; exercise new request, cold start, preload, sleeping wake and clone | Detected change blocks serving/clone until revalidation; no stale warm worker bypass; manifest reconciles after successful probes |
| REG-18 · operator | Modify a disposable local artifact during hashing and during validation | Unstable artifact cannot publish misleading evidence; retry after stable content succeeds |
| REG-19 · staff | Disable catalog model while READY, queued, sleeping and validating; wait for job completion and attempt inference | Admitted-work policy is explicit; no new work while disabled; concurrent job cannot silently undo a later disable decision |
| REG-20 · staff | Revalidate HF artifact after branch/tag advances while an immutable revision is requested | Original pinned revision remains reproducible; intentional revision change creates new identity and eligibility |
## Profiles, clones and trust

Modes: queue, vllm-sleep. Lane: real-service. Prerequisite: A/B validated, saved evidence copy, admin and limited user.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| PRO-01 · admin | List current active/inactive profiles and saved measurements from another fingerprint | All entries have stable IDs, eligibility reason, effective launch identity and administrative state |
| PRO-02 · admin | Disable profile with queued and active work; re-enable unchanged measured profile | New admission stops for disabled profile, admitted work drains, eligible enable succeeds without fabricating probes |
| PRO-03 · admin | Attempt enable with invalidated, wrong-mode, wrong-engine, changed-artifact and obsolete-format evidence | 409 with stable reason; no worker launch and no trust side effect |
| PRO-04 · admin | Edit each measurement-affecting field separately, with restart enabled and disabled; infer before revalidation | Evidence invalidated; stale workers cannot accept new requests under changed limits; admitted requests obey documented drain policy |
| PRO-05 · admin | Edit to an existing active and then inactive profile configuration | 409 identifies existing profile and recovery action; transaction does not partially edit or duplicate |
| PRO-06 · admin | Edit a profile then Validate/Revalidate it from API, CLI and TUI | Selected profile engine, artifact and complete launch settings reach real probes; replacements reflect the edit, not old job defaults |
| PRO-07 · admin | Clone A with only generation defaults; grant clone separately; infer from original and clone concurrently | Artifact shared without copy; separate queues/grants/accounting; unchanged measured launch may be reused only when eligible |
| PRO-08 · admin | Clone A with larger context or YaRN changes; validate clone; infer to new limit | Clone has a persisted validation workflow; invalidated evidence cannot be enabled/trusted; requested clone launch survives validation |
| PRO-09 · admin | Clone after source profile disable, launch drift, artifact change or missing artifact | Clone refused with actionable prerequisite; no unusable partial catalog entry |
| PRO-10 · admin | Change temperature/top_p/top_k/min_p/penalties/reasoning defaults; send omitted, explicit and null request values | Defaults apply only according to documented omission/null rules; explicit request wins; no launch evidence invalidation for generation-only defaults |
| PRO-11 · admin | Trust eligible saved evidence after changing only permitted machine fingerprint fields; supply reason | Same artifact, GPU UUIDs, mode, engine and launch required; audit records actor, reason, source and affected profile; no probe claimed |
| PRO-12 · admin/staff | Trust with missing/blank/oversized reason, nonadmin actor, missing source, changed GPU, changed engine or launch | 403/404/409/422 as appropriate; no new profile or partial trust audit |
| PRO-13 · admin | Trust evidence missing RAM/sleep/wake fields, wrong vector lengths, negative or nonfinite values, obsolete format or invalidation marker | All incomplete/invalid evidence is rejected; override cannot invent measurements |
| PRO-14 · admin | Revalidate a previously trusted profile and inspect provenance and measurement timestamp | New real evidence replaces override eligibility cleanly; previous audit remains attributable |
| PRO-15 · admin | Change one engine library/version without changing wrapper executable; change relevant environment; restore old values | Version/configuration mismatch blocks serving and trust until real revalidation; no false assurance from wrapper hash alone |
| PRO-16 · admin | Race edit/enable/trust/revalidate for the same profile with concurrent requests | One coherent saved state and eligibility decision; no invalid profile routed or orphan duplicate row |
## Inference protocol and limits

Modes: queue, vllm-sleep. Lane: real-service. Prerequisite: A/B validated; capability-selected tool/reasoning/vision/structured-output fixtures when supported.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| REQ-01 · user-a | Send first nonstreaming chat to a cold model then repeat warm | 200 OpenAI-shaped body; correct model/request identity, choices and authoritative usage; one reservation and settlement each |
| REQ-02 · user-a | Stream first cold and subsequent warm requests using a real SSE client | Incremental valid UTF-8 events, final usage and exactly one DONE; no missing/truncated chunks; first-token and total latency recorded |
| REQ-03 · user-a | Run multi-turn system/user/assistant conversation with Unicode and multiline content | Messages reach engine intact; valid response and usage; no cross-user conversation or output mixing |
| REQ-04 · user-a | Send every exposed sampling field and passthrough engine extension with deterministic seed where supported | Fields forwarded once and overrides preserved; engine-rejected options produce bounded meaningful error |
| REQ-05 · user-a | Send tool definitions, auto/none/named tool choice, parallel calls and tool result continuation; repeat streamed | Tool IDs, names, arguments and deltas preserved; router does not invent or strip engine tool output |
| REQ-06 · user-a | Send each supported reasoning effort and inspect reasoning fields streamed/nonstreamed | Engine reasoning and content channels preserved; unsupported value errors are explicit |
| REQ-07 · user-a | Send json_object and json_schema response_format on a model/engine that supports them | Validated supported formats remain usable; router does not advertise an action that registration can never qualify |
| REQ-08 · user-a | Send supported image/audio/video or other multimodal message parts with fixture URLs/bytes | Payload forwarded intact; engine support decides result; accounting/limits policy is explicit and no payload stored in metrics |
| REQ-09 · user-a | Exercise prompt/output/context boundaries at below/equal/above verified limit with exact engine-tokenized fixtures | In-range calls succeed; out-of-range errors are clear; no unexplained engine OOM or reserve/accounting mismatch |
| REQ-10 · user-a | Omit output limit; set max_tokens; set max_completion_tokens; set both equal and conflicting | One documented effective limit reaches engine; server cap applies; alias conflicts handled deterministically |
| REQ-11 · user-a | Send n=1, configured maximum and maximum+1 with streaming/nonstreaming | All choices indexed and forwarded; output reserve covers all choices; n limit enforced with no engine work on rejection |
| REQ-12 · user-a | Send empty messages, wrong JSON types, malformed JSON, unknown model, oversized IDs and invalid numeric values | Stable documented 4xx; response stays OpenAI error-shaped; no 500 or resource/quota leak |
| REQ-13 · user-a/user-b | Submit identical X-Request-ID and Idempotency-Key concurrently, after completion and after compaction | Duplicate operation cannot charge or execute twice; account scoping and cross-key behavior documented and tested |
| REQ-14 · user-a | Disconnect while queued, while cold-starting, before stream headers, midstream and after final event | Cancellation reaches ownership cleanup; no stuck lease/reservation; partial usage charged according to documented evidence policy |
| REQ-15 · user-a | Use slow consumer and abruptly reset TCP connection while other tenants send traffic | Bounded buffering and cleanup; healthy users keep progressing; no unbounded memory growth |
| REQ-16 · user-a | Exercise configured nonstream timeout, stream-idle timeout and drain watchdog using controlled stalls | Terminal error/stream event within bound; one settlement and release; unsafe worker is not reused |
| REQ-17 · user-a | Trigger genuine engine input error and engine 4xx/5xx; submit healthy request afterward | Safe error forwarded; failed request settlement correct; healthy traffic recovers without service restart |
| REQ-18 · observer | Inject authoritative zero usage, missing usage and malformed/negative usage from a controlled engine protocol fixture | Zero is never replaced by estimate; missing-usage policy explicit; malformed response fails once without charging a fabricated maximum |
| REQ-19 · observer | Inject split Unicode, multiline SSE, missing DONE, duplicate terminal frame, malformed JSON and partial choices | SSE framing remains correct or returns one terminal protocol error; finalization idempotent; no byte corruption |
| REQ-20 · user-a/user-b | Send interleaved streaming/nonstreaming calls to A and B from both tenants | Outputs, request IDs, model names and token charges remain attached to the right request/account |
| REQ-21 · observer | Compare every successful response usage to its request-ID ledger and account delta under actual-usage and requested-maximum charging | Prompt/completion/total usage and configured charge reconcile per request; totals, refunds and compaction all agree |
| REQ-22 · user-a | Request a model whose artifact/profile is invalid while v1/models is queried in parallel | Model list callable state agrees with actual routing eligibility; no misleading callable flag |
## Queue scheduling

Modes: queue. Lane: real-service. Prerequisite: A/B incompatible simultaneous residency, independently validated TP and replica profiles.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| QUE-01 · user-a | Start service idle; register/validate and resume with no user traffic; then send one cold request | No queue worker starts before real demand; launches and ready state explain request admission |
| QUE-02 · user-a | Finish last request and wait across idle timeout and minimum residency boundaries | Worker drains only after admitted work completes and policy timing permits; GPU context and port released before reuse |
| QUE-03 · user-a/user-b | Run a long A stream; queue older B backlog then newer C backlog against the same GPU | Admitted A survives drain; oldest incompatible backlog obtains freed capacity first; C cannot steal reserved placement |
| QUE-04 · user-a/user-b | Keep older resident A backlog and submit newer B; later reverse oldest-backlog ordering | Documented oldest-backlog policy is observed without permanent starvation; fair-share setting has a defined tested effect or is removed |
| QUE-05 · user-a/user-b | Send high-volume short requests and fewer long requests sharing a model across tenants | DRR uses reserved token cost, preserves within-tenant order and bounded service progress; same-account keys cannot multiply fair share |
| QUE-06 · user-a | Send a burst exceeding per-model and per-tenant queue limits while workers are saturated | Only queued work consumes queue capacity; 429 queue_full refunds reservation; accepted work continues |
| QUE-07 · user-a | Hold one TP1-only placement busy while a validated larger TP set is completely free; request the waiting model | Planner uses a fitting validated TP alternative instead of stalling on the globally smallest shape |
| QUE-08 · user-a | Request a TP model while only part of its GPU set is free and younger single-GPU work arrives | No partial TP launch; required GPUs are not stolen by younger work; launch waits for every blocker teardown |
| QUE-09 · user-a | Saturate one model across 1/2/4/8 simulated GPU inventories and available physical GPUs | Independent workers/ports/PIDs and disjoint reservations; replica scaling matches useful demand and validated sets |
| QUE-10 · user-a/user-b | Add second-model demand after replicas occupy the host; finish and cancel selected requests | Reclaim appropriate replicas without killing admitted work; surviving replicas keep progress and accounting |
| QUE-11 · operator | Introduce external allocation after planning but before launch; then release it | Live per-GPU check defers unsafe launch; subsequent request recovers; unrelated process remains untouched |
| QUE-12 · operator | Make telemetry unavailable during admission and teardown; restore it | No inference of free GPU capacity; ownership retained on uncertainty; recovery makes progress without manual ledger edits |
| QUE-13 · operator | Make graceful stop fail and require forced kill; then make verified teardown fail | Kill escalation targets owned process group only; unverified STOPPING worker keeps GPU/port; no subsequent conflicting launch |
| QUE-14 · user-a | Cancel the last queued request during cold startup then request another model | No stranded promise/reserve; unnecessary worker follows bounded cleanup; other model proceeds after verified release |
| QUE-15 · operator | Fail one replica or one TP child while other workers serve | Failure remains scoped; affected requests settle once; no overlapping replacement before process-group teardown |
| QUE-16 · observer | Inspect native engine command and issue internal sleep attempt in controlled engine contract lane | Queue launch does not enable sleep or experimental runtime; no automatic llama.cpp fallback after vLLM failure |
## Native vLLM sleep residency

Modes: vllm-sleep. Lane: real-service. Prerequisite: A/B/C with measured residual/wake/RAM evidence and constrained shared GPU; controlled pressure helpers.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| SLP-01 · user-a | Serve A, wait configured idle interval, observe native sleep, then stream to A | Level-1 sleep and wake retain PID; output valid after wake; no cold launch substituted without explicit eviction reason |
| SLP-02 · user-a/user-b | Keep an A stream active and queue incompatible B | A drains admitted work before offload; B gains admission only after safe residency transition; no interrupted healthy stream |
| SLP-03 · user-a | Alternate A/B/A and compare cached wake with cold candidate on another placement | Eligible cached worker is preferred according to policy; activation and queue waits recorded separately |
| SLP-04 · user-a | Send concurrent requests to a sleeping A and requests arriving during OFFLOADING/WAKING | Exactly one transition for that worker; work waits until READY; no duplicate load or overlapping wake |
| SLP-05 · observer | Compare measured sleeping residual and wake peak with live per-GPU usage under repeated transitions | Admission budgets include current residuals and wake peaks; topology and measurement format remain consistent |
| SLP-06 · operator | Pressure one GPU of a sleeping TP worker; wake it | All TP GPUs checked independently; worker does not partially wake or use unmeasured UUID set |
| SLP-07 · operator | Add external GPU pressure and wake A with other sleepers present | Only eligible idle sleepers evicted in LRU order; busy workers and unrelated GPU allocations untouched; memory rechecked after each stop |
| SLP-08 · operator | Make target residual unaccountable; retry wake after telemetry recovers | No guessed residual credit; explicit cold fallback only after verified target teardown or bounded deferral |
| SLP-09 · operator | Run cache population beyond host_cache_max_gib with A/B/C | LRU sleeping workers evicted until measured budget satisfied; no active-request eviction or unbounded RSS/PSS |
| SLP-10 · operator | Reduce available host memory below configured reserve using a bounded isolated helper | RAM headroom policy blocks/evicts cache and reports reason; unrelated host process is never targeted |
| SLP-11 · operator | Exercise cache swap usage at zero/below/equal/above configured threshold under dedicated cgroup | Swap policy uses documented accounting source; prevents unsupported swapping; recovery returns to a usable state |
| SLP-12 · operator | Remove or fail host process-memory telemetry; use cgroup limits lower than host RAM | No false zero-memory safety claim; fallback accounting is conservative and labeled; actual cgroup limit honored |
| SLP-13 · observer | Trigger simultaneous cache-budget enforcement from scheduler, status refresh and offload completion | One serialized decision at a time; no excess/stale eviction or inconsistent host-cache totals |
| SLP-14 · operator | Make sleep endpoint fail, hang or return invalid transition response while other workers serve | Worker becomes unroutable and is safely stopped; callers receive bounded error; no phantom SLEEPING or leaked port |
| SLP-15 · operator | Make wake endpoint fail/hang or kill the sleeping engine before wake | No phantom READY worker; failed request settles; replacement launches only after verified teardown |
| SLP-16 · operator | Preload named models, repeated selectors and wildcard; include missing/disabled/ineligible artifacts | Only configured eligible models preload; startup stays within memory budgets; skipped models have clear reasons |
| SLP-17 · user-a | Send interactive traffic during a multi-model preload | Real backlog progresses ahead of speculative warming; no prolonged user starvation or simultaneous unsafe starts |
| SLP-18 · admin | Enter maintenance while READY, SLEEPING, OFFLOADING and WAKING; then validate and resume | All retained engines/contexts fully unload for maintenance; validation never shares cached GPU state |
| SLP-19 · admin | Disable/edit/revalidate a sleeping profile and request its model | Old sleeping worker cannot revive invalid evidence; new request uses newly eligible launch or is rejected |
| SLP-20 · operator | Restart service with saved cached-worker state and surviving processes | Recovery treats cache as nonrecoverable unless explicitly verified; old groups reconciled before new admission; no duplicate engine ownership |
| SLP-21 · user-a/user-b | Run different request sizes and model-switch rates while cache pressure causes repeated eviction | Correct model responses, fair progress and stable RAM/VRAM/ports; eviction versus wake reasons explain latency |
| SLP-22 · observer | Test minimum residency, idle sleep and transition timeout at exact boundaries | No premature offload of admitted work; settings have observable documented effect; no stuck transition after timeout |
## Optional queue llama.cpp backend

Modes: queue. Lane: real-service. Prerequisite: Explicit llama.cpp-enabled lane, installed pinned llama-server and operator-selected GGUF.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| LCP-01 · staff | Register local GGUF with engine llama.cpp; drain for validation; resume and infer | Shared allocator and full probes used; measured profile is queue-only and no vLLM process starts |
| LCP-02 · staff | Register pinned HF GGUF repository with one GGUF; then try an ambiguous multi-GGUF source | Explicit single artifact selected and identity recorded; ambiguity explains local-file selection requirement |
| LCP-03 · admin | Compare validation and serving effective argv/environment/artifact/GPU placement | Identical measurement-affecting launch specification; context/parallel/offload/split choices preserved |
| LCP-04 · user-a | Run nonstreaming, streaming, concurrency, cancellation and multi-turn calls through public chat API | Same protocol, access, quotas and disconnect guarantees as queue/vLLM |
| LCP-05 · admin | Edit GGUF path/offload/context; Validate/Revalidate selected profile in CLI/TUI/API | Correct engine/artifact/settings re-probed; no vLLM-only TUI dead end or silent old-settings retry |
| LCP-06 · operator | Occupy old hard-coded validation port and exhaust shared allocator; then retry | No fixed-port dependence; failed teardown retains allocator reservation; recovery reuses only freed port |
| LCP-07 · operator | Kill GGUF worker and fail one cleanup step while other queue models run | Same verified process-group/GPU teardown as vLLM; no automatic engine substitution |
| LCP-08 · admin | Attempt llama.cpp registration/profile switch with feature disabled or in sleep mode | Unsupported combination rejected before subprocess launch; UI hides or explains unavailable action |
| LCP-09 · operator | Change GGUF contents and llama-server version; attempt trust and inference | Detected artifact/engine drift invalidates saved eligibility; probes required |
| LCP-10 · user-a | Exercise multi-GPU llama placement if advertised and alternate it with vLLM on managed GPUs | Profile placement/labels match actual engine GPU usage; no leaked reservations or false TP claim |
## Maintenance, failures and recovery

Modes: queue, vllm-sleep. Lane: real-service. Prerequisite: Dedicated instance, exact owned PID inventory, disposable database copies for storage faults.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| OPS-01 · admin | Drain while calls are queued, active and streaming; query status and submit new calls | Queued/new calls receive documented maintenance errors/refunds; admitted calls finish; ready only after all workers stop |
| OPS-02 · admin | Repeat drain/resume from all interfaces including simultaneous calls | Idempotent state transitions or explicit conflict; no double settlement or lost maintenance state |
| OPS-03 · operator | Gracefully stop while idle, serving and validating; restart | Documented shutdown handling, no orphan worker, recoverable pending jobs and consistent service mode |
| OPS-04 · operator | Abruptly kill only test router during queued/active/sleeping/validating phases; restart | Orphan reservations reconciled once; owned surviving workers/probes terminated or startup fails closed; no double charges |
| OPS-05 · operator | Simulate parent exit with surviving TP children and PID reuse unrelated to service | Recovery verifies ownership; never kills reused unrelated PID; cannot clear unresolved GPU ownership |
| OPS-06 · operator | Inject SQLite busy, disk-full and write failure at reserve/admit/settle/profile publish/worker persist | Transactions atomic; errors visible; owned resources eventually clean; no profile published without complete evidence |
| OPS-07 · operator | Inject cancellation between reservation and enqueue, admission and headers, settlement and lease release | Terminal request status, ledger and resource cleanup agree; repeated finalization harmless |
| OPS-08 · operator | Disconnect public network and kill private engine transport before headers and during stream | Finite failures and correct refunds/partial usage; reconnecting user can complete healthy request |
| OPS-09 · operator | Create slow/no-health engine startup and trigger shutdown midway | Startup timeout or interruption leaves no child process, port or GPU reservation; diagnostic points to engine log |
| OPS-10 · operator | Trigger unknown NVML/process inspection during shutdown and startup recovery | Uncertainty retains ownership and blocks unsafe reuse; explicit recovery status, never silent success |
| OPS-11 · admin | Archive stopped test DB, vault, config and revision; restore to isolated location; authenticate and infer after validation | Snapshot integrity and credential decryption verified; exported catalog contains no secret; source files unchanged |
| OPS-12 · operator | Try archive with missing/wrong vault, active owner, existing destination and inconsistent schema | Cannot claim verified rollback backup; error identifies missing prerequisite; source remains untouched |
| OPS-13 · operator | Roll back to previous isolated code/environment/state after stopping new service | Old code never opens new schema; no concurrent GPU owners; previous credentials/catalog behave as archived |
| OPS-14 · observer | Scrutinize logs, CLI errors, reports and exceptions from every failure case | Secrets and sensitive prompt/media content excluded from metrics; diagnostics retain needed request/worker/error identity |
| OPS-15 · observer | After each case and final drain compare API, SQLite, process groups, NVML and bound ports | No stuck lease/live reservation/queued request/unowned worker/port; terminal accounting remains exactly once |
## Real CLI and TUI operator journeys

Modes: queue, vllm-sleep. Lane: tui. Prerequisite: Installed CLI and headless Textual Pilot bound to the real test service; terminal capture for manual spot-check.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| UI-01 · admin | Launch no-argument CLI and interactive command; navigate every page and Back/Quit control | One service-start location; current mode/experimental status accurate; no duplicate or dead navigation |
| UI-02 · admin | Execute every command leaf and every visible action from the surface map against real API; verify via independent GET | Same business outcome, prerequisite and errors on API/CLI/TUI; no private helper mock substitutes for endpoint execution |
| UI-03 · admin | Run create/edit/register/trust forms with local validation failure, HTTP 4xx, 5xx, timeout and recovery | Values retained and correction resubmits once; no accidental operation on stale selected object |
| UI-04 · admin | Double-click/press Enter repeatedly and refresh while a mutation is in flight | At most one mutation; controls remain disabled or show pending state; reset correctly after success and failure |
| UI-05 · admin | Select middle row, sort/refresh as rows change, remove selected row and refresh again | Stable object identity retained when present; safe fallback when gone; no action applied by stale row index |
| UI-06 · admin | Resize to 80x24, 120x40 and larger; use keyboard-only forms, scrolling and modal buttons | Labels, fields and actions reachable; focused text visible; no clipped destructive action or duplicate control |
| UI-07 · staff/user-a | Use lower-role credentials and navigate management screens; force-click unavailable actions | UI communicates role/prerequisite; server still enforces permissions; forbidden actions cannot mutate state |
| UI-08 · admin | Observe profile Enable/Disable control for active, inactive and ineligible profile states | Exactly one state-appropriate action; unavailable enable explains revalidation/trust prerequisite |
| UI-09 · admin | Select saved evidence from older fingerprint; submit Advanced trust with missing then valid reason | Source and affected profile clear; no bulk/both-backend verification toggle; actor/reason audited |
| UI-10 · admin | Use dashboard refresh and maintenance status while traffic runs | Refresh only refreshes; values reconcile with real usage/worker state; selection and scroll remain stable |
| UI-11 · operator | Inspect connection info with environment key, local vault key and remote URL; start service from TUI | Credential source accurately labeled without revealing secret; same config resolver; selected explicit mode reaches server |
| UI-12 · admin | Cancel form and destructive confirmation via button and Escape; confirm once afterward | Cancel mutates nothing and preserves underlying selection; confirmation results in one operation |
| UI-13 · admin | Copy selected key then rotate/revoke it and refresh | Clipboard uses the selected current credential; copy does not expose keys in logs or operate on stale row |
| UI-14 · staff | Run registration, failure review and revalidation entirely through TUI with real persisted jobs | Form values and job progress survive errors/restarts; selected engine/context/overrides match actual probe |
## Controlled transitions and final soak

Modes: queue, vllm-sleep. Lane: physical. Prerequisite: All blockers closed, exact tested wheel/config/engine versions, A/B distinct artifacts, independent observers.

| ID / actor | User/operator steps | Required outcome |
| --- | --- | --- |
| SOAK-01 · operator | Run the previous code/environment with the same request trace and isolated prior state; record three baseline trials | Comparable latency/throughput/queue/memory baseline and variability recorded before judging regression |
| SOAK-02 · user-a/user-b | Execute 100 complete controlled A/B residency cycles with a valid request before and after each switch | Each cycle proven by events plus PID/NVML observations; at least 100 actual transitions; no count from requests or polling artifacts alone |
| SOAK-03 · user-a/user-b/staff | Run 60 continuous minutes of seeded mixed interactive/batch/stream traffic with two accounts and at least two distinct artifacts | One full hour of measured load; no unexplained error, starvation, accounting mismatch or resource leak; complete latency and throughput distributions |
| SOAK-04 · operator | Run bounded fault/recovery schedule separately after healthy soak; then repeat clean traffic and drain | Every injected error is attributed to expected cause; recovery bounded; healthy-soak success is not diluted by ignored errors |
| SOAK-05 · observer | Reconcile every request ID, complete cycle, terminal worker, GPU/process owner, port and quota ledger row; verify artifact/config hashes unchanged | Signed machine-readable result per case/mode/surface; missing/skipped/xfail case is not a pass; zero unresolved release blocker |
