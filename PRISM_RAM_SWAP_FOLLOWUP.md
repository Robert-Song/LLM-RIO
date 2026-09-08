# Prism host-RAM budget and swap follow-up

## Purpose

This is the handoff for the next development branch after the single-host
Prism-style implementation.  The current implementation has been rehearsed on
two GPUs and provides fast `SLEEPING -> WAKING` transitions using vLLM level-1
sleep, kvcached, and a bounded LRU cache of *worker processes*.

It does **not** yet enforce a byte-accurate host-RAM cache budget, and it must
not represent normal Linux swap as a transparent or supported model-cache
tier.  This document defines the work needed before making either claim.

## Current, validated behavior

- A worker that is idle for `prism_idle_sleep_seconds` leaves GPU VRAM while
  its immutable weight copy remains in host RAM.  Its process and PID persist.
- A requested `SLEEPING` worker is woken before a new vLLM process is started.
  On the rehearsal host, cached model switches completed in about 3--4.3
  seconds end-to-end and a cached Qwen wake/response took about 0.56 seconds.
- `prism_max_cached_workers_per_gpu` is the current cache bound.  It limits
  live cached workers associated with each GPU; it is not a GiB limit.
- Once that count limit is exhausted, the planner selects the least recently
  used `SLEEPING` worker with no admitted work, stops its process, and records
  `prism_ram_cache_lru`.  A later call to that model is a cold disk load.
- No active or admitted request may be slept or LRU-evicted.

The implementation lives on the `prism` branch.  The last validated commit at
the time of this note is `0a2192a`.

## Problem to solve

The process-count limit is intentionally simple, but a worker count does not
correspond to a memory budget:

- cached models have very different weight sizes;
- TP shape and duplicated replicas alter the amount of RAM retained;
- some mappings may be shared and should not be double-counted;
- CUDA-pinned or locked CPU buffers may be unevictable by Linux swap;
- overall host pressure includes the OS, other services, file cache, and the
  real service on port 8002.

The current value `prism_max_cached_workers_per_gpu = 8` is therefore a
capacity policy, not a safety guarantee.  It is acceptable for the rehearsed
working set, but should not be treated as permission to retain sixteen arbitrary
large models on a two-GPU host.

## Explicit non-goal: transparent SSD swap

Do not implement a feature that promises that SSD-backed virtual memory makes
the RAM cache seamlessly larger.  Linux can swap ordinary pageable anonymous
pages, but a cached vLLM process may hold pinned or otherwise non-swappable
buffers.  Even swappable weights can fault back over many seconds or minutes;
the kernel can also invoke an OOM killer before a useful wake completes.

Do not budget against `VmSize`/virtual address space or a configurable virtual
page size.  Those values do not represent resident usable host memory.  The
controller must budget physical/cgroup memory and must account for actual swap
pressure separately.  The kernel page size matters only for rounding reported
metrics, not as a capacity-control unit.

If future work deliberately introduces a disk tier, make it an explicit,
observable cold-tier design (for example, a stopped worker whose immutable
model snapshot already exists on local SSD).  It must not rely on incidental
kernel swapping of a live vLLM process, and it must be reported as a cold, not
a RAM-cache, activation.

## Proposed production design

Create a branch such as `prism-host-ram-budget` from `prism`.

### 1. Add explicit configuration

Use MiB or bytes, never a model count alone.  Exact names are open, but the
following is the intended policy shape:

```toml
# Total memory the controller may retain in SLEEPING Prism workers.
prism_host_cache_max_mib = 307200

# Keep this much host/cgroup memory free for the OS and non-Prism services.
prism_host_cache_min_available_mib = 65536

# Legacy placement-count guard; keep it as a secondary anti-context limit.
prism_max_cached_workers_per_gpu = 8

# Do not claim swap-backed cached wakes.  Evict cached workers before swap is
# materially used; "warn" may be useful for observability during rollout.
prism_swap_policy = "protect"
prism_swap_max_used_mib = 0
```

The values above are examples, not a preset.  Choose them after measuring the
host's physical RAM, baseline services, and the actual cached PSS/RSS of the
Qwen, Gemma, and Laguna placements.  Leave enough headroom for model loading,
validation, the OS page cache, and the service on port 8002.

Reject an unsafe configuration at startup: budget lower than a viable cached
worker, no configured headroom, or a cache budget above the effective cgroup
or physical memory allocation.

### 2. Account for real resident memory

Prefer cgroup v2 for enforcement when the service is deployed in a cgroup:

- place LLM-RIO and every worker in a known parent cgroup;
- read `memory.current`, `memory.max`, `memory.high`, `memory.events`, and
  `memory.swap.current`/`memory.swap.max` from that cgroup;
- use the cgroup's limit, when finite, rather than host-wide RAM as the hard
  ceiling.

For per-worker attribution and LRU explanations, sample
`/proc/<pid>/smaps_rollup` and record at least RSS, PSS, `Swap`, and `SwapPss`.
PSS is preferable for ranking process-specific retained weights because it
avoids charging shared mappings fully to every worker.  The hard admission
guard must use aggregate cgroup/system memory, not a sum of potentially stale
or overlapping PSS samples.

Use `/proc/meminfo` (`MemAvailable`, `SwapTotal`, `SwapFree`) only as a
fallback when cgroup data is unavailable.  Make the accounting source visible
in status and logs.

### 3. Evict before pressure, not after OOM

Before a worker completes its transition to `SLEEPING`, and before a cached
worker is retained after waking, compute whether retention would violate either
the cache budget or minimum available memory.  If it would:

1. choose the LRU eligible `SLEEPING` worker;
2. stop it cleanly and wait for its process/cgroup memory to fall;
3. resample memory;
4. repeat until the incoming cached worker fits or no eligible victim exists;
5. if none fits, do not retain the incoming worker in RAM; stop it after its
   work completes and mark the decision `prism_host_ram_budget`.

If swap use exceeds `prism_swap_max_used_mib`, treat it as an urgent cache
pressure signal: stop eligible sleeping workers first, emit a high-severity
runtime event, and never advertise their next activation as a normal RAM-cache
hit.  An active request remains protected even under pressure; only its future
retention may be denied.

Keep the current per-GPU cached-worker limit as a secondary guard because a
sleeping vLLM process retains a CUDA context and some VRAM even when its model
weights are in host RAM.

### 4. Make cache state truthful

Extend worker/status data with fields such as:

```text
weight_cache: host_ram | disk_cold | unavailable
host_cache_accounted_mib
host_cache_accounting_source: cgroup | smaps_rollup | meminfo_fallback
process_rss_mib, process_pss_mib, process_swap_mib
last_cache_eviction_reason: lru_count | ram_budget | swap_pressure | shutdown
```

`host_ram` may only be shown when the process is still alive, is actually
`SLEEPING`, and swap pressure is within policy.  A stopped process is
`disk_cold`, even if the Linux page cache happens to contain its model files.

## Tests required before enabling a RAM budget in production

Run the following work in an isolated test branch and on a dedicated port.  Do
not create intentional RAM or swap pressure on the real port-8002 service.

### Unit tests

- cache admission below, exactly at, and above the byte budget;
- LRU ordering with unequal worker memory sizes and identical access times;
- no eviction of `READY`, `WAKING`, `OFFLOADING`, or admitted workers;
- no retained cache entry when the incoming process alone is larger than the
  budget;
- cgroup accounting preferred over host fallback, including unavailable or
  malformed procfs data;
- swap-pressure response and status labels;
- restart/orphan recovery clears stale PIDs and cache-accounting metadata.

### Controlled integration tests

Use a cgroup or container memory/swap limit, not the whole server:

1. Start two small validated models; put both into `SLEEPING`.
2. Set a budget that holds exactly one measured cached model.
3. Request the second model and verify the true LRU process exits before the
   incoming cached worker is retained.
4. Confirm the survivor has a same-PID warm wake; confirm the evicted model
   launches a new PID and follows the cold path.
5. Run an in-flight generation while forcing cache pressure.  Verify it is not
   terminated and that eviction happens only after it becomes idle.
6. Create controlled swap pressure only inside the test cgroup.  Verify a
   `swap_pressure` event, prompt eviction of sleeping workers, and no false
   "warm hit" metric.
7. Kill and restart the controller; verify no stale worker/PID or budget
   accounting blocks a new placement.

### Live acceptance criteria

- The controller keeps aggregate cgroup memory below the configured budget and
  retains the configured free-memory headroom through repeated model switches.
- LRU eviction is visible in the dashboard, event log, and API status with a
  precise reason.
- A RAM-cache hit remains fast under no-swap conditions; a swap-pressured or
  disk-cold activation is labelled honestly rather than included in the warm
  latency claim.
- No test affects port 8002 or uses global memory exhaustion.

## Deployment/rollback notes

Ship this behind a disabled-by-default budget setting.  Until the integration
tests pass, retain the existing worker-count policy and set a conservative
count appropriate to the known working set.  A fast rollback is to disable the
byte-budget policy, restart the controller, and let existing `SLEEPING` workers
be reclaimed normally.

Before enabling a byte budget on the production-demo host, record:

- physical RAM and effective cgroup memory/swap limits;
- steady-state `MemAvailable` with the port-8002 service running;
- PSS/RSS/swap for each preloaded Qwen replica, Gemma, and Laguna worker after
  their first completed sleep;
- chosen cache budget and reserved headroom, with the rationale in the
  deployment runbook.
