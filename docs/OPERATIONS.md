# Operations and recovery

## Preserve the lab installation

1. Drain and stop the service that owns the lab database and GPUs. Do not stop unrelated
   GPU processes. Record the code revision and retain its environment for rollback.
2. Archive the consistent SQLite database, its matching `.<stem>-api-key-vault`, and its
   configuration. The archive destination must not already exist:

   ```sh
   uv run --no-sync python -m llm_rio.operations.archive state/lab.db \
     db_backup/lab-release-boundary --config config.toml --server-stopped
   ```

   `--server-stopped` is an operator assertion: the utility cannot stop or identify every
   remote owner. SQLite backup and integrity checking happen before a completion manifest
   is written. A directory without `manifest.json` is an incomplete archive.
3. Verify the manifest hashes, preserve archive permissions, and retain its credential-free
   `catalog.json`/`catalog.csv`. The catalog contains names, source revisions, and artifact
   paths. The database/vault/config are sensitive even though the catalog is credential-free.
4. Start release state at a new path, normally `state/release/llm-rio.db`, with newly generated
   credentials. Release startup refuses existing unversioned beta databases. Re-register
   and validate artifacts; do not import beta verification booleans or edit schema versions.

The initial lab archive in this workspace is under ignored
`db_backup/final-cleanup-baseline/`; it is local evidence, not part of a published package.
Original databases and model files are preserved.

## Registration and validation

For HF sources, resolve and save an immutable revision, download it to the managed cache,
then run probes. For local sources, use an absolute directory for vLLM or an absolute
`.gguf` file for queue's optional llama.cpp engine. No local source is copied or deleted.
Content manifests identify local artifacts. Detected changes block serving until revalidation.
Keep local sources immutable while registered: size/mtime checks provide inexpensive
serving-time change detection, while validation rebuilds the full content manifest.

Native GPU probes require maintenance. Drain waits for admitted requests, verifies engine
teardown, then permits validation. Review the persisted job stage and engine log on error.
`models validate NAME` retries the same workflow and always runs probes. Model-specific
context, concurrency and launch overrides are entered in the TUI validation form or the
validation API. Failed validation leaves the model requiring administrator review.

Use the profile's measured placement, context and concurrency; never infer eligibility
from model names, file size alone, or another GPU's result. Enablement and measurement
validity are separate. Editing launch settings invalidates evidence. Advanced trust is
appropriate only for existing complete compatible measurements; a reason and actor are
recorded. It may accept a changed machine fingerprint, but not changed UUIDs, artifacts,
mode, engine/launch configuration, or invalidated measurements.

## GPU and memory qualification

Choose managed GPU UUIDs explicitly on shared hosts. Queue releases GPU resources only
when teardown is verified. Native sleep measures residual memory and wake peaks and enforces
RAM/swap limits. Foreign allocations and missing telemetry can defer placement. Inspect
`status`, `status --dashboard`, validation job logs and runtime events before retrying.
Do not solve a teardown failure by clearing reservations manually while the process may live.

Follow [release qualification](release/QUALIFICATION.md). The deployment manifest describes
the tested installation, not a product allowlist. New engine/model/hardware combinations
remain eligible for registration and are qualified by their own validation results.

## Maintenance, interruption, and recovery

Use Drain before configuration changes, backups, engine upgrades or mode changes. Resume
is rejected while validation still owns GPUs. Stop the API gracefully; cancelled probes
must terminate their process groups and verify GPU reclamation. On restart, persisted jobs
resume and request/worker recovery reconciles ownership and quota reservations. Startup
must complete recovery before accepting traffic.

When a worker cannot terminate, preserve its PID, process group and GPU diagnostics. A
stuck reservation is a release blocker; do not run another service against those GPUs.
Memory or engine errors during sleep/wake cause cleanup, not unmeasured reuse.

Back up a stopped release database with its matching vault and configuration. Restore
these as a unit into a fresh location. Do not restore a database with an unrelated vault.

## Rollback

Stop the new service and verify its workers and GPU reservations are gone. Restore the
previous code **and environment**, previous configuration, and matching archived database
and vault into a separate location. Start only that installation. Never run old code against
the release schema, or old and new services concurrently on the same managed GPUs. Retain
new-state diagnostics separately so failed qualification can be investigated.
