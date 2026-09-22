> **Historical beta reference.** These instructions are retired. See the [release documentation](docs/README.md).

# Rebuild per-server databases while retaining credentials and models

This procedure creates a **new database path for each server**, retaining the old
files for rollback. Run each database's commands on the host that normally owns
that database. It does not merge the two servers.

The rebuild preserves API-key IDs, secrets, hashes, roles and activation flags;
model IDs, nicknames, artifact paths and defaults; model grants; valid active and
inactive placement profiles; registration and verification jobs; all runtime
events; and the recorded machine fingerprint. It copies the existing key vault to
the name required by the new database. Model weights and external validation logs
stay in place.

It drops request, reservation, ledger, usage-summary and worker history. Quota
limits and unlimited flags are preserved, balances are reset to their configured
limits, and usage baselines/counters start fresh. The new service mode is ACTIVE.
Existing hardware/backend compatibility checks still apply to preserved profiles.

## 1. Stop each owning service cleanly

Pause labmate traffic and disable any supervisor/cron auto-restart. Use Ctrl+C in
the service terminal or your normal service-manager stop command, then wait for
shutdown to complete. Ensure that the service **and its worker/probe processes**
have exited. Prefer graceful shutdown to `kill -9`: the replacement database will
not retain the old worker PIDs for startup cleanup.

Keep both services stopped until their individual migration and configuration
changes are complete. Do not delete or manually separate existing `.db`, `-wal`,
`-shm`, or hidden vault files. Their original directory is the rollback source.
The `--server-stopped` option is your assertion that shutdown has completed; it
cannot verify processes on another host.

## 2. Build and validate a new database

Run from the LLM-RIO project directory, using the updated checkout and its existing
Python environment. These commands do not overwrite or alter source records.
SQLite can create coordination sidecars when opening a WAL database read-only.

For the primary server (`config.toml`):

```bash
PYTHONPATH=src .venv/bin/python -m llm_rio.database_rebuild \
  --source state/llm-rio.db \
  --destination state/llm-rio-clean-20260921.db \
  --exclude-profile d0694b28-63d7-4e82-ac00-ac234d2bd2b6 \
  --server-stopped
```

The explicitly excluded ID is the proven request-shaped recovery artifact, not a
valid model profile. The command refuses to exclude a valid profile or an unknown
ID. No other malformed profiles are silently dropped.

For the Barra server (`config.barra.toml`), run on Barra:

```bash
PYTHONPATH=src .venv/bin/python -m llm_rio.database_rebuild \
  --source state_barra/llm-rio.db \
  --destination state_barra/llm-rio-clean-20260921.db \
  --exclude-profile d0694b28-63d7-4e82-ac00-ac234d2bd2b6 \
  --server-stopped
```

Proceed only if the command exits successfully and reports both `integrity_check`
and `foreign_key_check` as `ok`. Review the `preserved_rows` and
`active_keys_verified` counts. The primary preflight on September 21 retained 7
keys (3 active), 59 models, 244 profiles, 57 model jobs and 11,669 runtime events;
the rebuilt database was about 5.2 MiB.

Barra's live database initially could not be read reliably from the investigation
host. After shutdown, inspection confirmed that it contains the same misplaced
request record (disabled on Barra). With that explicit exclusion, its check-only
rebuild passed both integrity checks: 10 keys (6 active), 45 models, 285 valid
profiles, 43 model jobs and 5,185 runtime events, occupying about 2.4 MiB.
If the stopped-host command reports
`file is not a database`, `database disk image is malformed`, missing tables,
invalid keys/profiles, or broken references, stop that migration and retain the
original files. Recover its required metadata separately before changing its
configuration. Do not initialize an empty database, skip required tables, or
silently substitute an older backup: that could lose keys or validation history.

Existing destination databases, vaults and sidecars are never overwritten. If
rerunning, use a fresh destination filename and configure that exact filename.
A run without output creation is available by replacing `--destination ...` and
`--server-stopped` with `--check-only`.

## 3. Change only the successful server's database path

Keep a copy of each original configuration. For a successful primary migration,
set this top-level value in `config.toml`:

```toml
database_path = "./state/llm-rio-clean-20260921.db"
```

For a successful Barra migration, set this in `config.barra.toml`:

```toml
database_path = "./state_barra/llm-rio-clean-20260921.db"
```

If a launcher, shell, or `.env` sets `LLMRIO_DATABASE_PATH`, update/remove that
override too; it takes precedence over TOML. The rebuild automatically creates
`.llm-rio-clean-20260921-api-key-vault` beside each new database. Keep it with its
database; do not generate a replacement vault. Leave model-store paths, engine
settings, API ports and validation log directories unchanged.

## 4. Restart and verify on each owning host

Use your existing launch environment and serving-mode arguments. For the usual
config-driven launch, the commands are:

```bash
# On the primary host:
./llmctl serve --config config.toml

# On Barra, in its own terminal/host:
./llmctl serve --config config.barra.toml
```

Verify that the TUI lists the expected users, models, paths and validation
profiles. Send a small inference using an existing labmate API key to verify that
no key rotation or client configuration change is needed. New usage should begin
at zero and increase normally. Re-enable traffic/auto-restart after verification.

For the Barra TUI, select the corresponding config as usual:

```bash
LLMRIO_CONFIG=config.barra.toml ./llmctl interactive
```

Rollback: stop the replacement service and workers, restore the original
`database_path` (including any environment override), then restart with the old
files and their original vault still together. Rollback discards activity recorded
only in the replacement database; it also brings back the old integrity problems.
Keep the originals until the replacement has been verified.

## Future summarization

The TUI action and `llmctl summarize` continue to work after the rebuild. An
integration test covers opening a rebuilt database, authenticating an existing
key, loading its saved profile, recording new usage, and summarizing that usage
without changing its charged total or balance. Summarization's HTTP read timeout
is now ten minutes, while other requests keep their existing timeout.

Use the TUI button or run, for the appropriate server:

```bash
LLMRIO_CONFIG=config.toml ./llmctl summarize
# Or on Barra:
LLMRIO_CONFIG=config.barra.toml ./llmctl summarize
```

Summarizing regularly avoids another multi-million-row batch. It preserves
lifetime totals, removes settled per-call details and starts a fresh current
window. It can temporarily delay other database operations and does not
necessarily shrink the allocated database file. No finite timeout guarantees
completion for arbitrarily large histories. Rebuilding or summarizing does not
resolve the existing NFS/WAL deployment limitation.
