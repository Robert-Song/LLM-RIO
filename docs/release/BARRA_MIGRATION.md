# Barra release-state migration

Barra now has a separate queue-mode release configuration and schema-version-1
database. The beta config and both legacy Barra databases remain in place for rollback
and reference. The release state is at `state_barra/release/llm-rio.db`; the local,
host-specific config is `config.barra.release.toml` and is intentionally ignored by
Git.

## Migrated state

The Barra catalog contained 45 pinned Hugging Face model revisions. All 45 match
records in the main release catalog, including their recorded model settings and
artifact manifests. The release copy retains the 42 `AVAILABLE` and 3 `DISABLED`
states and references the existing model files; no artifacts were copied or removed.
Every file in the 42 available snapshots matched the saved size and Hugging Face
content-addressed blob name during preflight. This is catalog migration, not a router
allowlist: vLLM remains responsible for determining which registered models it can
serve, and additional supported models can be registered normally.

Barra had 77 active beta queue profiles. Each profile matched a pinned model revision,
vLLM 0.26.0, complete queue measurements, and at least one placement using Barra's
configured GPU UUIDs. Five equivalent historical records had identical measurements
and were collapsed, leaving 72 active profiles covering all 42 available models.
Their launch bindings were computed from the release queue config, and every profile
records its source profile, a digest of its source measurements, migration actor, and
reason. No profile from the main server was used because that server has different GPU
UUIDs. Native sleep and kv-cached profiles were not copied into Barra's queue state.

The initial bootstrap administrator was replaced with the users from the current
release database. Barra now has the same seven API-key records and seven quota
accounts: three active keys (two administrators and one user) and four inactive
records. API-key values, roles, active status, timestamps, quota balances, limits, and
reset settings match current release; the values were re-encrypted with Barra's
separate vault. The five current-release model grants all refer to models in Barra's
catalog and were copied. Model creator references now point to the matching imported
key IDs.

No beta identities were copied. Request, reservation, ledger, usage, registration-job,
worker, and runtime-event history was not carried over. The four Barra bootstrap
inventory/queue startup events were retained in the pre-migration archive and cleared
from the release database. The local CLI reads the existing Barra vault, so the
current-release API-key values work unchanged with Barra's API endpoint. Key values
are not stored in this document.

Before replacing the bootstrap administrator, a second verified rollback snapshot of
the Barra release database, config, and vault was created at
`db_backup/barra-before-api-key-migration-20260923T175452.563446Z/`. The migration
utility's read-only verifier confirmed that all seven key rows decrypt to the same
values as current release, quota/account fields match, grants map to Barra's catalog,
and SQLite integrity and foreign keys pass.

## Preserved files and rollback

Before creating release state, the migration made and hash-verified a consistent
SQLite backup of the database selected by `config.barra.toml`, along with that config
and its matching API-key vault. The archive is
`db_backup/barra-before-release-migration-20260923T110218Z/`. Its files are mode
`0600`, and the containing directory is mode `0700`. The archive manifest records
SHA-256 hashes and the repository revision identifier.

The configured beta database remains at
`state_barra/llm-rio-clean-20260921.db`; the separate older
`state_barra/llm-rio.db` was not opened or changed. `config.barra.toml` was not
changed. The new config continues to use Barra's existing `logs_barra` directory,
and existing log files were not migrated or removed. The active main release database
was opened read-only and was not used as a write target.

To return to beta, stop the release process first and use the preserved beta config
and database. Never point beta code at `state_barra/release/llm-rio.db`, and never run
both services against Barra's managed GPUs at once.

## Starting Barra

Start the release service on Barra, where the configured UUIDs are present:

```sh
./llmctl serve --config config.barra.release.toml
```

For later local administration, select the same config so the CLI uses Barra's port,
database, and vault:

```sh
LLMRIO_CONFIG_FILE=config.barra.release.toml ./llmctl status
```

The generated config preserves Barra's exact two-GPU UUID selection, machine ID,
address, port, model store, and `logs_barra` path. It uses the release queue and engine
settings. It does not copy the main server's Hugging Face credential; supply
`LLMRIO_HF_TOKEN` through Barra's protected service environment if gated model
registration requires it.

The migration host could not inspect Barra's NVML inventory or make model calls. The
database retains Barra's most recently recorded machine fingerprint, so the running
service will use the profiles only if its actual GPU UUIDs, driver, CUDA identity,
vLLM binary, and effective launch settings still match. Run the following check and
the real-GPU cases in
[the manual GPU acceptance kit](MANUAL_GPU_TEST.md) on Barra before production
traffic. If the inventory or launch binding differs, the profiles fail closed and
must be qualified on that host.

```sh
./llmctl doctor --config config.barra.release.toml
```

The one-time database/profile migration utility is
[`scripts/migrate_barra.py`](../../scripts/migrate_barra.py). The user and API-key
migration utility is [`scripts/migrate_barra_users.py`](../../scripts/migrate_barra_users.py);
it supports a read-only preflight and post-migration verifier, and requires an explicit
operator confirmation that the remote service is stopped before applying.
