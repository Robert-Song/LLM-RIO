# Beta user and model migration

This records the one-time migration into `state/release/llm-rio.db`. It was performed
after the release target was initialized and while no LLM-RIO service owned either
database or the GPUs. The beta source, its matching API-key vault, archived copy, and
model artifacts were left untouched.

## Schema differences

The configured beta database used schema version 0 and 15 tables. Release uses schema
version 1 and 14 tables. Release removes the beta-only `model_verification_jobs` table;
it did not contain rows. `model_catalog` adds `source_type`, `local_path`, and `engine`.
The beta catalog did not distinguish these fields. All 59 beta rows had a Hugging Face
repository, and all saved profiles identified vLLM, so these were migrated as
`source_type="huggingface"`, no local path, and `engine="vllm"`.

Profile records also needed a data conversion. Every beta profile lacked explicit
`serving_mode` and `launch_binding` fields. The migration inferred queue only where
saved native launch evidence identified queue, inferred native sleep only where its
complete sleep/wake measurement vectors were present, and left unknown or experimental
profiles inactive. Reusable profiles received a release launch binding and an
attributed migration audit. No validation probes ran.

## Preserved data and history

The migration copied 7 API-key rows, 7 quota accounts, 5 model grants, 59 model rows,
and all 247 profile records. All 3 live beta API key values are identical after
migration; their ciphertext was re-encrypted with the release database's vault. The
four revoked-key tombstones, roles, account links, quota balances/limits/reset fields,
grants, model states, revisions, capabilities, request limits, and artifact manifests
were preserved. The catalog has 54 `AVAILABLE`, 3 `DISABLED`, and 2
`NEEDS_ADMIN_REVIEW` models. All 54 available artifact paths existed during preflight;
artifacts were not copied or modified.

The selected release mode is queue. The beta had 170 active queue profile rows across
54 models. Some rows described the same effective placement after the release machine
fingerprint and launch binding were applied. Release uniqueness rules collapse those
75 equivalent records into one audited profile per effective configuration, leaving
95 active queue profiles with the same 54-model coverage. Their duplicate source IDs
are retained in the migration audit, and every measured field was identical within
those duplicate groups. The 47 complete native sleep profiles remain eligible but
inactive, matching their beta state. Twenty profiles had no provable native mode, seven
belonged to experimental kv-cached, and two lacked required measurements; these 29
stay inactive and need review or new validation before use.

One hundred twenty-two source profiles carried a different machine fingerprint than
the beta service record. Their profile revisions, current vLLM version, launch settings,
and GPU UUID placements were checked before migration and recorded in an attributed
override. Ninety-three older profiles also had beta trust overrides without actor or
reason; the release audit records the migration actor and reason instead.

Inference requests, reservations, quota ledger and usage summaries, workers, runtime
events, and 57 registration-job history rows were omitted. The release target's 14
startup-smoke events were also cleared. The beta had no verification-job history rows.

## Evidence and rollback

The read-only preflight matched the configured beta database against its verified
archive and matching vault, checked both SQLite integrity, verified credentials, and
confirmed matching native engine settings and vLLM 0.26.0 identity. Post-migration
checks confirmed release SQLite integrity, exact API-key values and metadata, exact
quota accounts, common model fields, grants, model states, and active queue model
coverage. Current launch-binding checks found all 95 active queue profiles and all 47
sleep profiles eligible. No model was loaded or called during these checks.

The rollback snapshot taken immediately before the committed migration is
`db_backup/release-before-user-model-migration-20260923T092455Z/`. Its database,
vault, and configuration hashes and SQLite integrity were verified. The original beta
database and matching archive remain available separately. Preserve these directories
with restrictive permissions; they contain credential-bearing material.

The migration utility is [`scripts/migrate_release_data.py`](../../scripts/migrate_release_data.py).
On a different installation, first stop the owning service, create and verify a
matching beta database/vault/config archive, and use an empty version-1 release target.
Run it once without `--apply` to inspect its aggregate preflight, then run with
`--apply` to create a release rollback snapshot and migrate atomically. It refuses a
nonempty release target. Do not rerun it against this already migrated database.
