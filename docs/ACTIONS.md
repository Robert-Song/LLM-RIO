# Administration action matrix

Paths are relative to the API base URL. `NAME` is a model/key nickname; `ID` is its
stable identifier. Admin actions require an active admin key; staff actions allow
TA and admin keys. Authentication failures return 401 and role restrictions 403.
Invalid input returns 422, absent objects 404, and conflicting/running jobs 409.
All CLI entries are prefixed by `llm-rio` (or `./llmctl`).

| Operation | HTTP | CLI | TUI | Role / prerequisite / specific outcome |
| --- | --- | --- | --- | --- |
| Capabilities | `GET /admin/capabilities` | `capabilities` | Mode and engines in title; engine choices | Admin; unavailable engines rejected server-side |
| Worker/resource snapshot | `GET /admin/status` | `status` | Maintenance → Refresh status | Admin; reports validation reservations |
| Dashboard | `GET /admin/dashboard` | `status --dashboard` | Dashboard → Refresh dashboard | Admin; refresh only |
| List keys | `GET /admin/keys` | `keys list`, `keys show NAME`, `keys usage NAME` | Users | Admin; includes secrets, handle output accordingly |
| Create key | `POST /admin/keys` | `keys create NAME --role user` | Users → Create | Admin; optional grants/quota account; 404 for missing grants/account |
| Rotate key | `POST /admin/keys/{key_id}/rotate` | `keys rotate NAME` | Users → Rotate | Admin; previous secret stops working |
| Revoke/restore key | `POST /admin/keys/{key_id}/revoke`, `POST /admin/keys/{key_id}/restore` | `keys revoke NAME`, `keys restore NAME` | Users → Revoke/Restore | Admin; audit retained |
| Delete key | `DELETE /admin/keys/{key_id}` | `keys delete NAME` | Users → Delete | Admin; credential utility removed, audit retained |
| Set quota | `PUT /admin/keys/{key_id}/quota` | `keys quota NAME --limit N --limited` | Users → Set quota | Admin; reservation/settlement remains atomic |
| Reset usage | `POST /admin/keys/{key_id}/usage/reset` | `keys reset-usage NAME` | Users → Reset usage | Admin; lifetime audit retained |
| Compact settled usage | `POST /admin/usage/summarize` | `summarize` | Maintenance → Summarize usage | Admin; live reservations excluded; idempotent settlement retained |
| List catalog/jobs | `GET /staff/models` | `models list` | Models → Refresh | Staff |
| Register source | `POST /staff/models` | `models add NAME REPO --revision SHA`, or `--local-path PATH --engine vllm` | Models → Add model | Staff; absolute server-local path or HF repo; unsupported engine/path 422 |
| Review validation | `GET /staff/model-jobs/{job_id}` | `models review NAME` | Models → Review job | Staff; includes failed stage and diagnostics |
| Validate/revalidate/retry | `POST /staff/model-jobs/{job_id}/retry` | `models validate NAME --max-model-len N` | Models/Profiles → Validate/Revalidate | Staff; always probes, retains other saved overrides; native probes wait for maintenance; running job 409 |
| Disable catalog model | `POST /staff/models/{model_id}/disable` | `models disable NAME` | Models → Disable | Staff; drains admitted work; successful revalidation restores availability |
| Model access | `POST /staff/model-access` | `models access KEY`, `models grant KEY NAME`, `models revoke KEY NAME` | Users → Change model access; Models → Change user access | Staff; missing model/key 404 |
| Replace grants | `PUT /staff/keys/{key_id}/model-grants` | Use grant/revoke to adjust set | Same access forms | Staff; validates full requested set |
| Request defaults | `PATCH /admin/models/{model_id}` | `models defaults NAME --values '{"temperature":0}'` | Models → Edit model | Admin; schema validation; null clears a field |
| Clone logical model | `POST /admin/models/{model_id}/clone` | `models clone-profile SOURCE NEW` | Models → Clone model | Admin; shared artifact, separate queue/grants; launch changes invalidate evidence |
| List profiles/evidence | `GET /admin/models/{model_id}/profiles` | `models profiles NAME --saved --json` | Models → Profiles; Advanced source selector | Admin; saved evidence includes older machine fingerprints |
| Edit launch profile | `PATCH /admin/models/{model_id}/profiles/{profile_id}` | `models profile-edit NAME PROFILE ...` | Profiles → Edit selected | Admin; engine capability enforced; launch changes require probes; conflicting profile 409 |
| Enable/disable profile | `POST /admin/models/{model_id}/profiles/{profile_id}/enable`, `POST /admin/models/{model_id}/profiles/{profile_id}/disable` | `models profile-state NAME PROFILE --enable` or `--disable` | Profiles → one state-appropriate Enable/Disable button | Admin; activation does not manufacture eligibility; disable drains workers |
| Advanced trust | `POST /admin/models/{model_id}/profiles/{profile_id}/trust` | `models trust-measurements NAME ID --reason TEXT` | Profiles → Advanced → Trust saved measurements | Admin; nonblank reason, compatible artifact/mode/engine/settings/UUIDs and complete evidence; 409 if invalid/incompatible; actor/source audited |
| Drain/resume | `POST /admin/maintenance` | `maintenance drain`, `maintenance resume` | Maintenance → Drain/Resume | Admin; resume blocked while validation owns GPUs |
| Maintenance status | `GET /admin/maintenance` | `maintenance status` | Maintenance → Refresh status | Admin |
| Qualification accounting | `GET /admin/requests?test_run_id=ID` | Qualification tool | Dashboard shows live requests | Admin; malformed ID 422 |
| Start service | Process launch, no HTTP operation | `serve --mode queue --config PATH` | Diagnostics → Start service | Host operator; explicit mode, fresh schema, exclusive GPU ownership |
| Host diagnostics | Local inspection | `doctor --json` | Diagnostics → Run doctor | Host operator; does not load a model |

User inference and own-usage operations are documented in [Inference](INFERENCE.md).
Forms retain values on errors; repeated operations are suppressed while in flight.
A trust override never replaces Validate/Revalidate and cannot revive invalidated evidence.
