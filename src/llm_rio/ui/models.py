from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from rich.panel import Panel
from rich.pretty import Pretty
from textual.widgets import (
    DataTable,
    Static,
)

from llm_rio import admin_client as client_api
from llm_rio.ui.components import (
    FieldSpec,
    FormModal,
    InputKind,
    _bool_value,
    _csv_value,
    _int_value,
    _optional_str,
    _revalidation_overrides,
    _str_value,
)

if TYPE_CHECKING:
    from llm_rio.tui import RioTui

FormResult = dict[str, str | bool]


class ModelsController:
    def __init__(self, app: RioTui) -> None:
        self.app = app

    async def refresh_models(self, *, notify_error: bool = True) -> None:
        ok, records = await self.app._call(
            "Loading models", client_api.model_records, notify_error=notify_error
        )
        if not ok or records is None:
            return
        self.app.model_records = records
        table = self.app.query_one("#models-table", DataTable)
        selected_key = self.app.table_selection(table)
        table.clear(columns=False)
        for record in records:
            job = record.get("registration_job")
            job_record = job if isinstance(job, dict) else {}
            table.add_row(
                str(record.get("nickname") or ""),
                str(record.get("state") or ""),
                str(record.get("huggingface_repo") or ""),
                str(job_record.get("state") or "—"),
                str(job_record.get("stage") or "—"),
                key=str(record.get("id") or record.get("nickname")),
            )
        self.app.restore_selection(table, selected_key)
        if records:
            self.app._show_model_details(table.cursor_row)
        else:
            self.app.query_one("#model-details", Static).update("No models found.")
        self.app._update_dashboard()

    def _show_model_details(self, index: int) -> None:
        if not 0 <= index < len(self.app.model_records):
            return
        record = self.app.model_records[index]
        self.app.query_one("#model-details", Static).update(
            Panel(Pretty(record, expand_all=True), title=str(record.get("nickname") or "Model"))
        )

    def _selected_model(self) -> dict[str, Any] | None:
        index = self.app.query_one("#models-table", DataTable).cursor_row
        if 0 <= index < len(self.app.model_records):
            return self.app.model_records[index]
        self.app.notify("Select a model first.", severity="warning")
        return None

    def _open_user_model_access(self, record: dict[str, Any]) -> None:
        models = sorted(self.app.model_records, key=lambda model: str(model.get("nickname") or ""))
        if not models:
            self.app.notify("No models are available.", severity="warning")
            return
        granted_models = {str(name) for name in record.get("granted_models") or []}
        fields = tuple(
            FieldSpec(
                f"model-{model['id']}",
                f"{model.get('nickname')} ({model.get('state') or 'unknown'})",
                value=str(model.get("nickname") or "") in granted_models,
            )
            for model in models
        )

        def finished(values: FormResult | None) -> None:
            if values is None:
                return
            selected_model_ids = [
                str(model["id"]) for model in models if _bool_value(values, f"model-{model['id']}")
            ]
            self.app.run_worker(
                self.app._replace_user_model_access(record, selected_model_ids), exit_on_error=False
            )

        self.app.show_form(
            FormModal(f"Change model access for {record.get('nickname')}", fields, "Save"), finished
        )

    async def _replace_user_model_access(
        self, record: dict[str, Any], model_ids: list[str]
    ) -> None:
        ok, _ = await self.app._call(
            "Saving model access",
            lambda: client_api.request(
                "PUT",
                f"/staff/keys/{record['id']}/model-grants",
                json_body={"model_ids": model_ids},
            ),
        )
        if ok:
            selected = set(model_ids)
            record["granted_models"] = [
                str(model.get("nickname") or "")
                for model in self.app.model_records
                if str(model.get("id") or "") in selected
            ]
            self.app.notify(f"Model access updated for {record.get('nickname')}.")
            await self.app.refresh_keys()

    def _open_add_model(self) -> None:
        fields = (
            FieldSpec("nickname", "Nickname", required=True),
            FieldSpec(
                "repository",
                "Hugging Face repository",
                placeholder="organization/model",
                required=False,
            ),
            FieldSpec("local_path", "Or absolute local artifact path"),
            FieldSpec(
                "engine",
                "Engine",
                value="vllm",
                options=tuple(
                    (engine, engine) for engine in self.app.capabilities.get("engines", ["vllm"])
                ),
                help_text="llama.cpp requires queue mode and engines.enable_llama_cpp=true.",
            ),
            FieldSpec("revision", "Revision (optional)"),
            FieldSpec("grants", "API key nicknames to grant (comma-separated)"),
        )
        self.app.show_form(
            FormModal("Register model", fields, "Register"), self.app._add_model_result
        )

    def _add_model_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.app.run_worker(self.app._add_model(values), exit_on_error=False)

    async def _add_model(self, values: FormResult) -> None:
        payload = {
            "nickname": _str_value(values, "nickname"),
            "huggingface_repo": _optional_str(values, "repository"),
            "local_path": _optional_str(values, "local_path"),
            "engine": _str_value(values, "engine") or "vllm",
            "revision": _optional_str(values, "revision"),
            "grant_to_keys": _csv_value(values, "grants"),
        }
        ok, result = await self.app._call(
            "Starting model registration",
            lambda: client_api.request("POST", "/staff/models", json_body=payload),
        )
        if ok and isinstance(result, dict):
            self.app.notify(
                f"Registration job: {result.get('job_id')}",
                title=f"Registration started for {_str_value(values, 'nickname')}",
                timeout=10,
            )
            await self.app.refresh_models()

    def _open_edit_model(self, record: dict[str, Any]) -> None:
        defaults = record.get("request_defaults")
        stored_defaults = defaults if isinstance(defaults, dict) else {}
        numeric_fields: tuple[tuple[str, str, Literal["integer", "number"]], ...] = (
            ("temperature", "Default temperature (blank clears)", "number"),
            ("top_p", "Default top-p (blank clears)", "number"),
            ("top_k", "Default top-k (blank clears)", "integer"),
            ("min_p", "Default min-p (blank clears)", "number"),
            ("presence_penalty", "Default presence penalty (blank clears)", "number"),
            ("repetition_penalty", "Default repetition penalty (blank clears)", "number"),
        )
        fields = tuple(
            (
                FieldSpec(
                    key, label, value=str(stored_defaults.get(key, "")), input_type=input_type
                )
                for key, label, input_type in numeric_fields
            )
        ) + (
            FieldSpec(
                "reasoning_effort",
                "Default reasoning effort",
                value=str(stored_defaults.get("reasoning_effort", "default")),
                options=(
                    ("No default", "default"),
                    ("None", "none"),
                    ("Minimal", "minimal"),
                    ("Low", "low"),
                    ("Medium", "medium"),
                    ("High", "high"),
                    ("Extra high", "xhigh"),
                    ("Maximum", "max"),
                ),
            ),
        )

        def finished(values: FormResult | None) -> None:
            if values is not None:
                self.app.run_worker(self.app._edit_model(record, values), exit_on_error=False)

        self.app.show_form(FormModal("Edit model defaults", fields, "Save defaults"), finished)

    async def _edit_model(self, record: dict[str, Any], values: FormResult) -> None:
        try:
            payload: dict[str, Any] = {}
            for key, label, minimum, maximum, strict_minimum in (
                ("temperature", "Temperature", 0.0, 2.0, False),
                ("top_p", "Top-p", 0.0, 1.0, True),
                ("min_p", "Min-p", 0.0, 1.0, False),
                ("presence_penalty", "Presence penalty", -2.0, 2.0, False),
                ("repetition_penalty", "Repetition penalty", 0.0, None, True),
            ):
                raw_value = _optional_str(values, key)
                if raw_value is None:
                    payload[key] = None
                    continue
                value = float(raw_value)
                if value < minimum or (strict_minimum and value == minimum):
                    raise ValueError(f"{label} must be greater than {minimum}.")
                if maximum is not None and value > maximum:
                    raise ValueError(f"{label} must be no more than {maximum}.")
                payload[key] = value
            raw_top_k = _optional_str(values, "top_k")
            payload["top_k"] = (
                None if raw_top_k is None else _int_value(values, "top_k", "Top-k", minimum=0)
            )
            reasoning_effort = _optional_str(values, "reasoning_effort")
            payload["reasoning_effort"] = (
                None if reasoning_effort in {None, "default"} else reasoning_effort
            )
        except ValueError as exc:
            self.app.restore_form()
            self.app.notify(str(exc), severity="error")
            return
        ok, _ = await self.app._call(
            "Updating model defaults",
            lambda: client_api.request("PATCH", f"/admin/models/{record['id']}", json_body=payload),
        )
        if ok:
            self.app.notify(f"Updated defaults for {record.get('nickname')}", timeout=8)
            await self.app.refresh_models()

    def _model_job_id(self, record: dict[str, Any]) -> str | None:
        job = record.get("registration_job")
        if isinstance(job, dict) and isinstance(job.get("id"), str):
            return cast(str, job["id"])
        self.app.notify(f"{record.get('nickname')} has no registration job.", severity="warning")
        return None

    async def _review_model(self, record: dict[str, Any]) -> None:
        job_id = self.app._model_job_id(record)
        if job_id is None:
            return
        ok, job = await self.app._call(
            "Loading registration job",
            lambda: client_api.request("GET", f"/staff/model-jobs/{job_id}"),
        )
        if ok:
            self.app.query_one("#model-details", Static).update(
                Panel(Pretty(job, expand_all=True), title="Registration job")
            )

    async def _open_revalidation(
        self, record: dict[str, Any], profile: dict[str, Any] | None = None
    ) -> None:
        job_id = self.app._model_job_id(record)
        if job_id is None:
            return
        if profile is not None and profile.get("engine") != "vllm":
            self.app.notify("Revalidation currently supports vLLM profiles.", severity="warning")
            return
        ok, job = await self.app._call(
            "Loading validation settings",
            lambda: client_api.request("GET", f"/staff/model-jobs/{job_id}"),
        )
        if not ok or not isinstance(job, dict):
            return
        if job.get("state") in {"QUEUED", "RUNNING"}:
            self.app.notify("This model already has a validation job running.", severity="warning")
            return
        defaults = dict(job.get("validation_overrides") or {})
        if profile is not None:
            defaults = {
                key: profile.get(key)
                for key in (
                    "max_model_len",
                    "gpu_memory_utilization",
                    "tensor_parallel_size",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                )
            }
            args = dict(profile.get("launch_args") or {})
            args.pop("enable_sleep_mode", None)
            for key in ("dtype", "quantization"):
                if profile.get(key) is not None:
                    args[key] = profile[key]
            defaults["launch_args"] = args
        numeric = (
            ("max_model_len", "Maximum context tokens (blank: automatic)", "integer"),
            ("gpu_memory_utilization", "GPU utilization (0–1, blank: automatic)", "number"),
            ("tensor_parallel_size", "Tensor parallelism / TP (blank: automatic)", "integer"),
            ("max_num_seqs", "Maximum concurrent sequences (optional)", "integer"),
            ("max_num_batched_tokens", "Maximum batched tokens (optional)", "integer"),
        )
        fields = tuple(
            (
                FieldSpec(
                    key,
                    label,
                    value="" if defaults.get(key) is None else str(defaults[key]),
                    input_type=cast(InputKind, kind),
                    help_text=(
                        "Runs real validation. Native validation waits for maintenance; "
                        "use Drain on the Maintenance page. Success activates the measured"
                        " profiles and keeps previous profiles inactive."
                    )
                    if key == "max_model_len"
                    else "",
                )
                for key, label, kind in numeric
            )
        ) + (
            FieldSpec(
                "launch_args",
                "Extra vLLM arguments (JSON object)",
                multiline=True,
                value=json.dumps(defaults.get("launch_args") or {}, indent=2),
                help_text=(
                    'Example: {"kv_cache_dtype": "fp8", "enforce_eager": true}. '
                    "Successful validation saves the effective settings for later "
                    "model loads."
                ),
            ),
        )

        def finished(values: FormResult | None) -> None:
            if values is not None:
                self.app.run_worker(
                    self.app._retry_model(record, _revalidation_overrides(values)),
                    exit_on_error=False,
                )

        self.app.show_form(
            FormModal(
                f"Re-validate {record.get('nickname')}",
                fields,
                "Queue validation",
                validate=_revalidation_overrides,
            ),
            finished,
        )

    async def _retry_model(self, record: dict[str, Any], overrides: dict[str, Any]) -> None:
        job_id = self.app._model_job_id(record)
        if job_id is None:
            return
        ok, result = await self.app._call(
            "Queueing validation",
            lambda: client_api.request(
                "POST",
                f"/staff/model-jobs/{job_id}/retry",
                json_body={"validation_overrides": overrides},
            ),
        )
        if ok:
            self.app.notify(f"Validation queued: {result}", timeout=10)
            await self.app.refresh_models()

    async def _disable_model(self, record: dict[str, Any]) -> None:
        ok, _ = await self.app._call(
            "Disabling model",
            lambda: client_api.request("POST", f"/staff/models/{record['id']}/disable"),
        )
        if ok:
            self.app.notify(f"Disabled {record.get('nickname')}.")
            await self.app.refresh_models()

    def _open_model_user_access(self, model: dict[str, Any]) -> None:
        users = sorted(self.app.key_records, key=lambda user: str(user.get("nickname") or ""))
        if not users:
            self.app.notify("No users are available.", severity="warning")
            return
        model_nickname = str(model.get("nickname") or "")
        fields = tuple(
            FieldSpec(
                f"user-{user['id']}",
                (
                    f"{user.get('nickname')}"
                    " ("
                    f"{user.get('role') or 'user'}"
                    ", "
                    f"{('active' if user.get('active') else 'revoked')}"
                    ")"
                ),
                value=model_nickname in {str(name) for name in user.get("granted_models") or []},
            )
            for user in users
        )

        def finished(values: FormResult | None) -> None:
            if values is None:
                return
            selected_user_ids = [
                str(user["id"]) for user in users if _bool_value(values, f"user-{user['id']}")
            ]
            self.app.run_worker(
                self.app._replace_model_user_access(model, selected_user_ids), exit_on_error=False
            )

        self.app.show_form(
            FormModal(f"Change user access for {model.get('nickname')}", fields, "Save"), finished
        )

    async def _replace_model_user_access(
        self, model: dict[str, Any], selected_user_ids: list[str]
    ) -> None:
        model_nickname = str(model.get("nickname") or "")
        model_ids_by_name = {
            str(candidate.get("nickname") or ""): str(candidate.get("id") or "")
            for candidate in self.app.model_records
            if candidate.get("nickname") and candidate.get("id")
        }
        selected_users = set(selected_user_ids)
        updates: list[tuple[str, list[str]]] = []
        for user in self.app.key_records:
            user_id = str(user.get("id") or "")
            grants = {str(name) for name in user.get("granted_models") or []}
            has_access = model_nickname in grants
            should_have_access = user_id in selected_users
            if has_access == should_have_access:
                continue
            if should_have_access:
                grants.add(model_nickname)
            else:
                grants.discard(model_nickname)
            missing_models = sorted(grants - model_ids_by_name.keys())
            if missing_models:
                self.app.notify("Refresh models before changing user access.", severity="warning")
                return
            updates.append((user_id, [model_ids_by_name[nickname] for nickname in sorted(grants)]))
        if not updates:
            self.app.notify("User access already matches the selection.")
            return

        def persist() -> int:
            for user_id, model_ids in updates:
                client_api.request(
                    "PUT", f"/staff/keys/{user_id}/model-grants", json_body={"model_ids": model_ids}
                )
            return len(updates)

        ok, updated_count = await self.app._call("Saving user access", persist)
        if ok:
            self.app.notify(f"Updated user access for {updated_count or 0} users.")
            await self.app.refresh_keys()
