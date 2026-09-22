from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.pretty import Pretty
from textual.widgets import (
    Button,
    DataTable,
    Static,
)

from llm_rio import admin_client as client_api
from llm_rio.ui.components import (
    FieldSpec,
    FormModal,
    _bool_value,
    _int_value,
    _optional_str,
    _str_value,
)

if TYPE_CHECKING:
    from llm_rio.tui import RioTui

FormResult = dict[str, str | bool]


class ProfilesController:
    def __init__(self, app: RioTui) -> None:
        self.app = app

    def _show_profile_details(self, index: int) -> None:
        if not 0 <= index < len(self.app.profile_records):
            return
        record = self.app.profile_records[index]
        self.app.query_one("#profile-details", Static).update(
            Panel(Pretty(record, expand_all=True), title=str(record.get("id") or "Profile"))
        )
        self.app._sync_profile_verification_buttons(record)

    def _sync_profile_verification_buttons(self, profile: dict[str, Any]) -> None:
        self.app.query_one("#profiles-activation", Button).label = (
            "Disable selected" if profile.get("active") else "Enable selected"
        )

    def _open_trust_measurements(self, profile: dict[str, Any]) -> None:
        sources = self.app.profile_payload.get("saved_measurements") or self.app.profile_records
        if not sources:
            self.app.notify(
                "No saved evidence exists. Run Validate/Revalidate first.", severity="warning"
            )
            return
        options = tuple(
            (
                f"{source['id']} ({source.get('serving_mode')}, "
                f"{source.get('machine_fingerprint')})",
                str(source["id"]),
            )
            for source in sources
        )

        def submitted(values: FormResult | None) -> None:
            if values is not None:
                self.app.run_worker(
                    self.app._trust_measurements(
                        {"id": _str_value(values, "profile")}, _str_value(values, "reason")
                    ),
                    exit_on_error=False,
                )

        self.app.show_form(
            FormModal(
                "Advanced: trust saved measurements (no probes)",
                [
                    FieldSpec(
                        "profile",
                        "Source measurements",
                        value=str(profile.get("id") or sources[0]["id"]),
                        options=options,
                    ),
                    FieldSpec("reason", "Reason for trusting this profile", required=True),
                ],
                "Trust measurements",
            ),
            submitted,
        )

    async def _trust_measurements(self, profile: dict[str, Any], reason: str) -> None:
        if self.app.profile_model is None:
            return
        model_id = self.app.profile_model["id"]
        ok, _ = await self.app._call(
            "Trusting saved measurements",
            lambda: client_api.request(
                "POST",
                f"/admin/models/{model_id}/profiles/{profile['id']}/trust",
                json_body={"reason": reason},
            ),
        )
        if ok:
            await self.app.refresh_profiles()

    def _selected_profile(self) -> dict[str, Any] | None:
        index = self.app.query_one("#profiles-table", DataTable).cursor_row
        if 0 <= index < len(self.app.profile_records):
            return self.app.profile_records[index]
        self.app.notify("Select a placement profile first.", severity="warning")
        return None

    def _open_clone_model_profile(self, record: dict[str, Any]) -> None:
        defaults = record.get("request_defaults")
        stored_defaults = defaults if isinstance(defaults, dict) else {}
        limits = record.get("request_limits")
        stored_limits = limits if isinstance(limits, dict) else {}
        native_context = stored_limits.get("max_context_tokens")
        fields = (
            FieldSpec(
                "nickname",
                "New model nickname",
                value=f"{record.get('nickname')}-clone",
                required=True,
            ),
            FieldSpec(
                "temperature",
                "Default temperature (blank inherits source)",
                value=str(stored_defaults.get("temperature", "")),
                input_type="number",
            ),
            FieldSpec(
                "top_p",
                "Default top-p (blank inherits source)",
                value=str(stored_defaults.get("top_p", "")),
                input_type="number",
            ),
            FieldSpec(
                "top_k",
                "Default top-k (blank inherits source)",
                value=str(stored_defaults.get("top_k", "")),
                input_type="integer",
            ),
            FieldSpec(
                "min_p",
                "Default min-p (blank inherits source)",
                value=str(stored_defaults.get("min_p", "")),
                input_type="number",
            ),
            FieldSpec(
                "presence_penalty",
                "Default presence penalty (blank inherits source)",
                value=str(stored_defaults.get("presence_penalty", "")),
                input_type="number",
            ),
            FieldSpec(
                "repetition_penalty",
                "Default repetition penalty (blank inherits source)",
                value=str(stored_defaults.get("repetition_penalty", "")),
                input_type="number",
            ),
            FieldSpec(
                "reasoning_effort",
                "Default reasoning effort",
                value=str(stored_defaults.get("reasoning_effort", "default")),
                options=(
                    ("Inherit source", "default"),
                    ("None", "none"),
                    ("Minimal", "minimal"),
                    ("Low", "low"),
                    ("Medium", "medium"),
                    ("High", "high"),
                    ("Extra high", "xhigh"),
                    ("Maximum", "max"),
                ),
            ),
            FieldSpec(
                "max_model_len",
                "Maximum context tokens (blank preserves source profiles)",
                input_type="integer",
                help_text=f"Source catalog limit: {native_context or 'unknown'} tokens.",
            ),
            FieldSpec(
                "yarn_factor",
                "YaRN factor (blank disables YaRN)",
                input_type="number",
                help_text="For 262,144 → 1,048,576, use factor 4.",
            ),
            FieldSpec(
                "yarn_original_max_model_len",
                "YaRN original context (blank reads model config)",
                input_type="integer",
            ),
            FieldSpec("inherit_grants", "Inherit source model access grants", value=True),
        )

        def finished(values: FormResult | None) -> None:
            if values is not None:
                self.app.run_worker(
                    self.app._clone_model_profile(record, values), exit_on_error=False
                )

        self.app.show_form(FormModal("Clone model", fields, "Clone"), finished)

    async def _clone_model_profile(self, source: dict[str, Any], values: FormResult) -> None:
        try:
            payload: dict[str, Any] = {
                "nickname": _str_value(values, "nickname"),
                "inherit_grants": _bool_value(values, "inherit_grants"),
            }
            for key, label, minimum, maximum in (
                ("temperature", "Temperature", 0.0, 2.0),
                ("top_p", "Top-p", 0.0, 1.0),
                ("min_p", "Min-p", 0.0, 1.0),
                ("presence_penalty", "Presence penalty", -2.0, 2.0),
                ("repetition_penalty", "Repetition penalty", 0.0, None),
                ("yarn_factor", "YaRN factor", 1.0, None),
            ):
                raw_value = _optional_str(values, key)
                if raw_value is None:
                    continue
                value = float(raw_value)
                if value < minimum or (
                    key in {"top_p", "yarn_factor", "repetition_penalty"} and value == minimum
                ):
                    raise ValueError(f"{label} must be greater than {minimum}.")
                if maximum is not None and value > maximum:
                    raise ValueError(f"{label} must be no more than {maximum}.")
                payload[key] = value
            for key, label in (
                ("top_k", "Top-k"),
                ("max_model_len", "Maximum context tokens"),
                ("yarn_original_max_model_len", "YaRN original context"),
            ):
                if _optional_str(values, key) is not None:
                    minimum_value = 0 if key == "top_k" else 1
                    payload[key] = _int_value(values, key, label, minimum=minimum_value)
            reasoning_effort = _optional_str(values, "reasoning_effort")
            if reasoning_effort not in {None, "default"}:
                payload["reasoning_effort"] = reasoning_effort
        except ValueError as exc:
            self.app.restore_form()
            self.app.notify(str(exc), severity="error")
            return
        ok, result = await self.app._call(
            "Cloning model",
            lambda: client_api.request(
                "POST", f"/admin/models/{source['id']}/clone", json_body=payload
            ),
        )
        if ok and isinstance(result, dict):
            model = result.get("model")
            nickname = model.get("nickname") if isinstance(model, dict) else payload["nickname"]
            self.app.notify(f"Created logical model {nickname} with shared weights.", timeout=10)
            await self.app.refresh_models()

    def _open_profile_edit(self, profile: dict[str, Any]) -> None:
        launch_args = profile.get("launch_args")
        launch = launch_args if isinstance(launch_args, dict) else {}
        gguf_files = self.app.profile_payload.get("available_gguf_files")
        gguf_help = ""
        if isinstance(gguf_files, list) and gguf_files:
            gguf_help = "Available: " + ", ".join(str(item) for item in gguf_files)
        fields = (
            FieldSpec(
                "engine",
                "Engine",
                value=str(profile.get("engine") or "vllm"),
                options=tuple(
                    (engine, engine) for engine in self.app.capabilities.get("engines", ["vllm"])
                ),
                help_text="Available engines are determined by the running mode.",
            ),
            FieldSpec(
                "tp",
                "Tensor-parallel GPU count",
                value=str(profile.get("tensor_parallel_size") or 1),
                required=True,
                input_type="integer",
            ),
            FieldSpec(
                "max_model_len",
                "Maximum context tokens",
                value=str(profile.get("max_model_len") or 4096),
                required=True,
                input_type="integer",
            ),
            FieldSpec(
                "max_num_seqs",
                "Maximum concurrent sequences (blank keeps engine default)",
                value="" if profile.get("max_num_seqs") is None else str(profile["max_num_seqs"]),
                input_type="integer",
            ),
            FieldSpec(
                "max_num_batched_tokens",
                "Maximum batched tokens (blank keeps engine default)",
                value=""
                if profile.get("max_num_batched_tokens") is None
                else str(profile["max_num_batched_tokens"]),
                input_type="integer",
            ),
            FieldSpec(
                "gpu_memory_utilization",
                "GPU memory utilization (0 < value <= 1)",
                value=str(profile.get("gpu_memory_utilization") or 0.9),
                required=True,
                input_type="number",
            ),
            FieldSpec("gguf_file", "GGUF file relative to model artifact", help_text=gguf_help),
            FieldSpec(
                "n_gpu_layers",
                "llama.cpp GPU layers to offload",
                value=str(launch.get("n_gpu_layers") or 99),
                input_type="integer",
            ),
            FieldSpec("make_default", "Make this the only active/default profile", value=False),
            FieldSpec(
                "restart_workers", "Drain current workers (verification still required)", value=True
            ),
        )

        def finished(values: FormResult | None) -> None:
            if values is not None:
                self.app.run_worker(self.app._edit_profile(profile, values), exit_on_error=False)

        self.app.show_form(
            FormModal("Override placement profile", fields, "Apply override"), finished
        )

    async def _edit_profile(self, profile: dict[str, Any], values: FormResult) -> None:
        if self.app.profile_model is None:
            return
        try:
            utilization = float(_str_value(values, "gpu_memory_utilization"))
            if not 0 < utilization <= 1:
                raise ValueError(
                    "GPU memory utilization must be greater than 0 and no more than 1."
                )
            payload: dict[str, Any] = {
                "engine": _str_value(values, "engine"),
                "tensor_parallel_size": _int_value(values, "tp", "Tensor parallel size", minimum=1),
                "max_model_len": _int_value(
                    values, "max_model_len", "Maximum context tokens", minimum=1
                ),
                "gpu_memory_utilization": utilization,
                "make_default": _bool_value(values, "make_default"),
                "restart_workers": _bool_value(values, "restart_workers"),
            }
            for key, label in (
                ("max_num_seqs", "Maximum concurrent sequences"),
                ("max_num_batched_tokens", "Maximum batched tokens"),
            ):
                if _optional_str(values, key) is not None:
                    payload[key] = _int_value(values, key, label, minimum=1)
            gguf_file = _optional_str(values, "gguf_file")
            if gguf_file is not None:
                payload["gguf_file"] = gguf_file
            if payload["engine"] == "llama.cpp":
                payload["n_gpu_layers"] = _int_value(
                    values, "n_gpu_layers", "GPU layers", minimum=0
                )
        except ValueError as exc:
            self.app.restore_form()
            self.app.notify(str(exc), severity="error")
            return
        model_id = self.app.profile_model["id"]
        ok, result = await self.app._call(
            "Updating placement profile",
            lambda: client_api.request(
                "PATCH", f"/admin/models/{model_id}/profiles/{profile['id']}", json_body=payload
            ),
        )
        if ok:
            self.app.notify("Placement profile updated.", timeout=8)
            if isinstance(result, dict) and result.get("verification_required"):
                self.app.notify(
                    "Verification required for: "
                    + ", ".join(result["verification_required"])
                    + (
                        ". Editing launch settings invalidates measurements. "
                        "Use Validate/Revalidate to run probes with the desired limits."
                    ),
                    severity="warning",
                    timeout=20,
                )
            if isinstance(result, dict) and result.get("drained_worker_ids"):
                self.app.notify(
                    "Draining workers: " + ", ".join(result["drained_worker_ids"]), timeout=10
                )
            await self.app.refresh_profiles()

    async def _set_profile_active(self, profile: dict[str, Any], active: bool) -> None:
        if self.app.profile_model is None:
            return
        model_id = self.app.profile_model["id"]
        action = "enable" if active else "disable"
        ok, result = await self.app._call(
            f"{action.title()} placement profile",
            lambda: client_api.request(
                "POST", f"/admin/models/{model_id}/profiles/{profile['id']}/{action}"
            ),
        )
        if ok:
            self.app.notify(f"Placement profile {action}d.", timeout=8)
            if isinstance(result, dict) and result.get("drained_worker_ids"):
                self.app.notify(
                    "Draining workers: " + ", ".join(result["drained_worker_ids"]), timeout=10
                )
            await self.app.refresh_profiles()
