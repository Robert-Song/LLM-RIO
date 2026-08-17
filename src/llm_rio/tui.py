from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from rich.panel import Panel
from rich.pretty import Pretty
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)
from textual.worker import WorkerCancelled, WorkerFailed

from llm_rio import cli as cli_api
from llm_rio.config import Settings
from llm_rio.inventory import InventoryError, discover_inventory

FormValue = str | bool
FormResult = dict[str, FormValue]
InputKind = Literal["text", "integer", "number"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    key: str
    label: str
    value: str | bool = ""
    placeholder: str = ""
    required: bool = False
    password: bool = False
    input_type: InputKind = "text"
    options: tuple[tuple[str, str], ...] = ()
    help_text: str = ""


class FormModal(ModalScreen[FormResult | None]):
    """A small reusable form used for all management operations."""

    DEFAULT_CSS = """
    FormModal {
        align: center middle;
        background: $background 65%;
    }

    FormModal > Vertical {
        width: 72;
        max-width: 94%;
        height: auto;
        max-height: 92%;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    FormModal .modal-title {
        width: 100%;
        text-style: bold;
        color: $text-accent;
        margin-bottom: 1;
    }

    FormModal ScrollableContainer {
        height: 1fr;
        min-height: 1;
    }

    FormModal Label {
        margin-top: 1;
    }

    FormModal .field-help {
        color: $text-muted;
        margin: 0 0 0 1;
    }

    FormModal .modal-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }

    FormModal .modal-actions Button {
        margin-left: 1;
        min-width: 12;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, fields: Iterable[FieldSpec], submit_label: str) -> None:
        super().__init__()
        self.form_title = title
        self.fields = tuple(fields)
        self.submit_label = submit_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.form_title, classes="modal-title")
            with ScrollableContainer():
                for field in self.fields:
                    if field.options:
                        yield Label(field.label)
                        yield Select(
                            field.options,
                            value=str(field.value),
                            allow_blank=False,
                            id=f"field-{field.key}",
                        )
                    elif isinstance(field.value, bool):
                        yield Checkbox(field.label, value=field.value, id=f"field-{field.key}")
                    else:
                        yield Label(field.label)
                        yield Input(
                            value=field.value,
                            placeholder=field.placeholder,
                            password=field.password,
                            type=field.input_type,
                            id=f"field-{field.key}",
                        )
                    if field.help_text:
                        yield Static(field.help_text, classes="field-help")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="form-cancel")
                yield Button(self.submit_label, id="form-submit", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form-cancel":
            self.dismiss(None)
            return
        if event.button.id != "form-submit":
            return

        values: FormResult = {}
        for field in self.fields:
            value: FormValue
            selector = f"#field-{field.key}"
            if field.options:
                raw = self.query_one(selector, Select).value
                value = "" if raw is Select.NULL else str(raw)
            elif isinstance(field.value, bool):
                value = self.query_one(selector, Checkbox).value
            else:
                value = self.query_one(selector, Input).value.strip()
            if field.required and isinstance(value, str) and not value:
                self.notify(f"{field.label} is required.", severity="error")
                return
            values[field.key] = value
        self.dismiss(values)


class ConfirmModal(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
        background: $background 65%;
    }

    ConfirmModal > Vertical {
        width: 64;
        max-width: 92%;
        height: auto;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    ConfirmModal .confirm-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    ConfirmModal Horizontal {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }

    ConfirmModal Button {
        margin-left: 1;
        min-width: 12;
    }
    """

    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(self, title: str, message: str, confirm_label: str) -> None:
        super().__init__()
        self.confirm_title = title
        self.message = message
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.confirm_title, classes="confirm-title")
            yield Static(self.message)
            with Horizontal():
                yield Button("Cancel", id="confirm-cancel")
                yield Button(self.confirm_label, id="confirm-submit", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-submit")


class RioTui(App[Path | None]):
    """Full-screen administration console for LLM-RIO."""

    TITLE = "LLM-RIO Control Center"
    SUB_TITLE = ""

    CSS = """
    Screen {
        background: $background;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 22;
        min-width: 18;
        padding: 0 1;
        background: $panel;
        border-right: solid $primary-background;
    }

    #brand {
        height: 2;
        content-align: center middle;
        text-style: bold;
        color: $text-accent;
        border-bottom: solid $primary-background;
        margin: 0;
    }

    #sidebar Button {
        width: 100%;
        height: 3;
        margin: 0;
        content-align: left middle;
    }

    #content {
        width: 1fr;
        height: 1fr;
    }

    .page {
        padding: 1 2 2 2;
    }

    .page-title {
        height: 2;
        text-style: bold;
        color: $text-accent;
    }

    .page-description {
        color: $text-muted;
        margin-bottom: 1;
    }

    .toolbar {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto;
        grid-gutter: 1 1;
        height: auto;
        margin-bottom: 1;
    }

    .toolbar Button {
        width: 100%;
        height: 3;
        margin: 0;
        content-align: center middle;
    }

    DataTable {
        height: 16;
        border: round $primary-background;
    }

    .details {
        min-height: 10;
        height: auto;
        margin-top: 1;
        padding: 0 1;
        border: round $primary-background;
        background: $surface;
    }

    #dashboard-summary {
        height: auto;
        min-height: 12;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    #maintenance-output, #diagnostics-output, #service-output {
        height: auto;
        min-height: 14;
        padding: 1;
        border: round $primary-background;
        background: $surface;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }

    .danger {
        color: $error;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.key_records: list[dict[str, Any]] = []
        self.model_records: list[dict[str, Any]] = []
        self.profile_records: list[dict[str, Any]] = []
        self.profile_payload: dict[str, Any] = {}
        self.profile_model: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("LLM-RIO", id="brand")
                yield Button("Dashboard", id="nav-dashboard", variant="primary")
                yield Button("Users", id="nav-keys")
                yield Button("Models", id="nav-models")
                yield Button("Maintenance", id="nav-maintenance")
                yield Button("Diagnostics", id="nav-system")
                yield Button("Quit", id="nav-quit")
            with ContentSwitcher(initial="dashboard", id="content"):
                with VerticalScroll(id="dashboard", classes="page"):
                    yield Static("Dashboard", classes="page-title")
                    yield Static(
                        "At-a-glance service state. Press R to refresh the current page.",
                        classes="page-description",
                    )
                    with Grid(classes="toolbar"):
                        yield Button(
                            "Refresh everything", id="dashboard-refresh", variant="primary"
                        )
                        yield Button("Start service", id="dashboard-start-service")
                    yield Static("Connecting to the local control plane…", id="dashboard-summary")
                with VerticalScroll(id="keys", classes="page"):
                    yield Static("Users", classes="page-title")
                    yield Static(
                        "Create credentials, inspect usage, and manage quotas and access.",
                        classes="page-description",
                    )
                    with Grid(classes="toolbar"):
                        yield Button("Refresh", id="keys-refresh", variant="primary")
                        yield Button("Create", id="keys-create")
                        yield Button("Rotate", id="keys-rotate")
                        yield Button("Copy API key", id="keys-copy")
                        yield Button("Set quota", id="keys-limit")
                        yield Button("Reset usage", id="keys-reset")
                        yield Button("Revoke", id="keys-revoke", variant="warning")
                        yield Button("Restore", id="keys-restore", variant="primary")
                        yield Button("Delete", id="keys-delete", variant="error")
                    yield DataTable(zebra_stripes=True, cursor_type="row", id="keys-table")
                    yield Static(
                        "Select a key to see its full details.", id="key-details", classes="details"
                    )
                with VerticalScroll(id="models", classes="page"):
                    yield Static("Models", classes="page-title")
                    yield Static(
                        "Register models, review jobs, control access, and tune "
                        "placement profiles.",
                        classes="page-description",
                    )
                    with Grid(classes="toolbar"):
                        yield Button("Refresh", id="models-refresh", variant="primary")
                        yield Button("Add model", id="models-add")
                        yield Button("Clone profile", id="models-clone")
                        yield Button("Edit model", id="models-edit")
                        yield Button("Review job", id="models-review")
                        yield Button("Retry job", id="models-retry")
                        yield Button("Disable", id="models-disable", variant="warning")
                        yield Button("Show key access", id="models-key-access")
                        yield Button("Change access", id="models-access-update")
                        yield Button("Profiles", id="models-profiles")
                    yield DataTable(zebra_stripes=True, cursor_type="row", id="models-table")
                    yield Static(
                        "Select a model to see catalog and registration details.",
                        id="model-details",
                        classes="details",
                    )
                with VerticalScroll(id="profiles", classes="page"):
                    yield Static("Placement Profiles", classes="page-title")
                    yield Static(
                        "Active and inactive measured profiles.", id="profiles-description"
                    )
                    with Grid(classes="toolbar"):
                        yield Button("Back to models", id="profiles-back")
                        yield Button("Refresh", id="profiles-refresh", variant="primary")
                        yield Button("Edit selected", id="profiles-edit", variant="warning")
                        yield Button("Enable selected", id="profiles-enable")
                        yield Button("Disable selected", id="profiles-disable", variant="warning")
                    yield DataTable(zebra_stripes=True, cursor_type="row", id="profiles-table")
                    yield Static(
                        "Select a profile to see its launch settings.",
                        id="profile-details",
                        classes="details",
                    )
                with VerticalScroll(id="maintenance", classes="page"):
                    yield Static("Maintenance", classes="page-title")
                    yield Static(
                        "Drain the host safely before maintenance, or resume request scheduling.",
                        classes="page-description",
                    )
                    with Grid(classes="toolbar"):
                        yield Button("Refresh status", id="maintenance-refresh", variant="primary")
                        yield Button("Drain", id="maintenance-drain", variant="warning")
                        yield Button("Resume", id="maintenance-resume")
                    yield Static("Status has not been loaded.", id="maintenance-output")
                with VerticalScroll(id="system", classes="page"):
                    yield Static("Diagnostics & Service", classes="page-title")
                    yield Static(
                        "Inspect host prerequisites, connection settings, and launch "
                        "the API service.",
                        classes="page-description",
                    )
                    with Grid(classes="toolbar"):
                        yield Button("Run doctor", id="system-doctor", variant="primary")
                        yield Button("Refresh service info", id="system-info")
                        yield Button("Start service", id="system-start-service", variant="warning")
                    yield Static("Diagnostics have not been run.", id="diagnostics-output")
                    yield Static("Loading service information…", id="service-output")
        yield Static("Ready", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#keys-table", DataTable).add_columns(
            "Nickname", "Role", "Active", "Quota remaining", "Models"
        )
        self.query_one("#models-table", DataTable).add_columns(
            "Nickname", "State", "Repository", "Job", "Stage"
        )
        self.query_one("#profiles-table", DataTable).add_columns(
            "#", "Active", "Engine", "GPUs / TP", "Context", "Max sequences"
        )
        self.run_worker(self.refresh_all(initial=True), name="initial-refresh", exit_on_error=False)

    def _set_status(self, message: str) -> None:
        self.query_one("#status-bar", Static).update(message)

    async def _call(
        self,
        label: str,
        operation: Callable[[], T],
        *,
        notify_error: bool = True,
    ) -> tuple[bool, T | None]:
        self._set_status(f"{label}…")
        try:
            result = await self.run_worker(
                operation,
                name=label,
                thread=True,
                exit_on_error=False,
            ).wait()
        except WorkerCancelled:
            self._set_status(f"{label} cancelled")
            return False, None
        except WorkerFailed as exc:
            message = str(exc.error)
            self._set_status(f"{label} failed: {message}")
            if notify_error:
                self.notify(message, title=f"{label} failed", severity="error", timeout=8)
            return False, None
        self._set_status(f"{label} complete")
        return True, result

    def _navigate(self, page: str) -> None:
        self.query_one("#content", ContentSwitcher).current = page
        for button in self.query("#sidebar Button").results(Button):
            button.variant = "primary" if button.id == f"nav-{page}" else "default"

    def action_refresh(self) -> None:
        page = self.query_one("#content", ContentSwitcher).current
        if page == "keys":
            self.run_worker(self.refresh_keys(), exit_on_error=False)
        elif page == "models":
            self.run_worker(self.refresh_models(), exit_on_error=False)
        elif page == "profiles":
            self.run_worker(self.refresh_profiles(), exit_on_error=False)
        elif page == "maintenance":
            self.run_worker(self.refresh_maintenance(), exit_on_error=False)
        elif page == "system":
            self.run_worker(self.refresh_service_info(), exit_on_error=False)
        else:
            self.run_worker(self.refresh_all(), exit_on_error=False)

    async def refresh_all(self, *, initial: bool = False) -> None:
        await self.refresh_keys(notify_error=not initial)
        await self.refresh_models(notify_error=not initial)
        await self.refresh_maintenance(notify_error=not initial)
        await self.refresh_service_info(notify_error=not initial)
        self._update_dashboard()

    async def refresh_keys(self, *, notify_error: bool = True) -> None:
        ok, records = await self._call(
            "Loading API keys", cli_api._key_records, notify_error=notify_error
        )
        if not ok or records is None:
            return
        self.key_records = records
        table = self.query_one("#keys-table", DataTable)
        table.clear(columns=False)
        for record in records:
            if record.get("unlimited"):
                quota = "unlimited"
            else:
                quota = f"{int(record.get('balance_tokens') or 0):,}"
            models = record.get("granted_models") or []
            table.add_row(
                str(record.get("nickname") or ""),
                str(record.get("role") or ""),
                "yes" if record.get("active") else "no",
                quota,
                ", ".join(str(model) for model in models) or "(none)",
                key=str(record.get("id") or record.get("nickname")),
            )
        if records:
            self._show_key_details(0)
        else:
            self.query_one("#key-details", Static).update("No API keys found.")
        self._update_dashboard()

    async def refresh_models(self, *, notify_error: bool = True) -> None:
        ok, records = await self._call(
            "Loading models", cli_api._model_records, notify_error=notify_error
        )
        if not ok or records is None:
            return
        self.model_records = records
        table = self.query_one("#models-table", DataTable)
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
        if records:
            self._show_model_details(0)
        else:
            self.query_one("#model-details", Static).update("No models found.")
        self._update_dashboard()

    async def refresh_maintenance(self, *, notify_error: bool = True) -> None:
        ok, payload = await self._call(
            "Loading maintenance status",
            lambda: cli_api._request("GET", "/admin/maintenance"),
            notify_error=notify_error,
        )
        if ok:
            self.query_one("#maintenance-output", Static).update(
                Panel(Pretty(payload, expand_all=True), title="Maintenance status")
            )

    async def refresh_service_info(self, *, notify_error: bool = True) -> None:
        def load_info() -> dict[str, str]:
            config = Path(os.environ.get("LLMRIO_CONFIG", "config.toml"))
            result = {
                "API base URL": cli_api._base_url(),
                "Config file": str(config.resolve()),
                "Administrator credential": "unavailable",
            }
            try:
                cli_api._api_key()
            except Exception as exc:
                result["Administrator credential"] = str(exc)
            else:
                result["Administrator credential"] = "recovered from the protected local vault"
            return result

        ok, info = await self._call(
            "Loading service information", load_info, notify_error=notify_error
        )
        if ok and info is not None:
            self.query_one("#service-output", Static).update(
                Panel(Pretty(info, expand_all=True), title="Connection")
            )

    async def refresh_profiles(self, *, notify_error: bool = True) -> None:
        if self.profile_model is None:
            return
        nickname = str(self.profile_model.get("nickname") or "")
        ok, payload = await self._call(
            "Loading placement profiles",
            lambda: cli_api._model_profiles(nickname),
            notify_error=notify_error,
        )
        if not ok or payload is None:
            return
        self.profile_payload = payload
        raw_records = payload.get("data")
        self.profile_records = (
            [record for record in raw_records if isinstance(record, dict)]
            if isinstance(raw_records, list)
            else []
        )
        table = self.query_one("#profiles-table", DataTable)
        table.clear(columns=False)
        for number, profile in enumerate(self.profile_records, 1):
            table.add_row(
                str(number),
                "yes" if profile.get("active") else "no",
                str(profile.get("engine") or ""),
                f"{profile.get('gpu_count') or 0} / {profile.get('tensor_parallel_size') or 0}",
                f"{int(profile.get('max_model_len') or 0):,}",
                str(profile.get("max_num_seqs") or "engine default"),
                key=str(profile.get("id") or number),
            )
        gguf_files = payload.get("available_gguf_files")
        gguf_note = ""
        if isinstance(gguf_files, list) and gguf_files:
            gguf_note = "  •  GGUF: " + ", ".join(str(item) for item in gguf_files)
        self.query_one("#profiles-description", Static).update(f"Model: {nickname}{gguf_note}")
        if self.profile_records:
            self._show_profile_details(0)
        else:
            self.query_one("#profile-details", Static).update(
                "No placement profiles are recorded for this model."
            )

    def _update_dashboard(self) -> None:
        available = sum(1 for model in self.model_records if model.get("state") == "AVAILABLE")
        review = sum(
            1 for model in self.model_records if model.get("state") == "NEEDS_ADMIN_REVIEW"
        )
        active_keys = sum(1 for key in self.key_records if key.get("active"))
        summary = {
            "API URL": _safe_base_url(),
            "Users": f"{len(self.key_records)} total / {active_keys} active",
            "Models": f"{len(self.model_records)} total / {available} available",
            "Needs administrator review": review,
            "Keyboard": "R refreshes the current page; Q exits",
        }
        self.query_one("#dashboard-summary", Static).update(
            Panel(Pretty(summary, expand_all=True), title="Control plane")
        )

    def _show_key_details(self, index: int) -> None:
        if not 0 <= index < len(self.key_records):
            return
        record = self.key_records[index]
        self.query_one("#key-details", Static).update(
            Panel(Pretty(record, expand_all=True), title=str(record.get("nickname") or "API key"))
        )

    def _show_model_details(self, index: int) -> None:
        if not 0 <= index < len(self.model_records):
            return
        record = self.model_records[index]
        self.query_one("#model-details", Static).update(
            Panel(Pretty(record, expand_all=True), title=str(record.get("nickname") or "Model"))
        )

    def _show_profile_details(self, index: int) -> None:
        if not 0 <= index < len(self.profile_records):
            return
        record = self.profile_records[index]
        self.query_one("#profile-details", Static).update(
            Panel(Pretty(record, expand_all=True), title=str(record.get("id") or "Profile"))
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "keys-table":
            self._show_key_details(event.cursor_row)
        elif event.data_table.id == "models-table":
            self._show_model_details(event.cursor_row)
        elif event.data_table.id == "profiles-table":
            self._show_profile_details(event.cursor_row)

    def _selected_key(self) -> dict[str, Any] | None:
        index = self.query_one("#keys-table", DataTable).cursor_row
        if 0 <= index < len(self.key_records):
            return self.key_records[index]
        self.notify("Select an API key first.", severity="warning")
        return None

    def _copy_key_to_clipboard(self, record: dict[str, Any]) -> None:
        api_key = record.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            self.notify("The selected user's API key is unavailable.", severity="warning")
            return
        self.copy_to_clipboard(api_key)
        self.notify(f"Copied API key for {record.get('nickname')} to the clipboard.")

    def _selected_model(self) -> dict[str, Any] | None:
        index = self.query_one("#models-table", DataTable).cursor_row
        if 0 <= index < len(self.model_records):
            return self.model_records[index]
        self.notify("Select a model first.", severity="warning")
        return None

    def _selected_profile(self) -> dict[str, Any] | None:
        index = self.query_one("#profiles-table", DataTable).cursor_row
        if 0 <= index < len(self.profile_records):
            return self.profile_records[index]
        self.notify("Select a placement profile first.", severity="warning")
        return None

    def _confirm(
        self,
        title: str,
        message: str,
        confirm_label: str,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        def finished(confirmed: bool | None) -> None:
            if confirmed:
                self.run_worker(operation(), exit_on_error=False)

        self.push_screen(ConfirmModal(title, message, confirm_label), finished)

    def _open_create_key(self) -> None:
        fields = (
            FieldSpec("nickname", "Nickname", required=True),
            FieldSpec(
                "role",
                "Role",
                value="user",
                options=(("User", "user"), ("Teaching assistant", "ta"), ("Admin", "admin")),
            ),
            FieldSpec("unlimited", "Unlimited token quota", value=True),
            FieldSpec(
                "limit",
                "Lifetime token limit (used when quota is limited)",
                value="1000000",
                input_type="integer",
            ),
            FieldSpec("account_id", "Existing quota account ID (optional)"),
            FieldSpec("grants", "Model nicknames to grant (comma-separated)"),
            FieldSpec(
                "api_key",
                "Custom rio_ API key (optional)",
                password=True,
                help_text="Leave blank to generate a secure key automatically.",
            ),
        )
        self.push_screen(FormModal("Create API key", fields, "Create"), self._create_key_result)

    def _create_key_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.run_worker(self._create_key(values), exit_on_error=False)

    async def _create_key(self, values: FormResult) -> None:
        try:
            unlimited = _bool_value(values, "unlimited")
            limit = None if unlimited else _int_value(values, "limit", "Token limit", minimum=0)
            payload = {
                "nickname": _str_value(values, "nickname"),
                "role": _str_value(values, "role"),
                "limit_tokens": limit,
                "quota_account_id": _optional_str(values, "account_id"),
                "models": _csv_value(values, "grants"),
                "api_key": _optional_str(values, "api_key"),
            }
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        ok, result = await self._call(
            "Creating API key", lambda: cli_api._request("POST", "/admin/keys", json_body=payload)
        )
        if ok and isinstance(result, dict):
            secret = result.get("api_key")
            self.notify(
                f"Created {result.get('nickname')}. Full key: {secret}",
                title="API key created",
                timeout=15,
            )
            await self.refresh_keys()

    async def _rotate_key(self, record: dict[str, Any]) -> None:
        ok, result = await self._call(
            "Rotating API key",
            lambda: cli_api._request("POST", f"/admin/keys/{record['id']}/rotate"),
        )
        if ok and isinstance(result, dict):
            self.notify(
                f"New full key: {result.get('api_key')}",
                title=f"Rotated {record.get('nickname')}",
                timeout=15,
            )
            await self.refresh_keys()

    def _open_key_limit(self, record: dict[str, Any]) -> None:
        fields = (
            FieldSpec("unlimited", "Unlimited token quota", value=bool(record.get("unlimited"))),
            FieldSpec(
                "limit",
                "Lifetime token limit",
                value=str(record.get("limit_tokens") or 0),
                required=True,
                input_type="integer",
            ),
        )

        def finished(values: FormResult | None) -> None:
            if values is not None:
                self.run_worker(self._set_key_limit(record, values), exit_on_error=False)

        self.push_screen(
            FormModal(f"Set quota for {record.get('nickname')}", fields, "Update"), finished
        )

    async def _set_key_limit(self, record: dict[str, Any], values: FormResult) -> None:
        try:
            payload = {
                "limit_tokens": _int_value(values, "limit", "Token limit", minimum=0),
                "unlimited": _bool_value(values, "unlimited"),
            }
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        ok, _ = await self._call(
            "Updating quota",
            lambda: cli_api._request("PUT", f"/admin/keys/{record['id']}/quota", json_body=payload),
        )
        if ok:
            self.notify(f"Quota updated for {record.get('nickname')}.")
            await self.refresh_keys()

    async def _key_action(self, record: dict[str, Any], action: str) -> None:
        if action == "reset":
            method, suffix, label, success = (
                "POST",
                "/usage/reset",
                "Resetting usage",
                "Usage reset",
            )
        elif action == "revoke":
            method, suffix, label, success = (
                "POST",
                "/revoke",
                "Revoking API key",
                "API key revoked",
            )
        elif action == "restore":
            method, suffix, label, success = (
                "POST",
                "/restore",
                "Restoring API key",
                "API key restored",
            )
        else:
            method, suffix, label, success = "DELETE", "", "Deleting API key", "API key deleted"
        ok, _ = await self._call(
            label, lambda: cli_api._request(method, f"/admin/keys/{record['id']}{suffix}")
        )
        if ok:
            self.notify(f"{success} for {record.get('nickname')}.")
            await self.refresh_keys()

    def _open_add_model(self) -> None:
        fields = (
            FieldSpec("nickname", "Nickname", required=True),
            FieldSpec(
                "repository",
                "Hugging Face repository",
                placeholder="organization/model",
                required=True,
            ),
            FieldSpec("revision", "Revision (optional)"),
            FieldSpec("grants", "API key nicknames to grant (comma-separated)"),
        )
        self.push_screen(FormModal("Register model", fields, "Register"), self._add_model_result)

    def _add_model_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.run_worker(self._add_model(values), exit_on_error=False)

    async def _add_model(self, values: FormResult) -> None:
        payload = {
            "nickname": _str_value(values, "nickname"),
            "huggingface_repo": _str_value(values, "repository"),
            "revision": _optional_str(values, "revision"),
            "grant_to_keys": _csv_value(values, "grants"),
        }
        ok, result = await self._call(
            "Starting model registration",
            lambda: cli_api._request("POST", "/staff/models", json_body=payload),
        )
        if ok and isinstance(result, dict):
            self.notify(
                f"Registration job: {result.get('job_id')}",
                title=f"Registration started for {_str_value(values, 'nickname')}",
                timeout=10,
            )
            await self.refresh_models()

    def _open_edit_model(self, record: dict[str, Any]) -> None:
        defaults = record.get("request_defaults")
        stored_defaults = defaults if isinstance(defaults, dict) else {}
        numeric_fields = (
            ("temperature", "Default temperature (blank clears)", "number"),
            ("top_p", "Default top-p (blank clears)", "number"),
            ("top_k", "Default top-k (blank clears)", "integer"),
            ("min_p", "Default min-p (blank clears)", "number"),
            ("presence_penalty", "Default presence penalty (blank clears)", "number"),
            ("repetition_penalty", "Default repetition penalty (blank clears)", "number"),
        )
        fields = tuple(
            FieldSpec(
                key,
                label,
                value=str(stored_defaults.get(key, "")),
                input_type=input_type,
            )
            for key, label, input_type in numeric_fields
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
                self.run_worker(self._edit_model(record, values), exit_on_error=False)

        self.push_screen(FormModal("Edit model defaults", fields, "Save defaults"), finished)

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
                self.run_worker(self._clone_model_profile(record, values), exit_on_error=False)

        self.push_screen(FormModal("Clone model profile", fields, "Clone"), finished)

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
            self.notify(str(exc), severity="error")
            return
        ok, result = await self._call(
            "Cloning model profile",
            lambda: cli_api._request(
                "POST", f"/admin/models/{source['id']}/clone", json_body=payload
            ),
        )
        if ok and isinstance(result, dict):
            model = result.get("model")
            nickname = model.get("nickname") if isinstance(model, dict) else payload["nickname"]
            self.notify(f"Created logical model {nickname} with shared weights.", timeout=10)
            await self.refresh_models()

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
            self.notify(str(exc), severity="error")
            return
        ok, _ = await self._call(
            "Updating model defaults",
            lambda: cli_api._request("PATCH", f"/admin/models/{record['id']}", json_body=payload),
        )
        if ok:
            self.notify(f"Updated defaults for {record.get('nickname')}", timeout=8)
            await self.refresh_models()

    def _model_job_id(self, record: dict[str, Any]) -> str | None:
        job = record.get("registration_job")
        if isinstance(job, dict) and isinstance(job.get("id"), str):
            return cast(str, job["id"])
        self.notify(f"{record.get('nickname')} has no registration job.", severity="warning")
        return None

    async def _review_model(self, record: dict[str, Any]) -> None:
        job_id = self._model_job_id(record)
        if job_id is None:
            return
        ok, job = await self._call(
            "Loading registration job",
            lambda: cli_api._request("GET", f"/staff/model-jobs/{job_id}"),
        )
        if ok:
            self.query_one("#model-details", Static).update(
                Panel(Pretty(job, expand_all=True), title="Registration job")
            )

    async def _retry_model(self, record: dict[str, Any]) -> None:
        job_id = self._model_job_id(record)
        if job_id is None:
            return
        ok, result = await self._call(
            "Retrying registration",
            lambda: cli_api._request("POST", f"/staff/model-jobs/{job_id}/retry"),
        )
        if ok:
            self.notify(f"Registration queued: {result}")
            await self.refresh_models()

    async def _disable_model(self, record: dict[str, Any]) -> None:
        ok, _ = await self._call(
            "Disabling model",
            lambda: cli_api._request("POST", f"/staff/models/{record['id']}/disable"),
        )
        if ok:
            self.notify(f"Disabled {record.get('nickname')}.")
            await self.refresh_models()

    def _open_key_access(self) -> None:
        fields = (FieldSpec("key", "API key nickname or full API key", required=True),)
        self.push_screen(FormModal("Show model access", fields, "Show"), self._key_access_result)

    def _key_access_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.run_worker(self._show_key_access(values), exit_on_error=False)

    async def _show_key_access(self, values: FormResult) -> None:
        ok, record = await self._call(
            "Loading key access", lambda: cli_api._key_record(_str_value(values, "key"))
        )
        if ok and record is not None:
            self.query_one("#model-details", Static).update(
                Panel(
                    Pretty(
                        {
                            "API key": record.get("nickname"),
                            "Granted models": record.get("granted_models") or [],
                        },
                        expand_all=True,
                    ),
                    title="Model access",
                )
            )

    def _open_access_update(self, model: dict[str, Any] | None) -> None:
        default_models = "" if model is None else str(model.get("nickname") or "")
        fields = (
            FieldSpec("key", "API key nickname or full API key", required=True),
            FieldSpec(
                "models",
                "Model nicknames (comma-separated)",
                value=default_models,
                required=True,
            ),
            FieldSpec(
                "mode",
                "Change",
                value="add",
                options=(("Grant access", "add"), ("Revoke access", "remove")),
            ),
        )
        self.push_screen(FormModal("Change model access", fields, "Apply"), self._access_result)

    def _access_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.run_worker(self._update_access(values), exit_on_error=False)

    async def _update_access(self, values: FormResult) -> None:
        payload = {
            "key": _str_value(values, "key"),
            "models": _csv_value(values, "models"),
            "mode": _str_value(values, "mode"),
        }
        ok, result = await self._call(
            "Updating model access",
            lambda: cli_api._request("POST", "/staff/model-access", json_body=payload),
        )
        if ok:
            self.notify(f"Model access updated: {result}")
            await self.refresh_keys()

    def _open_profile_edit(self, profile: dict[str, Any]) -> None:
        launch_args = profile.get("launch_args")
        launch = launch_args if isinstance(launch_args, dict) else {}
        gguf_files = self.profile_payload.get("available_gguf_files")
        gguf_help = ""
        if isinstance(gguf_files, list) and gguf_files:
            gguf_help = "Available: " + ", ".join(str(item) for item in gguf_files)
        fields = (
            FieldSpec(
                "engine",
                "Engine",
                value=str(profile.get("engine") or "vllm"),
                options=(("vLLM", "vllm"), ("llama.cpp", "llama.cpp")),
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
                value=(
                    ""
                    if profile.get("max_num_batched_tokens") is None
                    else str(profile["max_num_batched_tokens"])
                ),
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
            FieldSpec("restart_workers", "Drain current workers and use immediately", value=True),
        )

        def finished(values: FormResult | None) -> None:
            if values is not None:
                self.run_worker(self._edit_profile(profile, values), exit_on_error=False)

        self.push_screen(
            FormModal("Override placement profile", fields, "Apply override"), finished
        )

    async def _edit_profile(self, profile: dict[str, Any], values: FormResult) -> None:
        if self.profile_model is None:
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
            self.notify(str(exc), severity="error")
            return
        model_id = self.profile_model["id"]
        ok, result = await self._call(
            "Updating placement profile",
            lambda: cli_api._request(
                "PATCH",
                f"/admin/models/{model_id}/profiles/{profile['id']}",
                json_body=payload,
            ),
        )
        if ok:
            self.notify("Placement profile updated.", timeout=8)
            if isinstance(result, dict) and result.get("drained_worker_ids"):
                self.notify(
                    "Draining workers: " + ", ".join(result["drained_worker_ids"]), timeout=10
                )
            await self.refresh_profiles()

    async def _set_profile_active(self, profile: dict[str, Any], active: bool) -> None:
        if self.profile_model is None:
            return
        action = "enable" if active else "disable"
        ok, result = await self._call(
            f"{action.title()} placement profile",
            lambda: cli_api._request(
                "POST",
                f"/admin/models/{self.profile_model['id']}/profiles/{profile['id']}/{action}",
            ),
        )
        if ok:
            self.notify(f"Placement profile {action}d.", timeout=8)
            if isinstance(result, dict) and result.get("drained_worker_ids"):
                self.notify(
                    "Draining workers: " + ", ".join(result["drained_worker_ids"]), timeout=10
                )
            await self.refresh_profiles()

    async def _set_maintenance(self, mode: str) -> None:
        ok, payload = await self._call(
            "Updating maintenance mode",
            lambda: cli_api._request("POST", "/admin/maintenance", json_body={"mode": mode}),
        )
        if ok:
            self.query_one("#maintenance-output", Static).update(
                Panel(Pretty(payload, expand_all=True), title="Maintenance status")
            )
            self.notify("Machine is draining." if mode == "drain" else "Normal service resumed.")

    def _open_doctor(self) -> None:
        config = os.environ.get("LLMRIO_CONFIG", "config.toml")
        fields = (FieldSpec("config", "Configuration file", value=config, required=True),)
        self.push_screen(FormModal("Host diagnostics", fields, "Run doctor"), self._doctor_result)

    def _doctor_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.run_worker(
                self._run_doctor(Path(_str_value(values, "config"))), exit_on_error=False
            )

    async def _run_doctor(self, config: Path) -> None:
        ok, report = await self._call("Running host diagnostics", lambda: _doctor_report(config))
        if ok and report is not None:
            self.query_one("#diagnostics-output", Static).update(
                Panel(Pretty(report, expand_all=True), title="Doctor report")
            )
            errors = report.get("errors")
            if isinstance(errors, list) and errors:
                self.notify(f"Doctor found {len(errors)} issue(s).", severity="warning", timeout=8)
            else:
                self.notify("Doctor checks passed.")

    def _open_start_service(self) -> None:
        config = os.environ.get("LLMRIO_CONFIG", "config.toml")
        fields = (
            FieldSpec(
                "config",
                "Configuration file",
                value=config,
                required=True,
                help_text="The TUI will close and the service will take over this terminal.",
            ),
        )
        self.push_screen(
            FormModal("Start LLM-RIO service", fields, "Start service"), self._serve_result
        )

    def _serve_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.exit(Path(_str_value(values, "config")))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("nav-"):
            page = button_id.removeprefix("nav-")
            if page == "quit":
                self.exit(None)
            else:
                self._navigate(page)
            return
        if button_id == "dashboard-refresh":
            self.run_worker(self.refresh_all(), exit_on_error=False)
        elif button_id in {"dashboard-start-service", "system-start-service"}:
            self._open_start_service()
        elif button_id == "keys-refresh":
            self.run_worker(self.refresh_keys(), exit_on_error=False)
        elif button_id == "keys-create":
            self._open_create_key()
        elif button_id == "keys-copy":
            if (record := self._selected_key()) is not None:
                self._copy_key_to_clipboard(record)
        elif button_id == "keys-rotate":
            if (key_record := self._selected_key()) is not None:
                self._confirm(
                    "Rotate API key",
                    f"Rotate '{key_record.get('nickname')}'? Existing clients will stop "
                    "authenticating.",
                    "Rotate",
                    lambda: self._rotate_key(key_record),
                )
        elif button_id == "keys-limit":
            if (record := self._selected_key()) is not None:
                self._open_key_limit(record)
        elif button_id in {"keys-reset", "keys-restore", "keys-revoke", "keys-delete"}:
            if (action_key_record := self._selected_key()) is not None:
                action = button_id.removeprefix("keys-")
                label = {
                    "reset": "Reset usage",
                    "restore": "Restore",
                    "revoke": "Revoke",
                    "delete": "Delete",
                }[action]
                self._confirm(
                    label,
                    f"{label} for API key '{action_key_record.get('nickname')}'?",
                    label,
                    lambda: self._key_action(action_key_record, action),
                )
        elif button_id == "models-refresh":
            self.run_worker(self.refresh_models(), exit_on_error=False)
        elif button_id == "models-add":
            self._open_add_model()
        elif button_id == "models-clone":
            if (clone_model_record := self._selected_model()) is not None:
                self._open_clone_model_profile(clone_model_record)
        elif button_id == "models-edit":
            if (edit_model_record := self._selected_model()) is not None:
                self._open_edit_model(edit_model_record)
        elif button_id == "models-review":
            if (record := self._selected_model()) is not None:
                self.run_worker(self._review_model(record), exit_on_error=False)
        elif button_id == "models-retry":
            if (retry_model_record := self._selected_model()) is not None:
                self._confirm(
                    "Retry registration",
                    f"Queue the registration job for '{retry_model_record.get('nickname')}' again?",
                    "Retry",
                    lambda: self._retry_model(retry_model_record),
                )
        elif button_id == "models-disable":
            if (disable_model_record := self._selected_model()) is not None:
                self._confirm(
                    "Disable model",
                    f"Disable '{disable_model_record.get('nickname')}' and prevent new requests?",
                    "Disable",
                    lambda: self._disable_model(disable_model_record),
                )
        elif button_id == "models-key-access":
            self._open_key_access()
        elif button_id == "models-access-update":
            self._open_access_update(self._selected_model())
        elif button_id == "models-profiles":
            if (record := self._selected_model()) is not None:
                self.profile_model = record
                self._navigate("profiles")
                self.run_worker(self.refresh_profiles(), exit_on_error=False)
        elif button_id == "profiles-back":
            self._navigate("models")
        elif button_id == "profiles-refresh":
            self.run_worker(self.refresh_profiles(), exit_on_error=False)
        elif button_id == "profiles-edit":
            if (profile := self._selected_profile()) is not None:
                self._open_profile_edit(profile)
        elif button_id in {"profiles-enable", "profiles-disable"}:
            if (profile := self._selected_profile()) is not None:
                active = button_id == "profiles-enable"
                if bool(profile.get("active")) is active:
                    self.notify(
                        f"Placement profile is already {'enabled' if active else 'disabled'}."
                    )
                elif active:
                    self.run_worker(self._set_profile_active(profile, True), exit_on_error=False)
                else:
                    self._confirm(
                        "Disable placement profile",
                        "Disable the selected profile and drain workers using it?",
                        "Disable",
                        lambda: self._set_profile_active(profile, False),
                    )

        elif button_id == "maintenance-refresh":
            self.run_worker(self.refresh_maintenance(), exit_on_error=False)
        elif button_id == "maintenance-drain":
            self._confirm(
                "Enter maintenance mode",
                "Stop accepting new work and drain active workers?",
                "Drain",
                lambda: self._set_maintenance("drain"),
            )
        elif button_id == "maintenance-resume":
            self._confirm(
                "Resume service",
                "Return this machine to active request scheduling?",
                "Resume",
                lambda: self._set_maintenance("active"),
            )
        elif button_id == "system-doctor":
            self._open_doctor()
        elif button_id == "system-info":
            self.run_worker(self.refresh_service_info(), exit_on_error=False)


def _str_value(values: FormResult, key: str) -> str:
    value = values.get(key, "")
    return value if isinstance(value, str) else str(value)


def _optional_str(values: FormResult, key: str) -> str | None:
    value = _str_value(values, key).strip()
    return value or None


def _bool_value(values: FormResult, key: str) -> bool:
    value = values.get(key, False)
    return value if isinstance(value, bool) else value.lower() in {"1", "true", "yes"}


def _csv_value(values: FormResult, key: str) -> list[str]:
    return [item.strip() for item in _str_value(values, key).split(",") if item.strip()]


def _int_value(
    values: FormResult,
    key: str,
    label: str,
    *,
    minimum: int,
) -> int:
    try:
        value = int(_str_value(values, key))
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _safe_base_url() -> str:
    try:
        return cli_api._base_url()
    except Exception as exc:
        return f"unavailable ({exc})"


def _doctor_report(config: Path) -> dict[str, Any]:
    settings = Settings(config_file=config)
    report: dict[str, Any] = {
        "machine_id": settings.machine_id,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executables": {
            "nvidia-smi": shutil.which("nvidia-smi"),
            "vllm": shutil.which(settings.engines.vllm_executable),
            "llama.cpp": shutil.which(settings.engines.llama_cpp_executable),
        },
        "paths": {},
        "errors": [],
    }
    for name, path in {
        "database_parent": settings.database_path.parent,
        "model_store": settings.model_store,
        "log_dir": settings.log_dir,
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        report["paths"][name] = {
            "path": str(path.resolve()),
            "writable": os.access(path, os.W_OK),
        }
    try:
        inventory = discover_inventory(settings.machine_id, settings.managed_gpu_uuids)
        report["inventory"] = {
            "driver_version": inventory.driver_version,
            "cuda_driver_version": inventory.cuda_driver_version,
            "fingerprint": inventory.fingerprint,
            "topology_hash": inventory.topology_hash,
            "gpus": [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "vram_mib": gpu.total_vram_mib,
                    "compute_capability": gpu.compute_capability,
                    "pci_bus_id": gpu.pci_bus_id,
                }
                for gpu in inventory.gpus
            ],
        }
    except InventoryError as exc:
        report["errors"].append({"stage": "inventory", "message": str(exc)})
    if not report["executables"]["vllm"]:
        report["errors"].append({"stage": "engine", "message": "vllm executable not found"})
    return report


def run_tui() -> Path | None:
    """Run the terminal app and return a config path when the user chooses Serve."""
    result = RioTui().run()
    return result if isinstance(result, Path) else None
