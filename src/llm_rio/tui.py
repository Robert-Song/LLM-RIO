from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Static,
)
from textual.worker import WorkerCancelled, WorkerFailed

from llm_rio import admin_client as client_api
from llm_rio.ui.components import (
    ConfirmModal,
    FormModal,
    ServiceLaunch,
)
from llm_rio.ui.dashboard import DashboardController
from llm_rio.ui.keys import KeysController
from llm_rio.ui.models import ModelsController
from llm_rio.ui.profiles import ProfilesController
from llm_rio.ui.system import SystemController

FormValue = str | bool
FormResult = dict[str, FormValue]
InputKind = Literal["text", "integer", "number"]
T = TypeVar("T")
_form_retry: ContextVar[Callable[[], None] | None] = ContextVar("form_retry", default=None)


class RioTui(App[ServiceLaunch | None]):
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

    #dashboard-summary, #dashboard-usage {
        height: auto;
        min-height: 8;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    #dashboard-requests-table, #dashboard-models-table, #dashboard-gpus-table {
        height: 12;
        margin-bottom: 1;
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
        self.dashboard_payload: dict[str, Any] = {}
        self._dashboard_refreshing = False
        self.key_records: list[dict[str, Any]] = []
        self.model_records: list[dict[str, Any]] = []
        self.profile_records: list[dict[str, Any]] = []
        self.profile_payload: dict[str, Any] = {}
        self.profile_model: dict[str, Any] | None = None
        self.capabilities: dict[str, Any] = {}
        self._pending_operations: set[str] = set()

    @property
    def models_controller(self) -> ModelsController:
        if not hasattr(self, "_models_controller"):
            self._models_controller = ModelsController(self)
        return self._models_controller

    @property
    def profiles_controller(self) -> ProfilesController:
        if not hasattr(self, "_profiles_controller"):
            self._profiles_controller = ProfilesController(self)
        return self._profiles_controller

    @property
    def keys_controller(self) -> KeysController:
        if not hasattr(self, "_keys_controller"):
            self._keys_controller = KeysController(self)
        return self._keys_controller

    @property
    def dashboard_controller(self) -> DashboardController:
        if not hasattr(self, "_dashboard_controller"):
            self._dashboard_controller = DashboardController(self)
        return self._dashboard_controller

    @property
    def system_controller(self) -> SystemController:
        if not hasattr(self, "_system_controller"):
            self._system_controller = SystemController(self)
        return self._system_controller

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
                        yield Button("Refresh dashboard", id="dashboard-refresh", variant="primary")
                    yield Static("Loading usage analytics…", id="dashboard-usage")
                    yield Static("Live request queue", classes="page-title")
                    yield DataTable(
                        zebra_stripes=True, cursor_type="row", id="dashboard-requests-table"
                    )
                    yield Static("Model popularity", classes="page-title")
                    yield DataTable(
                        zebra_stripes=True, cursor_type="row", id="dashboard-models-table"
                    )
                    yield Static("Live GPU status", classes="page-title")
                    yield DataTable(
                        zebra_stripes=True, cursor_type="row", id="dashboard-gpus-table"
                    )
                    yield Static("Connecting to the control plane…", id="dashboard-summary")
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
                        yield Button("Change model access", id="keys-access-update")
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
                        yield Button("Clone model", id="models-clone")
                        yield Button("Edit model", id="models-edit")
                        yield Button("Review job", id="models-review")
                        yield Button("Validate/Revalidate…", id="models-retry")
                        yield Button("Disable", id="models-disable", variant="warning")
                        yield Button("Change user access", id="models-user-access")
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
                        yield Button("Validate/Revalidate…", id="profiles-revalidate")
                        yield Button("Enable selected", id="profiles-activation")
                        yield Button("Advanced…", id="profiles-advanced")
                    yield DataTable(zebra_stripes=True, cursor_type="row", id="profiles-table")
                    yield Static(
                        "Select a profile to see its launch settings.",
                        id="profile-details",
                        classes="details",
                    )
                with VerticalScroll(id="maintenance", classes="page"):
                    yield Static("Maintenance", classes="page-title")
                    yield Static(
                        "Drain to validate pending models in normal mode. "
                        "Resume after validation finishes.",
                        classes="page-description",
                    )
                    with Grid(classes="toolbar"):
                        yield Button("Refresh status", id="maintenance-refresh", variant="primary")
                        yield Button("Drain", id="maintenance-drain", variant="warning")
                        yield Button("Resume", id="maintenance-resume")
                        yield Button(
                            "Summarize usage", id="maintenance-summarize", variant="warning"
                        )
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
            "#", "Active", "Engine", "GPUs / TP", "Context", "Max sequences", "Mode", "Measured"
        )
        self.query_one("#dashboard-models-table", DataTable).add_columns(
            "Model", "Current rank", "Current tokens", "Total rank", "Total tokens", "Share"
        )
        self.query_one("#dashboard-requests-table", DataTable).add_columns(
            "State", "API key", "Model", "Input est.", "Reserved", "Queued at"
        )
        self.query_one("#dashboard-gpus-table", DataTable).add_columns(
            "GPU", "Util", "VRAM", "Temp", "Power", "Model", "State", "Weights", "Slots"
        )
        self.run_worker(self.refresh_all(initial=True), name="initial-refresh", exit_on_error=False)
        self.set_interval(2.0, self._refresh_dashboard_if_visible)

    def _set_status(self, message: str) -> None:
        self.query_one("#status-bar", Static).update(message)

    def show_form(self, form: FormModal, callback: Callable[[FormResult | None], None]) -> None:
        def submitted(values: FormResult | None) -> None:
            if values is None:
                callback(None)
                return
            fields = tuple(
                replace(field, value=values.get(field.key, field.value)) for field in form.fields
            )

            def retry() -> None:
                self.show_form(
                    FormModal(form.form_title, fields, form.submit_label, validate=form.validate),
                    callback,
                )

            token = _form_retry.set(retry)
            try:
                callback(values)
            finally:
                _form_retry.reset(token)

        self.push_screen(form, submitted)

    def restore_form(self) -> None:
        retry = _form_retry.get()
        _form_retry.set(None)
        if retry is not None:
            retry()

    async def _call(
        self,
        label: str,
        operation: Callable[[], T],
        *,
        notify_error: bool = True,
    ) -> tuple[bool, T | None]:
        if label in self._pending_operations:
            self.notify(f"{label} is already in progress.", severity="warning")
            return False, None
        self._pending_operations.add(label)
        retry = _form_retry.get()
        _form_retry.set(None)
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
            if retry is not None:
                retry()
            return False, None
        except WorkerFailed as exc:
            message = str(exc.error)
            self._set_status(f"{label} failed: {message}")
            if notify_error:
                self.notify(message, title=f"{label} failed", severity="error", timeout=8)
            if retry is not None:
                retry()
            return False, None
        finally:
            self._pending_operations.discard(label)
        self._set_status(f"{label} complete")
        return True, result

    async def refresh_capabilities(self, *, notify_error: bool = True) -> None:
        ok, payload = await self._call(
            "Loading capabilities",
            lambda: client_api.request("GET", "/admin/capabilities"),
            notify_error=notify_error,
        )
        if ok and isinstance(payload, dict):
            self.capabilities = payload
            name = str(payload.get("name", "unknown"))
            engines = ", ".join(payload.get("engines", []))
            self.sub_title = f"{name} • {engines}" + (
                " • experimental" if payload.get("experimental") else ""
            )

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
            self.run_worker(self.refresh_dashboard(), exit_on_error=False)

    async def refresh_all(self, *, initial: bool = False) -> None:
        await self.refresh_capabilities(notify_error=not initial)
        await self.refresh_keys(notify_error=not initial)
        await self.refresh_models(notify_error=not initial)
        await self.refresh_maintenance(notify_error=not initial)
        await self.refresh_dashboard(notify_error=not initial)
        await self.refresh_service_info(notify_error=not initial)
        self._update_dashboard()

    async def _refresh_dashboard_if_visible(self) -> None:
        return await self.dashboard_controller._refresh_dashboard_if_visible()

    async def refresh_dashboard(self, *, notify_error: bool = True) -> None:
        return await self.dashboard_controller.refresh_dashboard(notify_error=notify_error)

    def _render_dashboard(self, payload: dict[str, Any]) -> None:
        return self.dashboard_controller._render_dashboard(payload)

    @staticmethod
    def _token_estimate(value: Any) -> str:
        return f"{int(value):,}" if value is not None else "-"

    @staticmethod
    def table_selection(table: DataTable[Any]) -> str | None:
        if table.row_count:
            return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        return None

    @staticmethod
    def restore_selection(table: DataTable[Any], key: str | None) -> None:
        if key is not None and key in table.rows:
            table.move_cursor(row=table.get_row_index(key), animate=False)

    def _replace_table_rows(
        self,
        table: DataTable[Any],
        rows: Iterable[tuple[tuple[Any, ...], str]],
    ) -> None:
        """Refresh rows without resetting the user's table scroll position."""
        selected_key = self.table_selection(table)
        scroll_x, scroll_y = table.scroll_offset
        table.clear(columns=False)
        for cells, key in rows:
            table.add_row(*cells, key=key)
        self.restore_selection(table, selected_key)
        table.call_after_refresh(table.scroll_to, scroll_x, scroll_y, animate=False)

    async def refresh_keys(self, *, notify_error: bool = True) -> None:
        return await self.keys_controller.refresh_keys(notify_error=notify_error)

    async def refresh_models(self, *, notify_error: bool = True) -> None:
        return await self.models_controller.refresh_models(notify_error=notify_error)

    async def refresh_maintenance(self, *, notify_error: bool = True) -> None:
        return await self.system_controller.refresh_maintenance(notify_error=notify_error)

    async def refresh_service_info(self, *, notify_error: bool = True) -> None:
        return await self.system_controller.refresh_service_info(notify_error=notify_error)

    async def refresh_profiles(self, *, notify_error: bool = True) -> None:
        if self.profile_model is None:
            return
        nickname = str(self.profile_model.get("nickname") or "")
        ok, payload = await self._call(
            "Loading placement profiles",
            lambda: client_api.model_profiles(nickname),
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
        selected_key = self.table_selection(table)
        table.clear(columns=False)
        for number, profile in enumerate(self.profile_records, 1):
            table.add_row(
                str(number),
                "yes" if profile.get("active") else "no",
                str(profile.get("engine") or ""),
                f"{profile.get('gpu_count') or 0} / {profile.get('tensor_parallel_size') or 0}",
                f"{int(profile.get('max_model_len') or 0):,}",
                str(profile.get("max_num_seqs") or "engine default"),
                str(profile.get("serving_mode") or ""),
                "yes" if profile.get("measurements_valid") else "no",
                key=str(profile.get("id") or number),
            )
        self.restore_selection(table, selected_key)
        gguf_files = payload.get("available_gguf_files")
        gguf_note = ""
        if isinstance(gguf_files, list) and gguf_files:
            gguf_note = "  •  GGUF: " + ", ".join(str(item) for item in gguf_files)
        self.query_one("#profiles-description", Static).update(f"Model: {nickname}{gguf_note}")
        if self.profile_records:
            self._show_profile_details(table.cursor_row)
        else:
            self.query_one("#profile-details", Static).update(
                "No placement profiles are recorded for this model."
            )

    def _update_dashboard(self) -> None:
        return self.dashboard_controller._update_dashboard()

    def _show_key_details(self, index: int) -> None:
        return self.keys_controller._show_key_details(index)

    def _show_model_details(self, index: int) -> None:
        return self.models_controller._show_model_details(index)

    def _show_profile_details(self, index: int) -> None:
        return self.profiles_controller._show_profile_details(index)

    def _sync_profile_verification_buttons(self, profile: dict[str, Any]) -> None:
        return self.profiles_controller._sync_profile_verification_buttons(profile)

    def _open_trust_measurements(self, profile: dict[str, Any]) -> None:
        return self.profiles_controller._open_trust_measurements(profile)

    async def _trust_measurements(self, profile: dict[str, Any], reason: str) -> None:
        return await self.profiles_controller._trust_measurements(profile, reason)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "keys-table":
            self._show_key_details(event.cursor_row)
        elif event.data_table.id == "models-table":
            self._show_model_details(event.cursor_row)
        elif event.data_table.id == "profiles-table":
            self._show_profile_details(event.cursor_row)

    def _selected_key(self) -> dict[str, Any] | None:
        return self.keys_controller._selected_key()

    def _copy_key_to_clipboard(self, record: dict[str, Any]) -> None:
        return self.keys_controller._copy_key_to_clipboard(record)

    def _selected_model(self) -> dict[str, Any] | None:
        return self.models_controller._selected_model()

    def _selected_profile(self) -> dict[str, Any] | None:
        return self.profiles_controller._selected_profile()

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
        return self.keys_controller._open_create_key()

    def _create_key_result(self, values: FormResult | None) -> None:
        return self.keys_controller._create_key_result(values)

    async def _create_key(self, values: FormResult) -> None:
        return await self.keys_controller._create_key(values)

    async def _rotate_key(self, record: dict[str, Any]) -> None:
        return await self.keys_controller._rotate_key(record)

    def _open_key_limit(self, record: dict[str, Any]) -> None:
        return self.keys_controller._open_key_limit(record)

    async def _set_key_limit(self, record: dict[str, Any], values: FormResult) -> None:
        return await self.keys_controller._set_key_limit(record, values)

    async def _key_action(self, record: dict[str, Any], action: str) -> None:
        return await self.keys_controller._key_action(record, action)

    def _open_user_model_access(self, record: dict[str, Any]) -> None:
        return self.models_controller._open_user_model_access(record)

    async def _replace_user_model_access(
        self, record: dict[str, Any], model_ids: list[str]
    ) -> None:
        return await self.models_controller._replace_user_model_access(record, model_ids)

    def _open_add_model(self) -> None:
        return self.models_controller._open_add_model()

    def _add_model_result(self, values: FormResult | None) -> None:
        return self.models_controller._add_model_result(values)

    async def _add_model(self, values: FormResult) -> None:
        return await self.models_controller._add_model(values)

    def _open_edit_model(self, record: dict[str, Any]) -> None:
        return self.models_controller._open_edit_model(record)

    def _open_clone_model_profile(self, record: dict[str, Any]) -> None:
        return self.profiles_controller._open_clone_model_profile(record)

    async def _clone_model_profile(self, source: dict[str, Any], values: FormResult) -> None:
        return await self.profiles_controller._clone_model_profile(source, values)

    async def _edit_model(self, record: dict[str, Any], values: FormResult) -> None:
        return await self.models_controller._edit_model(record, values)

    def _model_job_id(self, record: dict[str, Any]) -> str | None:
        return self.models_controller._model_job_id(record)

    async def _review_model(self, record: dict[str, Any]) -> None:
        return await self.models_controller._review_model(record)

    async def _open_revalidation(
        self, record: dict[str, Any], profile: dict[str, Any] | None = None
    ) -> None:
        return await self.models_controller._open_revalidation(record, profile)

    async def _retry_model(
        self, record: dict[str, Any], overrides: dict[str, Any], *, profile_id: str | None = None
    ) -> None:
        return await self.models_controller._retry_model(record, overrides, profile_id=profile_id)

    async def _disable_model(self, record: dict[str, Any]) -> None:
        return await self.models_controller._disable_model(record)

    def _open_model_user_access(self, model: dict[str, Any]) -> None:
        return self.models_controller._open_model_user_access(model)

    async def _replace_model_user_access(
        self, model: dict[str, Any], selected_user_ids: list[str]
    ) -> None:
        return await self.models_controller._replace_model_user_access(model, selected_user_ids)

    def _open_key_access(self) -> None:
        return self.keys_controller._open_key_access()

    def _key_access_result(self, values: FormResult | None) -> None:
        return self.keys_controller._key_access_result(values)

    async def _show_key_access(self, values: FormResult) -> None:
        return await self.keys_controller._show_key_access(values)

    def _open_access_update(self, model: dict[str, Any] | None) -> None:
        return self.keys_controller._open_access_update(model)

    def _access_result(self, values: FormResult | None) -> None:
        return self.keys_controller._access_result(values)

    async def _update_access(self, values: FormResult) -> None:
        return await self.keys_controller._update_access(values)

    def _open_profile_edit(self, profile: dict[str, Any]) -> None:
        return self.profiles_controller._open_profile_edit(profile)

    async def _edit_profile(self, profile: dict[str, Any], values: FormResult) -> None:
        return await self.profiles_controller._edit_profile(profile, values)

    async def _set_profile_active(self, profile: dict[str, Any], active: bool) -> None:
        return await self.profiles_controller._set_profile_active(profile, active)

    async def _summarize_usage(self) -> None:
        return await self.system_controller._summarize_usage()

    async def _set_maintenance(self, mode: str) -> None:
        return await self.system_controller._set_maintenance(mode)

    def _open_doctor(self) -> None:
        return self.system_controller._open_doctor()

    def _doctor_result(self, values: FormResult | None) -> None:
        return self.system_controller._doctor_result(values)

    async def _run_doctor(self, config: Path) -> None:
        return await self.system_controller._run_doctor(config)

    def _open_start_service(self) -> None:
        return self.system_controller._open_start_service()

    def _serve_result(self, values: FormResult | None) -> None:
        return self.system_controller._serve_result(values)

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
            self.run_worker(self.refresh_dashboard(), exit_on_error=False)
        elif button_id in {"dashboard-start-service", "system-start-service"}:
            self._open_start_service()
        elif button_id == "keys-refresh":
            self.run_worker(self.refresh_keys(), exit_on_error=False)
        elif button_id == "keys-create":
            self._open_create_key()
        elif button_id == "keys-copy":
            if (record := self._selected_key()) is not None:
                self._copy_key_to_clipboard(record)
        elif button_id == "keys-access-update":
            if (record := self._selected_key()) is not None:
                self._open_user_model_access(record)
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
                self.run_worker(self._open_revalidation(retry_model_record), exit_on_error=False)
        elif button_id == "profiles-revalidate":
            if self.profile_model is not None and (profile := self._selected_profile()) is not None:
                self.run_worker(
                    self._open_revalidation(self.profile_model, profile), exit_on_error=False
                )
        elif button_id == "models-disable":
            if (disable_model_record := self._selected_model()) is not None:
                self._confirm(
                    "Disable model",
                    f"Disable '{disable_model_record.get('nickname')}' and prevent new requests?",
                    "Disable",
                    lambda: self._disable_model(disable_model_record),
                )
        elif button_id == "models-user-access":
            if (record := self._selected_model()) is not None:
                self._open_model_user_access(record)
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
        elif button_id == "profiles-advanced":
            self._open_trust_measurements({})
        elif button_id == "profiles-activation":
            if (profile := self._selected_profile()) is not None:
                active = not bool(profile.get("active"))
                self._confirm(
                    "Change profile availability",
                    "Change availability and drain affected workers?",
                    "Enable" if active else "Disable",
                    lambda: self._set_profile_active(profile, active),
                )

        elif button_id == "maintenance-refresh":
            self.run_worker(self.refresh_maintenance(), exit_on_error=False)
        elif button_id == "maintenance-summarize":
            self._confirm(
                "Summarize settled usage",
                "Replace the current summary, extend lifetime totals, and delete settled "
                "per-call rows through now?",
                "Summarize",
                self._summarize_usage,
            )
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


def run_tui() -> ServiceLaunch | None:
    """Return the configuration and mode when the user chooses Start service."""
    return RioTui().run()
