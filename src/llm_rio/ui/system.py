from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.panel import Panel
from rich.pretty import Pretty
from textual.widgets import (
    Static,
)

from llm_rio import admin_client as client_api
from llm_rio import connection
from llm_rio.config import ServingMode
from llm_rio.ui.components import (
    FieldSpec,
    FormModal,
    ServiceLaunch,
    _doctor_report,
    _str_value,
)

if TYPE_CHECKING:
    from llm_rio.tui import RioTui

FormResult = dict[str, str | bool]


class SystemController:
    def __init__(self, app: RioTui) -> None:
        self.app = app

    @staticmethod
    def _configuration_file() -> Path:
        try:
            return connection.settings().config_file
        except click.ClickException:
            return Path(os.environ.get("LLMRIO_CONFIG_FILE", "config.toml"))

    async def refresh_maintenance(self, *, notify_error: bool = True) -> None:
        ok, payload = await self.app._call(
            "Loading maintenance status",
            lambda: client_api.request("GET", "/admin/maintenance"),
            notify_error=notify_error,
        )
        if ok:
            self.app.query_one("#maintenance-output", Static).update(
                Panel(Pretty(payload, expand_all=True), title="Maintenance status")
            )

    async def refresh_service_info(self, *, notify_error: bool = True) -> None:
        def load_info() -> dict[str, str]:
            config = self._configuration_file()
            result = {
                "Config file": str(config.resolve()),
                "Administrator credential": "unavailable",
            }
            try:
                result["API base URL"] = connection.base_url()
            except click.ClickException as exc:
                result["Configuration error"] = str(exc)
                return result
            try:
                connection.api_key()
            except Exception as exc:
                result["Administrator credential"] = str(exc)
            else:
                result["Administrator credential"] = "recovered from the protected local vault"
            return result

        ok, info = await self.app._call(
            "Loading service information", load_info, notify_error=notify_error
        )
        if ok and info is not None:
            self.app.query_one("#service-output", Static).update(
                Panel(Pretty(info, expand_all=True), title="Connection")
            )

    async def _summarize_usage(self) -> None:
        ok, payload = await self.app._call(
            "Summarizing usage", lambda: client_api.request("POST", "/admin/usage/summarize")
        )
        if ok and payload is not None:
            self.app.query_one("#maintenance-output", Static).update(
                Panel(Pretty(payload, expand_all=True), title="Usage summary")
            )
            self.app.notify("Settled usage records summarized.", timeout=8)

    async def _set_maintenance(self, mode: str) -> None:
        ok, payload = await self.app._call(
            "Updating maintenance mode",
            lambda: client_api.request("POST", "/admin/maintenance", json_body={"mode": mode}),
        )
        if ok:
            self.app.query_one("#maintenance-output", Static).update(
                Panel(Pretty(payload, expand_all=True), title="Maintenance status")
            )
            self.app.notify(
                "Machine is draining." if mode == "drain" else "Normal service resumed."
            )

    def _open_doctor(self) -> None:
        config = str(self._configuration_file())
        fields = (FieldSpec("config", "Configuration file", value=config, required=True),)
        self.app.show_form(
            FormModal("Host diagnostics", fields, "Run doctor"), self.app._doctor_result
        )

    def _doctor_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.app.run_worker(
                self.app._run_doctor(Path(_str_value(values, "config"))), exit_on_error=False
            )

    async def _run_doctor(self, config: Path) -> None:
        ok, report = await self.app._call(
            "Running host diagnostics", lambda: _doctor_report(config)
        )
        if ok and report is not None:
            self.app.query_one("#diagnostics-output", Static).update(
                Panel(Pretty(report, expand_all=True), title="Doctor report")
            )
            errors = report.get("errors")
            if isinstance(errors, list) and errors:
                self.app.notify(
                    f"Doctor found {len(errors)} issue(s).", severity="warning", timeout=8
                )
            else:
                self.app.notify("Doctor checks passed.")

    def _open_start_service(self) -> None:
        config = str(self._configuration_file())
        fields = (
            FieldSpec(
                "config",
                "Configuration file",
                value=config,
                required=True,
                help_text="The TUI will close and the service will take over this terminal.",
            ),
            FieldSpec(
                "mode",
                "Serving mode",
                value="configured",
                options=(("Use configuration / environment", "configured"),)
                + tuple((mode.value, mode.value) for mode in ServingMode),
                help_text=(
                    "Overrides serving_mode. Settings in other modes.* sections stay inactive."
                ),
            ),
        )
        self.app.show_form(
            FormModal(
                "Start LLM-RIO service", fields, "Start service", validate=self._validate_serve
            ),
            self.app._serve_result,
        )

    @staticmethod
    def _validate_serve(values: FormResult) -> None:
        mode = _str_value(values, "mode")
        try:
            connection.settings(
                Path(_str_value(values, "config")),
                mode=None if mode == "configured" else ServingMode(mode),
            )
        except click.ClickException as exc:
            raise ValueError(str(exc)) from None

    def _serve_result(self, values: FormResult | None) -> None:
        if values is not None:
            mode = _str_value(values, "mode")
            self.app.exit(
                ServiceLaunch(
                    config=Path(_str_value(values, "config")),
                    mode=None if mode == "configured" else ServingMode(mode),
                )
            )
