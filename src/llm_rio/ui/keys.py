from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    _bool_value,
    _csv_value,
    _int_value,
    _optional_str,
    _str_value,
)

if TYPE_CHECKING:
    from llm_rio.tui import RioTui

FormResult = dict[str, str | bool]


class KeysController:
    def __init__(self, app: RioTui) -> None:
        self.app = app

    async def refresh_keys(self, *, notify_error: bool = True) -> None:
        ok, records = await self.app._call(
            "Loading API keys", client_api.key_records, notify_error=notify_error
        )
        if not ok or records is None:
            return
        self.app.key_records = records
        table = self.app.query_one("#keys-table", DataTable)
        selected_key = self.app.table_selection(table)
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
        self.app.restore_selection(table, selected_key)
        if records:
            self.app._show_key_details(table.cursor_row)
        else:
            self.app.query_one("#key-details", Static).update("No API keys found.")
        self.app._update_dashboard()

    def _show_key_details(self, index: int) -> None:
        if not 0 <= index < len(self.app.key_records):
            return
        record = self.app.key_records[index]
        self.app.query_one("#key-details", Static).update(
            Panel(Pretty(record, expand_all=True), title=str(record.get("nickname") or "API key"))
        )

    def _selected_key(self) -> dict[str, Any] | None:
        index = self.app.query_one("#keys-table", DataTable).cursor_row
        if 0 <= index < len(self.app.key_records):
            return self.app.key_records[index]
        self.app.notify("Select an API key first.", severity="warning")
        return None

    def _copy_key_to_clipboard(self, record: dict[str, Any]) -> None:
        api_key = record.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            self.app.notify("The selected user's API key is unavailable.", severity="warning")
            return
        self.app.copy_to_clipboard(api_key)
        self.app.notify(f"Copied API key for {record.get('nickname')} to the clipboard.")

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
        self.app.show_form(
            FormModal("Create API key", fields, "Create"), self.app._create_key_result
        )

    def _create_key_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.app.run_worker(self.app._create_key(values), exit_on_error=False)

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
            self.app.restore_form()
            self.app.notify(str(exc), severity="error")
            return
        ok, result = await self.app._call(
            "Creating API key", lambda: client_api.request("POST", "/admin/keys", json_body=payload)
        )
        if ok and isinstance(result, dict):
            secret = result.get("api_key")
            self.app.notify(
                f"Created {result.get('nickname')}. Full key: {secret}",
                title="API key created",
                timeout=15,
            )
            await self.app.refresh_keys()

    async def _rotate_key(self, record: dict[str, Any]) -> None:
        ok, result = await self.app._call(
            "Rotating API key",
            lambda: client_api.request("POST", f"/admin/keys/{record['id']}/rotate"),
        )
        if ok and isinstance(result, dict):
            self.app.notify(
                f"New full key: {result.get('api_key')}",
                title=f"Rotated {record.get('nickname')}",
                timeout=15,
            )
            await self.app.refresh_keys()

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
                self.app.run_worker(self.app._set_key_limit(record, values), exit_on_error=False)

        self.app.show_form(
            FormModal(f"Set quota for {record.get('nickname')}", fields, "Update"), finished
        )

    async def _set_key_limit(self, record: dict[str, Any], values: FormResult) -> None:
        try:
            payload = {
                "limit_tokens": _int_value(values, "limit", "Token limit", minimum=0),
                "unlimited": _bool_value(values, "unlimited"),
            }
        except ValueError as exc:
            self.app.restore_form()
            self.app.notify(str(exc), severity="error")
            return
        ok, _ = await self.app._call(
            "Updating quota",
            lambda: client_api.request(
                "PUT", f"/admin/keys/{record['id']}/quota", json_body=payload
            ),
        )
        if ok:
            self.app.notify(f"Quota updated for {record.get('nickname')}.")
            await self.app.refresh_keys()

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
            method, suffix, label, success = ("DELETE", "", "Deleting API key", "API key deleted")
        ok, _ = await self.app._call(
            label, lambda: client_api.request(method, f"/admin/keys/{record['id']}{suffix}")
        )
        if ok:
            self.app.notify(f"{success} for {record.get('nickname')}.")
            await self.app.refresh_keys()

    def _open_key_access(self) -> None:
        fields = (FieldSpec("key", "API key nickname or full API key", required=True),)
        self.app.show_form(
            FormModal("Show model access", fields, "Show"), self.app._key_access_result
        )

    def _key_access_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.app.run_worker(self.app._show_key_access(values), exit_on_error=False)

    async def _show_key_access(self, values: FormResult) -> None:
        ok, record = await self.app._call(
            "Loading key access", lambda: client_api.key_record(_str_value(values, "key"))
        )
        if ok and record is not None:
            self.app.query_one("#model-details", Static).update(
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
                "models", "Model nicknames (comma-separated)", value=default_models, required=True
            ),
            FieldSpec(
                "mode",
                "Change",
                value="add",
                options=(("Grant access", "add"), ("Revoke access", "remove")),
            ),
        )
        self.app.show_form(
            FormModal("Change model access", fields, "Apply"), self.app._access_result
        )

    def _access_result(self, values: FormResult | None) -> None:
        if values is not None:
            self.app.run_worker(self.app._update_access(values), exit_on_error=False)

    async def _update_access(self, values: FormResult) -> None:
        payload = {
            "key": _str_value(values, "key"),
            "models": _csv_value(values, "models"),
            "mode": _str_value(values, "mode"),
        }
        ok, result = await self.app._call(
            "Updating model access",
            lambda: client_api.request("POST", "/staff/model-access", json_body=payload),
        )
        if ok:
            self.app.notify(f"Model access updated: {result}")
            await self.app.refresh_keys()
