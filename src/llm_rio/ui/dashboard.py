from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.pretty import Pretty
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Static,
)

from llm_rio import admin_client as client_api
from llm_rio.ui.components import (
    _safe_base_url,
)

if TYPE_CHECKING:
    from llm_rio.tui import RioTui

FormResult = dict[str, str | bool]


class DashboardController:
    def __init__(self, app: RioTui) -> None:
        self.app = app

    async def _refresh_dashboard_if_visible(self) -> None:
        if self.app.query_one("#content", ContentSwitcher).current == "dashboard":
            await self.app.refresh_dashboard(notify_error=False)

    async def refresh_dashboard(self, *, notify_error: bool = True) -> None:
        if self.app._dashboard_refreshing:
            return
        self.app._dashboard_refreshing = True
        try:
            ok, payload = await self.app._call(
                "Refreshing live dashboard",
                lambda: client_api.request("GET", "/admin/dashboard"),
                notify_error=notify_error,
            )
            if ok and isinstance(payload, dict):
                self.app.dashboard_payload = payload
                self.app._render_dashboard(payload)
        finally:
            self.app._dashboard_refreshing = False

    def _render_dashboard(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage")
        usage_payload = usage if isinstance(usage, dict) else {}
        current = usage_payload.get("current")
        current_window = current if isinstance(current, dict) else {}
        total = usage_payload.get("total")
        total_window = total if isinstance(total, dict) else {}

        def rate(window: dict[str, Any], key: str) -> str:
            value = window.get(key)
            return "-" if value is None else f"{float(value):,.2f}"

        usage_summary = {
            "Current window": (
                f"{current_window.get('period_start', '-')}"
                " -> "
                f"{current_window.get('period_end', '-')}"
            ),
            "Current tokens": f"{int(current_window.get('token_usage') or 0):,}",
            "Current throughput": f"{rate(current_window, 'tokens_per_minute')} tokens/min",
            "Current avg output": (
                f"{rate(current_window, 'average_output_tokens_per_second')} tokens/s"
            ),
            "Current requests": f"{int(current_window.get('request_count') or 0):,}",
            "Total tokens": f"{int(total_window.get('token_usage') or 0):,}",
            "Total throughput": f"{rate(total_window, 'tokens_per_minute')} tokens/min",
            "Total avg output": (
                f"{rate(total_window, 'average_output_tokens_per_second')} tokens/s"
            ),
            "Total requests": f"{int(total_window.get('request_count') or 0):,}",
        }
        self.app.query_one("#dashboard-usage", Static).update(
            Panel(Pretty(usage_summary, expand_all=True), title="Token usage")
        )
        raw_requests = payload.get("requests")
        requests = (
            [item for item in raw_requests if isinstance(item, dict)]
            if isinstance(raw_requests, list)
            else []
        )
        request_rows: list[tuple[tuple[Any, ...], str]] = []
        for row_number, item in enumerate(requests, start=1):
            request_rows.append(
                (
                    (
                        str(item.get("state") or "-"),
                        str(item.get("api_key") or "-"),
                        str(item.get("model") or "-"),
                        self.app._token_estimate(item.get("estimated_prompt_tokens")),
                        self.app._token_estimate(item.get("estimated_tokens")),
                        str(item.get("created_at") or "-"),
                    ),
                    str(item.get("request_id") or row_number),
                )
            )
        self.app._replace_table_rows(
            self.app.query_one("#dashboard-requests-table", DataTable), request_rows
        )
        popularity = usage_payload.get("model_popularity")
        popularity_payload = popularity if isinstance(popularity, dict) else {}
        current_models = popularity_payload.get("current")
        total_models = popularity_payload.get("total")
        current_records = (
            [item for item in current_models if isinstance(item, dict)]
            if isinstance(current_models, list)
            else []
        )
        total_records = (
            [item for item in total_models if isinstance(item, dict)]
            if isinstance(total_models, list)
            else []
        )
        current_by_id = {str(item.get("model_id")): item for item in current_records}
        total_by_id = {str(item.get("model_id")): item for item in total_records}
        model_ids = list(total_by_id)
        model_ids.extend(model_id for model_id in current_by_id if model_id not in total_by_id)
        model_ids.sort(
            key=lambda model_id: (
                int(total_by_id.get(model_id, {}).get("rank") or 10**9),
                int(current_by_id.get(model_id, {}).get("rank") or 10**9),
            )
        )
        model_rows: list[tuple[tuple[Any, ...], str]] = []
        for model_id in model_ids:
            current_item = current_by_id.get(model_id, {})
            total_item = total_by_id.get(model_id, {})
            model_rows.append(
                (
                    (
                        str(total_item.get("model") or current_item.get("model") or model_id),
                        str(current_item.get("rank") or "-"),
                        f"{int(current_item.get('token_usage') or 0):,}",
                        str(total_item.get("rank") or "-"),
                        f"{int(total_item.get('token_usage') or 0):,}",
                        f"{float(total_item.get('share') or 0) * 100:.1f}%",
                    ),
                    model_id,
                )
            )
        self.app._replace_table_rows(
            self.app.query_one("#dashboard-models-table", DataTable), model_rows
        )
        raw_gpus = payload.get("gpus")
        gpus = (
            [item for item in raw_gpus if isinstance(item, dict)]
            if isinstance(raw_gpus, list)
            else []
        )
        gpu_rows: list[tuple[tuple[Any, ...], str]] = []
        for gpu in gpus:
            raw_placements = gpu.get("placements")
            placements = (
                [item for item in raw_placements if isinstance(item, dict)]
                if isinstance(raw_placements, list)
                else []
            )
            models = ", ".join(str(item.get("model") or "-") for item in placements) or "idle"
            states = ", ".join(str(item.get("state") or "-") for item in placements) or "idle"
            weight_storage = (
                ", ".join(str(item.get("weight_storage") or "-") for item in placements) or "none"
            )
            slot_values: list[str] = []
            for placement in placements:
                raw_slots = placement.get("continuous_batching_slots")
                slots = raw_slots if isinstance(raw_slots, dict) else {}
                capacity = slots.get("capacity")
                slot_values.append(
                    f"{int(slots.get('active') or 0)}/{(capacity if capacity is not None else '?')}"
                )
            used_vram = int(gpu.get("used_vram_mib") or 0)
            total_vram = int(gpu.get("total_vram_mib") or 0)
            temperature = gpu.get("temperature_c")
            power = gpu.get("power_draw_w")
            gpu_rows.append(
                (
                    (
                        f"{gpu.get('index', '?')}: {gpu.get('name', 'GPU')}",
                        f"{int(gpu.get('gpu_utilization_percent') or 0)}%"
                        if gpu.get("available")
                        else "N/A",
                        f"{used_vram:,}/{total_vram:,} MiB",
                        f"{temperature} degrees C" if temperature is not None else "-",
                        f"{float(power):.1f} W" if power is not None else "-",
                        models,
                        states,
                        weight_storage,
                        ", ".join(slot_values) or "0/?",
                    ),
                    str(gpu.get("uuid") or gpu.get("index")),
                )
            )
        self.app._replace_table_rows(
            self.app.query_one("#dashboard-gpus-table", DataTable), gpu_rows
        )

    def _update_dashboard(self) -> None:
        available = sum(1 for model in self.app.model_records if model.get("state") == "AVAILABLE")
        review = sum(
            1 for model in self.app.model_records if model.get("state") == "NEEDS_ADMIN_REVIEW"
        )
        active_keys = sum(1 for key in self.app.key_records if key.get("active"))
        summary = {
            "API URL": _safe_base_url(),
            "Users": f"{len(self.app.key_records)} total / {active_keys} active",
            "Models": f"{len(self.app.model_records)} total / {available} available",
            "Needs administrator review": review,
            "Keyboard": "R refreshes the current page; Q exits",
        }
        self.app.query_one("#dashboard-summary", Static).update(
            Panel(Pretty(summary, expand_all=True), title="Control plane")
        )
