from __future__ import annotations

import json
import os
import platform
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

from llm_rio import connection
from llm_rio.api.schemas import ModelValidationOverrides
from llm_rio.config import ServingMode
from llm_rio.inventory import InventoryError, discover_inventory

FormValue = str | bool
FormResult = dict[str, FormValue]
InputKind = Literal["text", "integer", "number"]


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
    multiline: bool = False


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

    FormModal Input {
        color: #f5f5f5;
        background: #171717;
        height: 3;
        min-height: 3;
        padding: 0 1;
        border: tall #767676;
    }

    FormModal Input:focus {
        color: #ffffff;
        background: #252525;
        border: tall #8ab4f8;
    }

    FormModal TextArea {
        height: 8;
        min-height: 5;
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

    def __init__(
        self,
        title: str,
        fields: Iterable[FieldSpec],
        submit_label: str,
        *,
        validate: Callable[[FormResult], object] | None = None,
    ) -> None:
        super().__init__()
        self.form_title = title
        self.fields = tuple(fields)
        self.submit_label = submit_label
        self.validate = validate

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
                    elif field.multiline:
                        yield Label(field.label)
                        yield TextArea(str(field.value), id=f"field-{field.key}")
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

    def on_mount(self) -> None:
        for field in self.fields:
            selector = f"#field-{field.key}"
            if field.options:
                self.query_one(selector, Select).focus()
            elif isinstance(field.value, bool):
                self.query_one(selector, Checkbox).focus()
            elif field.multiline:
                self.query_one(selector, TextArea).focus()
            else:
                self.query_one(selector, Input).focus()
            return

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
            elif field.multiline:
                value = self.query_one(selector, TextArea).text.strip()
            else:
                value = self.query_one(selector, Input).value.strip()
            if field.required and isinstance(value, str) and (not value):
                self.notify(f"{field.label} is required.", severity="error")
                return
            values[field.key] = value
        if self.validate is not None:
            try:
                self.validate(values)
            except ValueError as exc:
                self.notify(str(exc), severity="error", timeout=10)
                return
        self.dismiss(values)


def _revalidation_overrides(values: FormResult) -> dict[str, Any]:
    raw_args = str(values.get("launch_args") or "{}").strip()
    try:
        launch_args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Engine arguments must be valid JSON: {exc.msg}") from exc
    if not isinstance(launch_args, dict):
        raise ValueError("Engine arguments must be a JSON object.")
    raw: dict[str, Any] = {"launch_args": launch_args}
    for key in (
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        value = str(values.get(key) or "").strip()
        if value:
            raw[key] = value
    return ModelValidationOverrides.model_validate(raw).model_dump(exclude_none=True)


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


@dataclass(frozen=True, slots=True)
class ServiceLaunch:
    config: Path
    mode: ServingMode | None


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


def _int_value(values: FormResult, key: str, label: str, *, minimum: int) -> int:
    try:
        value = int(_str_value(values, key))
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _safe_base_url() -> str:
    try:
        return connection.base_url()
    except Exception as exc:
        return f"unavailable ({exc})"


def _doctor_report(config: Path) -> dict[str, Any]:
    settings = connection.settings(config)
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
        report["paths"][name] = {"path": str(path.resolve()), "writable": os.access(path, os.W_OK)}
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
