"""Check native acceptance DESIGN coverage against the current public surfaces.

This does not run acceptance cases and cannot qualify a release.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import click
import typer
from fastapi.routing import APIRoute

from llm_rio.api.app import create_app
from llm_rio.cli import app as cli
from llm_rio.config import EngineSettings, Settings
from llm_rio.modes.vllm_sleep.settings import SleepSettings

ROOT = Path(__file__).resolve().parents[1]


def commands(command: click.Command, prefix: str = "") -> set[str]:
    if isinstance(command, click.Group):
        return {
            leaf
            for name, child in command.commands.items()
            for leaf in commands(child, f"{prefix} {name}".strip())
        }
    return {prefix}


def current_surfaces() -> dict[str, set[str]]:
    app = create_app(
        Settings(serving_mode="queue", config_file=ROOT / "acceptance-absent.toml", _env_file=None)
    )
    api = {
        f"{method} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    buttons: set[str] = set()
    for path in [ROOT / "src/llm_rio/tui.py", *(ROOT / "src/llm_rio/ui").glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Button"
            ):
                for argument in node.keywords:
                    if argument.arg == "id" and isinstance(argument.value, ast.Constant):
                        buttons.add(str(argument.value.value))
    settings = set(Settings.model_fields) - {"engines", "modes"}
    settings.update(f"engines.{name}" for name in EngineSettings.model_fields)
    from llm_rio.modes.queue.settings import QueueSettings

    settings.update(f"modes.queue.{name}" for name in QueueSettings.model_fields)
    settings.update(f"modes.vllm_sleep.{name}" for name in SleepSettings.model_fields)
    return {
        "api": api,
        "cli": commands(typer.main.get_command(cli)),
        "tui": buttons,
        "settings": settings,
    }


def main() -> None:
    for name in list(os.environ):
        if name.startswith("LLMRIO_"):
            del os.environ[name]
    plan = json.loads((ROOT / "docs/release/native-acceptance.json").read_text())
    errors: list[str] = []
    cases = plan["cases"]
    ids = {case["id"] for case in cases}
    if len(ids) != len(cases):
        errors.append("Duplicate acceptance case ID")
    for case in cases:
        if not set(case["modes"]) <= {"queue", "vllm-sleep"} or not case["modes"]:
            errors.append(f"{case['id']}: invalid native scope")
        for field in ("actor", "requires", "steps", "expected", "evidence", "lane"):
            if not case.get(field):
                errors.append(f"{case['id']}: missing {field}")
    actual = current_surfaces()
    for kind, items in actual.items():
        planned = set(plan["surfaces"][kind])
        for missing in sorted(items - planned):
            errors.append(f"Unplanned {kind}: {missing}")
        for stale in sorted(planned - items):
            errors.append(f"Stale {kind}: {stale}")
    units = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/llm_rio").rglob("*.py")
        if "kv_cached" not in path.parts
    }
    planned_units = set(plan["implementation_units"])
    for missing in sorted(units - planned_units):
        errors.append(f"Unplanned implementation unit: {missing}")
    for stale in sorted(planned_units - units):
        errors.append(f"Stale implementation unit: {stale}")
    for group in [
        *plan["surfaces"].values(),
        plan["requirements"],
        plan["findings"],
        plan["implementation_units"],
    ]:
        for item, references in group.items():
            if not references or set(references) - ids:
                errors.append(f"{item}: missing or unknown case references {references}")
    readable = (ROOT / "docs/release/ACCEPTANCE_CASES.md").read_text()
    if set(re.findall(r"\b[A-Z]+-\d{2}\b", readable)) != ids:
        errors.append("Readable case matrix and machine-readable IDs disagree")
    for name in ("queue", "vllm_sleep"):
        forbidden = {"queue", "vllm_sleep", "kv_cached"} - {name}
        for path in (ROOT / "src/llm_rio/modes" / name).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                imports = []
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                if any(
                    module.startswith(f"llm_rio.modes.{other}")
                    for module in imports
                    for other in forbidden
                ):
                    errors.append(f"Cross-mode import: {path.relative_to(ROOT)}:{node.lineno}")
    if errors:
        raise SystemExit("\n".join(errors))
    counts = ", ".join(f"{kind}={len(values)}" for kind, values in actual.items())
    print(
        f"Acceptance design coverage: {len(cases)} cases; {counts}; "
        f"{len(units)} native modules. Cases NOT executed."
    )


if __name__ == "__main__":
    main()
