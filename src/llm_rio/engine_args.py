"""Serialize extra engine options identically during validation and serving."""

from __future__ import annotations

import json
from typing import Any


def extra_engine_arguments(
    arguments: dict[str, Any], *, explicit_false: bool = True
) -> list[str]:
    result: list[str] = []
    for key, value in arguments.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                result.append(flag)
            elif explicit_false:
                result.append(f"--no-{key.replace('_', '-')}")
        elif isinstance(value, list):
            for item in value:
                result.extend([flag, str(item)])
        elif isinstance(value, dict):
            result.extend([flag, json.dumps(value, separators=(",", ":"), sort_keys=True)])
        elif value is not None:
            result.extend([flag, str(value)])
    return result
