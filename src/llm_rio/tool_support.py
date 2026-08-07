from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _chat_template(model_path: Path) -> str:
    template_path = model_path / "chat_template.jinja"
    if template_path.is_file():
        return template_path.read_text(encoding="utf-8", errors="replace")
    tokenizer_path = model_path / "tokenizer_config.json"
    if not tokenizer_path.is_file():
        return ""
    try:
        raw: dict[str, Any] = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    template = raw.get("chat_template")
    if isinstance(template, str):
        return template
    return json.dumps(template) if template is not None else ""


def detect_vllm_tool_parser(model_path: str | Path) -> str | None:
    """Return the vLLM parser matching a recognized tool-capable chat template."""
    template = _chat_template(Path(model_path))
    if "tools" not in template:
        return None
    if "<|tool_call>" in template and "call:" in template:
        return "gemma4"
    if "<tool_call>" in template and "<function=" in template:
        return "qwen3_xml"
    if all(marker in template for marker in ("<tool_call>", "<arg_key>", "<arg_value>")):
        return "poolside_v1"
    return None
