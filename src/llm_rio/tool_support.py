from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VllmParserConfiguration:
    """vLLM parser choices inferred from a model's shipped metadata."""

    tool_parser: str | None = None
    reasoning_parser: str | None = None


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


def _model_type(model_path: Path) -> str:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return ""
    try:
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    model_type = raw.get("model_type")
    return model_type.lower() if isinstance(model_type, str) else ""


def detect_vllm_parser_configuration(model_path: str | Path) -> VllmParserConfiguration:
    """Infer only parser pairs that are known to match the shipped model format.

    Unknown formats deliberately produce no parser instead of guessing from a
    generic ``<think>`` tag. This keeps generated text intact while making
    support for each new format an explicit, tested compatibility decision.
    """
    path = Path(model_path)
    template = _chat_template(path)
    if "tools" not in template:
        tool_parser = None
    elif "<|tool_call>" in template and "call:" in template:
        tool_parser = "gemma4"
    elif "<tool_call>" in template and "<function=" in template:
        tool_parser = "qwen3_xml"
    elif all(marker in template for marker in ("<tool_call>", "<arg_key>", "<arg_value>")):
        tool_parser = "poolside_v1"
    else:
        tool_parser = None

    model_type = _model_type(path)
    has_qwen_thinking_tags = "<think>" in template and "</think>" in template
    if model_type.startswith("qwen3") and has_qwen_thinking_tags:
        reasoning_parser = "qwen3"
    elif model_type == "gemma4" and "<|think|>" in template:
        reasoning_parser = "gemma4"
    elif model_type == "laguna" and has_qwen_thinking_tags:
        reasoning_parser = "poolside_v1"
    else:
        reasoning_parser = None

    return VllmParserConfiguration(
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
    )


def detect_vllm_tool_parser(model_path: str | Path) -> str | None:
    """Return the vLLM tool parser matching a recognized chat template."""
    return detect_vllm_parser_configuration(model_path).tool_parser


def detect_vllm_reasoning_parser(model_path: str | Path) -> str | None:
    """Return the vLLM reasoning parser for a recognized reasoning format."""
    return detect_vllm_parser_configuration(model_path).reasoning_parser
