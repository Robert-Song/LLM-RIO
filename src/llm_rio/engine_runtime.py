"""Native runtime metadata. Experimental implementation is imported only on selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KVCachedMode = Literal["none", "auto", "required"]


class KVCachedCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KVCachedRuntime:
    enabled: bool
    package_version: str | None
    vllm_version: str | None
    officially_tested: bool
    reason: str
    source_revision: str | None = None

    @property
    def memory_backend(self) -> str:
        return "kvcached" if self.enabled else "native"

    def environment(self, *, pythonpath: str | None = None) -> dict[str, str]:
        if self.enabled:
            from llm_rio.modes.kv_cached.runtime import KVCachedRuntime as ExperimentalRuntime

            return ExperimentalRuntime(
                self.enabled,
                self.package_version,
                self.vllm_version,
                self.officially_tested,
                self.reason,
                self.source_revision,
            ).environment(pythonpath=pythonpath)
        return {}


def detect_kvcached(mode: str | None) -> KVCachedRuntime:
    if mode in (None, "none", "", "disabled"):
        return KVCachedRuntime(False, None, None, False, "disabled_by_configuration")
    from llm_rio.modes.kv_cached.runtime import detect_kvcached as detect

    result = detect(mode)
    return KVCachedRuntime(
        result.enabled,
        result.package_version,
        result.vllm_version,
        result.officially_tested,
        result.reason,
        result.source_revision,
    )


def add_kvcached_vllm_flags(command: list[str], runtime: KVCachedRuntime) -> None:
    if runtime.enabled:
        from llm_rio.modes.kv_cached.runtime import add_kvcached_vllm_flags as add

        add(command, runtime)  # type: ignore[arg-type]
