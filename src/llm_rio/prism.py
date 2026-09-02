from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

KVCachedMode = Literal["disabled", "auto", "required"]
KVCACHED_COMPAT_REVISION = "60cad949389af6bbf1d65c4eddf325113df5a9eb"


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
        if not self.enabled:
            return {}
        environment = {
            "ENABLE_KVCACHED": "true",
            "KVCACHED_AUTOPATCH": "1",
            "KVCACHED_CONTIGUOUS_LAYOUT": "false",
            "KVCACHED_PAGE_SIZE_MB": "4",
            "KVCACHED_MAX_CACHED_TOKENS": "0",
            "KVCACHED_MIN_RESERVED_PAGES": "0",
            "KVCACHED_MAX_RESERVED_PAGES": "0",
            "KVCACHED_PAGE_PREALLOC_ENABLED": "false",
            "NCCL_NET_PLUGIN": "none",
            "NCCL_NET": "Socket",
        }
        if self.vllm_version is not None and _release_tuple(self.vllm_version) >= (0, 26, 0):
            # kvcached's worker/allocation patches target the legacy model runner.
            # vLLM 0.26 defaults some models to the new V2 runner, whose class lives
            # at a different import path and therefore bypasses those patches.
            environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        else:
            environment["VLLM_USE_V1"] = "1"
        if not self.officially_tested:
            bootstrap = Path(__file__).with_name("kvcached_bootstrap")
            inherited = pythonpath if pythonpath is not None else os.environ.get("PYTHONPATH")
            environment["LLM_RIO_KVCACHED_VLLM026_SHIM"] = "1"
            environment["PYTHONPATH"] = (
                str(bootstrap) if not inherited else f"{bootstrap}{os.pathsep}{inherited}"
            )
        return environment


def _release_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _source_revision(distribution: importlib.metadata.Distribution) -> str | None:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        direct_url = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    return str(commit_id) if commit_id else None


def detect_kvcached(mode: KVCachedMode) -> KVCachedRuntime:
    if mode == "disabled":
        return KVCachedRuntime(False, None, None, False, "disabled_by_configuration")
    try:
        distribution = importlib.metadata.distribution("kvcached")
        package_version = distribution.version
    except importlib.metadata.PackageNotFoundError as exc:
        if mode == "required":
            raise KVCachedCompatibilityError(
                "kvcached mode requires kvcached; install the legacy-named 'prism' extra"
            ) from exc
        return KVCachedRuntime(False, None, None, False, "kvcached_not_installed")
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        if mode == "required":
            raise KVCachedCompatibilityError(
                "kvcached mode requires vLLM in the same environment as kvcached"
            ) from exc
        return KVCachedRuntime(False, package_version, None, False, "vllm_not_installed")

    source_revision = _source_revision(distribution)
    vllm_release = _release_tuple(vllm_version)
    officially_tested = (0, 8, 4) <= vllm_release <= (0, 24, 0)
    experimental_vllm = (0, 26, 0) <= vllm_release < (0, 27, 0)
    if not officially_tested and (
        not experimental_vllm or source_revision != KVCACHED_COMPAT_REVISION
    ):
        message = (
            f"vLLM {vllm_version} is outside the tested Prism compatibility set; "
            f"the experimental path requires vLLM 0.26.x and kvcached revision "
            f"{KVCACHED_COMPAT_REVISION}"
        )
        if mode == "required":
            raise KVCachedCompatibilityError(message)
        return KVCachedRuntime(
            False,
            package_version,
            vllm_version,
            False,
            "incompatible_kvcached_source",
            source_revision,
        )
    runtime = KVCachedRuntime(
        True,
        package_version,
        vllm_version,
        officially_tested,
        ("compatible_package_set" if officially_tested else "experimental_compatibility_revision"),
        source_revision,
    )
    if not officially_tested:
        logger.warning(
            "Enabling experimental kvcached %s integration with untested vLLM %s",
            package_version,
            vllm_version,
        )
    return runtime


def add_kvcached_vllm_flags(command: list[str], runtime: KVCachedRuntime) -> None:
    if not runtime.enabled:
        return
    if "--enable-prefix-caching" in command:
        raise KVCachedCompatibilityError("kvcached profiles cannot enable vLLM prefix caching")
    if "--no-enable-prefix-caching" not in command:
        command.append("--no-enable-prefix-caching")
