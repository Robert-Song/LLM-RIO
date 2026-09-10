from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_KIB_PER_MIB = 1024


@dataclass(frozen=True, slots=True)
class ProcessMemorySample:
    rss_mib: float
    pss_mib: float
    swap_mib: float
    swap_pss_mib: float
    process_count: int
    source: str

    @property
    def accounted_mib(self) -> float:
        return self.pss_mib if self.pss_mib > 0 else self.rss_mib


@dataclass(frozen=True, slots=True)
class HostMemorySample:
    source: str
    effective_total_mib: float
    current_mib: float
    available_mib: float
    swap_used_mib: float
    swap_total_mib: float


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _parse_kib_fields(path: Path) -> dict[str, int]:
    raw = _read_text(path)
    if raw is None:
        return {}
    result: dict[str, int] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        token = value.strip().split(maxsplit=1)[0]
        try:
            result[key] = int(token)
        except ValueError:
            continue
    return result


def _process_group_id(pid: int) -> int | None:
    raw = _read_text(Path("/proc") / str(pid) / "stat")
    if not raw:
        return None
    end = raw.rfind(")")
    if end < 0:
        return None
    fields = raw[end + 1 :].split()
    try:
        # Fields after the command start with state, ppid, then process group.
        return int(fields[2])
    except (IndexError, ValueError):
        return None


def sample_process_group_memory(pid: int | None) -> ProcessMemorySample:
    """Account for the API parent and every vLLM/TP child in its process group."""
    if pid is None or pid <= 0:
        return ProcessMemorySample(0, 0, 0, 0, 0, "unavailable")
    process_group = _process_group_id(pid) or pid
    totals = {"Rss": 0, "Pss": 0, "Swap": 0, "SwapPss": 0}
    count = 0
    proc_root = Path("/proc")
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        entries = ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        member_pid = int(entry.name)
        if _process_group_id(member_pid) != process_group:
            continue
        fields = _parse_kib_fields(entry / "smaps_rollup")
        if not fields:
            continue
        count += 1
        for key in totals:
            totals[key] += fields.get(key, 0)
    source = "smaps_rollup" if count else "unavailable"
    return ProcessMemorySample(
        rss_mib=totals["Rss"] / _KIB_PER_MIB,
        pss_mib=totals["Pss"] / _KIB_PER_MIB,
        swap_mib=totals["Swap"] / _KIB_PER_MIB,
        swap_pss_mib=totals["SwapPss"] / _KIB_PER_MIB,
        process_count=count,
        source=source,
    )


def _host_meminfo() -> HostMemorySample:
    fields = _parse_kib_fields(Path("/proc/meminfo"))
    total = fields.get("MemTotal", 0) / _KIB_PER_MIB
    available = fields.get("MemAvailable", fields.get("MemFree", 0)) / _KIB_PER_MIB
    swap_total = fields.get("SwapTotal", 0) / _KIB_PER_MIB
    swap_free = fields.get("SwapFree", 0) / _KIB_PER_MIB
    return HostMemorySample(
        source="meminfo_fallback",
        effective_total_mib=total,
        current_mib=max(0.0, total - available),
        available_mib=available,
        swap_used_mib=max(0.0, swap_total - swap_free),
        swap_total_mib=swap_total,
    )


def _cgroup_v2_path() -> Path | None:
    raw = _read_text(Path("/proc/self/cgroup"))
    if raw is None:
        return None
    for line in raw.splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            return Path("/sys/fs/cgroup") / relative.lstrip("/")
    return None


def _bytes_value(path: Path) -> int | None:
    raw = _read_text(path)
    if raw is None or raw in {"", "max"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def sample_host_memory() -> HostMemorySample:
    """Prefer an effective cgroup-v2 limit, with host meminfo as fallback."""
    host = _host_meminfo()
    cgroup = _cgroup_v2_path()
    if cgroup is None:
        return host
    current_bytes = _bytes_value(cgroup / "memory.current")
    maximum_bytes = _bytes_value(cgroup / "memory.max")
    if current_bytes is None or maximum_bytes is None or maximum_bytes <= 0:
        return host
    current = current_bytes / (1024 * 1024)
    maximum = maximum_bytes / (1024 * 1024)
    cgroup_available = max(0.0, maximum - current)
    swap_current = (_bytes_value(cgroup / "memory.swap.current") or 0) / (1024 * 1024)
    swap_maximum_bytes = _bytes_value(cgroup / "memory.swap.max")
    swap_maximum = (
        host.swap_total_mib if swap_maximum_bytes is None else swap_maximum_bytes / (1024 * 1024)
    )
    return HostMemorySample(
        source="cgroup",
        effective_total_mib=min(maximum, host.effective_total_mib or maximum),
        current_mib=current,
        available_mib=min(cgroup_available, host.available_mib or cgroup_available),
        swap_used_mib=swap_current,
        swap_total_mib=swap_maximum,
    )


def gib_to_mib(value: float | None) -> float | None:
    return None if value is None else value * 1024
