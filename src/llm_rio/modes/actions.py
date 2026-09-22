from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from llm_rio.domain import PlacementProfile


@dataclass(frozen=True, slots=True)
class QueuePressure:
    model_id: str
    requests: int
    estimated_tokens: int
    oldest_enqueued_at: datetime
    preload: bool = False
    desired_workers: int = 1


@dataclass(frozen=True, slots=True)
class StartPlacement:
    profile: PlacementProfile
    gpu_uuids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class DrainPlacement:
    worker_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class SleepPlacement:
    worker_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class WakePlacement:
    worker_id: str
    reason: str


PlannerAction = StartPlacement | DrainPlacement | SleepPlacement | WakePlacement
