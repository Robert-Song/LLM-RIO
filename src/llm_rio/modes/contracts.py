from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from llm_rio.domain import PlacementProfile, WorkerPlacement
from llm_rio.modes.actions import PlannerAction, QueuePressure


@dataclass(frozen=True, slots=True)
class ProfileEligibility:
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ModeCapabilities:
    name: str
    experimental: bool
    engines: tuple[str, ...]
    sleep: bool
    validation_requires_maintenance: bool


class PlacementPlanner(Protocol):
    def plan(
        self,
        *,
        now: datetime,
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
        pressures: list[QueuePressure],
        profiles: dict[str, list[PlacementProfile]],
    ) -> list[PlannerAction]: ...


class ModePolicy(Protocol):
    @property
    def capabilities(self) -> ModeCapabilities: ...

    @property
    def planner(self) -> PlacementPlanner: ...

    def eligibility(self, profile: PlacementProfile) -> ProfileEligibility: ...
