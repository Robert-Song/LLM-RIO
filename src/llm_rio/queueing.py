from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from llm_rio.errors import QueueFullError


@dataclass(slots=True)
class QueuedRequest:
    id: str
    model_id: str
    tenant_id: str
    estimated_tokens: int
    payload: dict[str, Any]
    reservation_id: str
    is_stream: bool = False
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    assignment: asyncio.Future[Any] | None = None


class DeficitRoundRobinQueue:
    """Optionally bounded per-tenant DRR admission queue for one model."""

    def __init__(
        self,
        total_capacity: int | None,
        tenant_capacity: int | None,
        quantum: int = 4096,
    ) -> None:
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        if any(value is not None and value <= 0 for value in (total_capacity, tenant_capacity)):
            raise ValueError("queue capacities must be positive or None")
        self.total_capacity = total_capacity
        self.tenant_capacity = tenant_capacity
        self.quantum = quantum
        self._queues: OrderedDict[str, deque[QueuedRequest]] = OrderedDict()
        self._deficits: dict[str, int] = {}
        self._size = 0
        self._active_tenant: str | None = None

    def __len__(self) -> int:
        return self._size

    @property
    def tenants(self) -> int:
        return len(self._queues)

    @property
    def estimated_token_work(self) -> int:
        return sum(item.estimated_tokens for queue in self._queues.values() for item in queue)

    @property
    def oldest_enqueued_at(self) -> datetime | None:
        values = [queue[0].enqueued_at for queue in self._queues.values() if queue]
        return min(values) if values else None

    def put(self, request: QueuedRequest) -> None:
        if request.estimated_tokens < 0:
            raise ValueError("request token cost must be nonnegative")
        tenant_queue = self._queues.setdefault(request.tenant_id, deque())
        if (self.total_capacity is not None and self._size >= self.total_capacity) or (
            self.tenant_capacity is not None and len(tenant_queue) >= self.tenant_capacity
        ):
            if not tenant_queue:
                self._queues.pop(request.tenant_id, None)
            raise QueueFullError()
        tenant_queue.append(request)
        self._deficits.setdefault(request.tenant_id, 0)
        self._size += 1

    def pop(self) -> QueuedRequest | None:
        if not self._size:
            return None
        if self._active_tenant is None:
            # Skip complete rounds in which no head request can fit. Large requests
            # should not make admission spin cost/quantum times on the event loop.
            skipped_rounds = min(
                max(0, (queue[0].estimated_tokens - self._deficits[tenant] - 1) // self.quantum)
                for tenant, queue in self._queues.items()
            )
            if skipped_rounds:
                for tenant in self._deficits:
                    self._deficits[tenant] += skipped_rounds * self.quantum
        while self._queues:
            tenant_id = next(iter(self._queues))
            tenant_queue = self._queues[tenant_id]
            if self._active_tenant != tenant_id:
                self._deficits[tenant_id] += self.quantum
                self._active_tenant = tenant_id
            request = tenant_queue[0]
            if request.estimated_tokens > self._deficits[tenant_id]:
                self._queues.move_to_end(tenant_id)
                self._active_tenant = None
                continue
            tenant_queue.popleft()
            self._deficits[tenant_id] -= request.estimated_tokens
            self._size -= 1
            if not tenant_queue:
                self._queues.pop(tenant_id)
                self._deficits.pop(tenant_id)
                self._active_tenant = None
            elif tenant_queue[0].estimated_tokens > self._deficits[tenant_id]:
                self._queues.move_to_end(tenant_id)
                self._active_tenant = None
            # Continue the same tenant's turn on the next pop, using its remaining
            # deficit. Granting fresh credit for every request breaks token fairness.
            return request
        return None

    def remove(self, request_id: str) -> QueuedRequest | None:
        for tenant_id, tenant_queue in list(self._queues.items()):
            for request in tenant_queue:
                if request.id != request_id:
                    continue
                tenant_queue.remove(request)
                self._size -= 1
                if not tenant_queue:
                    self._queues.pop(tenant_id, None)
                    self._deficits.pop(tenant_id, None)
                    if self._active_tenant == tenant_id:
                        self._active_tenant = None
                return request
        return None

    def drain(self) -> list[QueuedRequest]:
        requests = [request for queue in self._queues.values() for request in queue]
        self._queues.clear()
        self._deficits.clear()
        self._size = 0
        self._active_tenant = None
        return requests


class ModelQueues:
    def __init__(self, total_capacity: int | None, tenant_capacity: int | None) -> None:
        self.total_capacity = total_capacity
        self.tenant_capacity = tenant_capacity
        self._models: dict[str, DeficitRoundRobinQueue] = {}

    def for_model(self, model_id: str) -> DeficitRoundRobinQueue:
        if model_id not in self._models:
            self._models[model_id] = DeficitRoundRobinQueue(
                self.total_capacity, self.tenant_capacity
            )
        return self._models[model_id]

    def pending_models(self) -> list[str]:
        return [model_id for model_id, queue in self._models.items() if len(queue)]

    def remove(self, model_id: str, request_id: str) -> QueuedRequest | None:
        queue = self._models.get(model_id)
        return queue.remove(request_id) if queue else None

    def drain_all(self) -> list[QueuedRequest]:
        result: list[QueuedRequest] = []
        for queue in self._models.values():
            result.extend(queue.drain())
        return result
