"""Unit tests for the deficit round-robin admission queue."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llm_rio.errors import QueueFullError
from llm_rio.queueing import DeficitRoundRobinQueue, ModelQueues, QueuedRequest


def make_request(tenant_id: str, cost: int, index: int = 0) -> QueuedRequest:
    return QueuedRequest(
        id=f"{tenant_id}-{index}",
        model_id="model-a",
        tenant_id=tenant_id,
        estimated_tokens=cost,
        payload={},
        reservation_id=f"res-{tenant_id}-{index}",
    )


class TestDeficitRoundRobinQueue:
    def test_empty_queue_pops_none(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=10, tenant_capacity=5)
        assert queue.pop() is None
        assert len(queue) == 0

    def test_fair_interleaving_between_tenants(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50, quantum=100)
        for index in range(3):
            queue.put(make_request("alice", 100, index))
            queue.put(make_request("bob", 100, index))
        served = [queue.pop().tenant_id for _ in range(6)]
        assert served == ["alice", "bob", "alice", "bob", "alice", "bob"]

    def test_large_request_rotates_until_eligible(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50, quantum=100)
        queue.put(make_request("heavy", 350, 0))
        queue.put(make_request("light", 100, 0))
        # The light tenant is served first while heavy accrues deficit.
        assert queue.pop().tenant_id == "light"
        # Heavy needs four rounds of quantum (100, 200, 300, 400) before it is eligible.
        served = queue.pop()
        assert served is not None and served.tenant_id == "heavy"

    def test_tenant_capacity_is_bounded(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=2)
        queue.put(make_request("alice", 10, 0))
        queue.put(make_request("alice", 10, 1))
        with pytest.raises(QueueFullError):
            queue.put(make_request("alice", 10, 2))
        assert len(queue) == 2

    def test_total_capacity_is_bounded(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=3, tenant_capacity=100)
        queue.put(make_request("alice", 10, 0))
        queue.put(make_request("bob", 10, 0))
        queue.put(make_request("carol", 10, 0))
        with pytest.raises(QueueFullError):
            queue.put(make_request("dave", 10, 0))
        assert len(queue) == 3

    def test_remove_returns_and_removes_request(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50)
        request = make_request("alice", 10, 0)
        queue.put(request)
        queue.put(make_request("alice", 10, 1))
        assert queue.remove(request.id) is request
        assert queue.remove(request.id) is None
        assert len(queue) == 1

    def test_drain_empties_everything(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50)
        for index in range(4):
            queue.put(make_request("alice", 10, index))
        queue.put(make_request("bob", 10, 0))
        drained = queue.drain()
        assert len(drained) == 5
        assert len(queue) == 0
        assert queue.tenants == 0
        assert queue.drain() == []

    def test_oldest_enqueued_at_across_tenants(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50)
        older = make_request("alice", 10, 0)
        older.enqueued_at = datetime(2026, 1, 1, tzinfo=UTC)
        newer = make_request("bob", 10, 0)
        newer.enqueued_at = datetime(2026, 1, 2, tzinfo=UTC)
        queue.put(older)
        queue.put(newer)
        assert queue.oldest_enqueued_at == older.enqueued_at

    def test_estimated_token_work(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50)
        queue.put(make_request("alice", 120, 0))
        queue.put(make_request("bob", 80, 0))
        assert queue.estimated_token_work == 200

    def test_tenant_count_tracks_active_queues(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=100, tenant_capacity=50)
        assert queue.tenants == 0
        queue.put(make_request("alice", 10, 0))
        queue.put(make_request("bob", 10, 0))
        assert queue.tenants == 2
        queue.pop()
        queue.pop()
        assert queue.tenants == 0

    def test_empty_tenant_entry_cleaned_on_rejected_put(self) -> None:
        queue = DeficitRoundRobinQueue(total_capacity=1, tenant_capacity=1)
        queue.put(make_request("alice", 10, 0))
        with pytest.raises(QueueFullError):
            queue.put(make_request("bob", 10, 0))
        assert queue.tenants == 1


class TestModelQueues:
    def test_for_model_reuses_queue(self) -> None:
        queues = ModelQueues(total_capacity=10, tenant_capacity=5)
        assert queues.for_model("m1") is queues.for_model("m1")

    def test_pending_models_only_lists_nonempty(self) -> None:
        queues = ModelQueues(total_capacity=10, tenant_capacity=5)
        assert queues.pending_models() == []
        queues.for_model("m1").put(make_request("alice", 10, 0))
        assert queues.pending_models() == ["m1"]
        queues.for_model("m2")
        assert queues.pending_models() == ["m1"]

    def test_remove_by_model_and_request(self) -> None:
        queues = ModelQueues(total_capacity=10, tenant_capacity=5)
        request = make_request("alice", 10, 0)
        queues.for_model("m1").put(request)
        assert queues.remove("m1", request.id) is request
        assert queues.remove("m1", request.id) is None
        assert queues.remove("m2", request.id) is None

    def test_drain_all(self) -> None:
        queues = ModelQueues(total_capacity=10, tenant_capacity=5)
        queues.for_model("m1").put(make_request("alice", 10, 0))
        queues.for_model("m2").put(make_request("bob", 10, 0))
        assert len(queues.drain_all()) == 2
        assert queues.pending_models() == []
