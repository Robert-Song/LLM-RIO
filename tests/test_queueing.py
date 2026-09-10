from __future__ import annotations

import pytest

from llm_rio.queueing import DeficitRoundRobinQueue, QueuedRequest


def request(tenant: str, index: int, cost: int) -> QueuedRequest:
    return QueuedRequest(f"{tenant}-{index}", "model", tenant, cost, {}, f"reserve-{index}")


def test_drr_allocates_equal_token_work_to_backlogged_tenants() -> None:
    queue = DeficitRoundRobinQueue(None, None, quantum=10)
    for i in range(100):
        queue.put(request("small", i, 1))
        queue.put(request("large", i, 10))
    work = {"small": 0, "large": 0}
    for _ in range(11):
        item = queue.pop()
        assert item is not None
        work[item.tenant_id] += item.estimated_tokens
    assert work == {"small": 10, "large": 10}


def test_large_request_accumulates_credit_without_starving_small_requests() -> None:
    queue = DeficitRoundRobinQueue(None, None, quantum=10)
    queue.put(request("large", 0, 25))
    for i in range(50):
        queue.put(request("small", i, 1))
    work = 0
    while (item := queue.pop()) is not None:
        if item.tenant_id == "large":
            break
        work += item.estimated_tokens
    assert work == 20


def test_huge_requests_skip_empty_rounds() -> None:
    queue = DeficitRoundRobinQueue(None, None, quantum=1)
    queue.put(request("large", 1, 1))
    queue.put(request("large", 0, 10**12))
    assert queue.pop().estimated_tokens == 1
    assert queue.pop().estimated_tokens == 10**12
    assert len(queue) == 0


def test_remove_active_tenant_and_drain_reset_credit() -> None:
    queue = DeficitRoundRobinQueue(None, None, quantum=10)
    queue.put(request("a", 0, 1))
    queue.put(request("a", 1, 1))
    assert queue.pop().id == "a-0"
    assert queue.remove("a-1").id == "a-1"
    assert queue.tenants == 0
    queue.put(request("b", 0, 10))
    assert [item.id for item in queue.drain()] == ["b-0"]
    assert queue.pop() is None
    queue.put(request("a", 2, 10))
    assert queue.pop().id == "a-2"


@pytest.mark.parametrize(
    "kwargs", [{"quantum": 0}, {"quantum": -1}, {"total_capacity": 0}, {"tenant_capacity": -1}]
)
def test_invalid_queue_settings_fail_instead_of_hanging(kwargs) -> None:
    with pytest.raises(ValueError):
        DeficitRoundRobinQueue(**{"total_capacity": None, "tenant_capacity": None, **kwargs})
