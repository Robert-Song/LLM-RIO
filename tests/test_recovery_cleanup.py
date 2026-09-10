from unittest.mock import AsyncMock

import pytest

from llm_rio import recovery
from llm_rio.process_cleanup import TeardownError


@pytest.mark.parametrize("verified", [True, False])
async def test_recovery_requires_verified_teardown(monkeypatch, verified):
    database = AsyncMock()
    database.fetchall.return_value = [
        {"id": "w", "pid": 991122, "port": 19000, "gpu_uuids_json": '["GPU-0"]'}
    ]
    monkeypatch.setattr(recovery, "_matching_managed_process", lambda *_: True)
    terminate = AsyncMock(side_effect=None if verified else TeardownError("GPU still owned"))
    monkeypatch.setattr(recovery, "terminate_engine", terminate)
    if verified:
        result = await recovery.terminate_recorded_workers(database)
        assert result[0]["action"] == "terminated"
    else:
        with pytest.raises(TeardownError):
            await recovery.terminate_recorded_workers(database)
    assert terminate.call_args.kwargs["gpu_uuids"] == ("GPU-0",)


async def test_recovery_does_not_signal_unidentified_surviving_group(monkeypatch):
    database = AsyncMock()
    database.fetchall.return_value = [
        {"id": "w", "pid": 991122, "port": 19000, "gpu_uuids_json": '["GPU-0"]'}
    ]
    monkeypatch.setattr(recovery, "_matching_managed_process", lambda *_: False)
    monkeypatch.setattr(recovery, "group_members", lambda _: {991123: "S"})
    terminate = AsyncMock()
    monkeypatch.setattr(recovery, "terminate_engine", terminate)
    with pytest.raises(TeardownError, match="identity cannot be verified"):
        await recovery.terminate_recorded_workers(database)
    terminate.assert_not_awaited()
