from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from test_profile_admin_contract import make_profile

from llm_rio.database_rebuild import RebuildError, rebuild_database
from llm_rio.domain import CatalogState, Role
from llm_rio.profiles import ProfileRepository, profile_key, profile_to_dict
from llm_rio.security import Principal, default_key_vault_path
from llm_rio.storage import Database, _now

SECRET = "rio_test_prefix_123456789_secret"
PRINCIPAL = Principal("key", "user", Role.USER, "account", False)


@pytest.fixture
async def source(tmp_path: Path) -> Path:
    path = tmp_path / "original.db"
    db = Database(path)
    await db.open()
    try:
        await db.create_key(
            key_id="key",
            nickname="user",
            role=Role.USER,
            account_id="account",
            account_nickname="account",
            prefix=SECRET[:24],
            api_key=SECRET,
            limit_tokens=1000,
            unlimited=False,
        )
        model_id, job_id = await db.create_model_job(
            nickname="model",
            repo="org/model",
            revision="revision",
            creator_key_id="key",
            grant_key_ids=["key"],
        )
        await db.update_model_job(
            job_id,
            job_state="COMPLETED",
            stage="complete",
            catalog_state=CatalogState.AVAILABLE,
            artifact_path="/models/existing",
            resolved_revision="revision",
            progress={"validation": "passed"},
        )
        await db.execute("INSERT INTO model_grants VALUES (?, ?, ?)", ("key", model_id, _now()))
        profile = make_profile("profile", model_id, ("GPU-0",))
        await ProfileRepository(db, "machine").save(profile, profile_key(profile_to_dict(profile)))
        await db.execute(
            "INSERT INTO model_verification_jobs "
            "(id,model_id,backend,state,stage,created_at,updated_at) "
            "VALUES ('verification',?,'kvcached','COMPLETED','complete',?,?)",
            (model_id, _now(), _now()),
        )
        await db.set_machine_fingerprint("machine")
        await db.record_event("MODEL_VALIDATED", model_id, {"passed": True})
        reservation = await db.reserve_quota(
            request_id="old",
            idempotency_hash="old",
            principal=PRINCIPAL,
            model_id=model_id,
            estimated_tokens=100,
        )
        await db.settle_quota(reservation_id=reservation, actual_tokens=25)
        await db.reset_usage("key")
        await db.reserve_quota(
            request_id="unfinished",
            idempotency_hash="unfinished",
            principal=PRINCIPAL,
            model_id=model_id,
            estimated_tokens=100,
        )
    finally:
        await db.close()
    return path


def read_table(path: Path, table: str) -> list[tuple]:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()


async def test_rebuild_preserves_identity_and_validation_and_future_summary(
    source: Path, tmp_path: Path
) -> None:
    original_bytes = source.read_bytes()
    vault_bytes = default_key_vault_path(source).read_bytes()
    destination = tmp_path / "new state" / "clean.db"
    report = rebuild_database(source, destination=destination, server_stopped=True)
    assert source.read_bytes() == original_bytes
    assert default_key_vault_path(source).read_bytes() == vault_bytes
    assert default_key_vault_path(destination).read_bytes() == vault_bytes
    assert destination.stat().st_mode & 0o777 == 0o600
    assert default_key_vault_path(destination).stat().st_mode & 0o777 == 0o600
    assert report["active_keys_verified"] == 1
    for table in (
        "api_keys",
        "model_catalog",
        "model_grants",
        "model_profiles",
        "model_jobs",
        "model_verification_jobs",
        "runtime_events",
    ):
        assert read_table(source, table) == read_table(destination, table)
    for table in ("inference_requests", "quota_reservations", "quota_ledger", "workers"):
        assert not read_table(destination, table)
    db = Database(destination)
    await db.open()
    try:
        assert await db.authenticate(SECRET[:24], SECRET) is not None
        usage = await db.usage(PRINCIPAL)
        assert usage["balance_tokens"] == 1000
        assert usage["used_tokens"] == usage["lifetime_charged_tokens"] == 0
        model = await db.model_by_nickname("model")
        assert model is not None
        assert model["artifact_path"] == "/models/existing"
        assert len(await ProfileRepository(db, "machine").for_model(model["id"])) == 1
        reservation = await db.reserve_quota(
            request_id="new",
            idempotency_hash="new",
            principal=PRINCIPAL,
            model_id=model["id"],
            estimated_tokens=50,
        )
        await db.settle_quota(
            reservation_id=reservation, actual_tokens=30, prompt_tokens=20, completion_tokens=10
        )
        summary = await db.summarize_usage()
        assert summary["summarized_requests"] == 1
        usage = await db.usage(PRINCIPAL)
        assert usage["used_tokens"] == usage["lifetime_charged_tokens"] == 30
        assert usage["balance_tokens"] == 970
        assert (await db.dashboard_usage())["total"]["token_usage"] == 30
        assert await db.fetchall("PRAGMA foreign_key_check") == []
    finally:
        await db.close()


def test_rebuild_check_only_preserves_source_and_vault(source: Path) -> None:
    before = source.read_bytes()
    vault = default_key_vault_path(source)
    before_vault = vault.read_bytes()
    report = rebuild_database(source)
    assert report["destination"] is None
    assert source.read_bytes() == before
    assert vault.read_bytes() == before_vault
    # SQLite may create WAL/SHM coordination files even for a read-only connection.
    assert set(source.parent.iterdir()) <= {
        source,
        vault,
        Path(f"{source}-wal"),
        Path(f"{source}-shm"),
    }


@pytest.mark.parametrize("existing", ["database", "vault", "wal", "shm"])
def test_rebuild_never_overwrites_outputs(source: Path, tmp_path: Path, existing: str) -> None:
    destination = tmp_path / "clean.db"
    collision = {
        "database": destination,
        "vault": default_key_vault_path(destination),
        "wal": Path(f"{destination}-wal"),
        "shm": Path(f"{destination}-shm"),
    }[existing]
    collision.write_bytes(b"keep-me")
    with pytest.raises(RebuildError, match="already exists"):
        rebuild_database(source, destination=destination, server_stopped=True)
    assert collision.read_bytes() == b"keep-me"


def test_rebuild_requires_stopped_service_and_distinct_path(source: Path, tmp_path: Path) -> None:
    with pytest.raises(RebuildError, match="Stop the owning service"):
        rebuild_database(source, destination=tmp_path / "clean.db")
    with pytest.raises(RebuildError, match="different paths"):
        rebuild_database(source, destination=source, server_stopped=True)


def test_rebuild_requires_explicit_exclusion_of_invalid_profile(source: Path) -> None:
    with sqlite3.connect(source) as c:
        c.execute(
            "INSERT INTO model_profiles VALUES ('bad','missing','account','bad-key',"
            "'reservation','worker','COMPLETED')"
        )
    with pytest.raises(RebuildError, match="Invalid profile bad"):
        rebuild_database(source)
    report = rebuild_database(source, exclude_profiles={"bad"})
    assert report["excluded_profiles"] == ["bad"]
    assert report["preserved_rows"]["model_profiles"] == 1
    with pytest.raises(RebuildError, match="valid profile"):
        rebuild_database(source, exclude_profiles={"profile", "bad"})
    with pytest.raises(RebuildError, match="not found"):
        rebuild_database(source, exclude_profiles={"bad", "unknown"})


@pytest.mark.parametrize(
    "damage", ["missing_table", "broken_grant", "wrong_vault", "bad_hash", "bad_job_json"]
)
def test_rebuild_aborts_on_unpreservable_core(source: Path, tmp_path: Path, damage: str) -> None:
    with sqlite3.connect(source) as c:
        if damage == "missing_table":
            c.execute("DROP TABLE model_jobs")
        elif damage == "broken_grant":
            c.execute("UPDATE model_grants SET key_id='missing'")
        elif damage == "bad_hash":
            c.execute("UPDATE api_keys SET token_hash='invalid'")
        elif damage == "bad_job_json":
            c.execute("UPDATE model_jobs SET progress_json='invalid'")
    if damage == "wrong_vault":
        default_key_vault_path(source).write_bytes(b"invalid-vault")
    destination = tmp_path / "clean.db"
    with pytest.raises((RebuildError, sqlite3.Error)):
        rebuild_database(source, destination=destination, server_stopped=True)
    assert not destination.exists()
    assert not default_key_vault_path(destination).exists()


def test_rebuild_defers_self_references_in_model_catalog(source: Path) -> None:
    with sqlite3.connect(source) as c:
        parent = c.execute("SELECT id FROM model_catalog").fetchone()[0]
        c.execute("UPDATE model_catalog SET source_model_id=?", (parent,))
    assert rebuild_database(source)["foreign_key_check"] == "ok"
