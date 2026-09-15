from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, Checkbox, ContentSwitcher, DataTable, Input, Select
from typer.testing import CliRunner

from llm_rio import cli as cli_api
from llm_rio import tui as tui_module
from llm_rio.config import ServingMode
from llm_rio.tui import RioTui, ServiceLaunch


@pytest.fixture
def management_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {
        "keys": [
            {
                "id": "key-1",
                "nickname": "researcher",
                "api_key": "rio_secret",
                "role": "user",
                "active": True,
                "unlimited": False,
                "balance_tokens": 900,
                "limit_tokens": 1_000,
                "granted_models": ["gemma"],
            }
        ],
        "models": [
            {
                "id": "model-1",
                "nickname": "gemma",
                "state": "AVAILABLE",
                "huggingface_repo": "org/gemma",
                "artifact_path": "/models/gemma",
                "request_defaults": {},
                "request_limits": {"max_context_tokens": 262_144},
                "registration_job": {
                    "id": "job-1",
                    "state": "SUCCEEDED",
                    "stage": "complete",
                },
            }
        ],
    }

    def request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        if (method, path) == ("GET", "/admin/maintenance"):
            return {"mode": "active", "workers": []}
        if (method, path) == ("POST", "/admin/usage/summarize"):
            records.setdefault("summary_requests", []).append({"path": path})
            return {
                "period_start": "2026-08-14T00:00:00+00:00",
                "period_end": "2026-08-21T00:00:00+00:00",
                "summarized_requests": 3,
                "deleted": {
                    "inference_requests": 3,
                    "quota_reservations": 3,
                    "quota_ledger": 6,
                },
                "raw_requests_remaining": 0,
            }
        if (method, path) == ("GET", "/admin/dashboard"):
            window = {
                "period_start": "2026-08-21T00:00:00+00:00",
                "period_end": "2026-08-21T00:01:00+00:00",
                "token_usage": 100,
                "tokens_per_minute": 100.0,
                "average_output_tokens_per_second": 20.0,
                "request_count": 2,
            }
            popularity = [
                {
                    "model_id": "model-1",
                    "model": "gemma",
                    "rank": 1,
                    "token_usage": 100,
                    "share": 1.0,
                }
            ]
            return {
                "generated_at": "2026-08-21T00:01:00+00:00",
                "mode": "ACTIVE",
                "usage": {
                    "generated_at": "2026-08-21T00:01:00+00:00",
                    "current": window,
                    "total": window,
                    "model_popularity": {
                        "current": popularity,
                        "total": popularity,
                    },
                },
                "gpus": [],
                "requests": [
                    {
                        "request_id": "request-1",
                        "state": "QUEUED",
                        "api_key": "researcher",
                        "model": "gemma",
                        "estimated_prompt_tokens": 24,
                        "estimated_tokens": 512,
                        "created_at": "2026-08-21T00:00:00+00:00",
                    }
                ],
            }
        if (method, path) == ("POST", "/admin/keys/key-1/restore"):
            records["keys"][0]["active"] = True
            records.setdefault("restore_requests", []).append(path)
            return {}
        if (method, path) == ("GET", "/admin/models/model-1/profiles"):
            profiles = records.setdefault(
                "profiles",
                [
                    {
                        "id": "profile-1",
                        "active": True,
                        "engine": "vllm",
                        "gpu_count": 1,
                        "tensor_parallel_size": 1,
                        "max_model_len": 4096,
                        "max_num_seqs": 128,
                        "normal_verified": True,
                        "kvcached_verified": False,
                    }
                ],
            )
            return {
                "data": profiles,
                "available_gguf_files": [],
                "kvcached_verification_job": None,
            }
        if (method, path) == (
            "POST",
            "/admin/models/model-1/verify-kvcached",
        ):
            records.setdefault("verification_requests", []).append({"path": path})
            return {
                "id": "verification-job-1",
                "state": "QUEUED",
                "stage": "queued",
            }
        verification_prefix = "/admin/models/model-1/profiles/profile-1/verification/"
        if method == "POST" and path.startswith(verification_prefix):
            backend, action = path.removeprefix(verification_prefix).split("/", 1)
            assert backend in {"native", "kvcached"}
            assert action in {"validate", "invalidate"}
            field = "normal_verified" if backend == "native" else "kvcached_verified"
            records["profiles"][0][field] = action == "validate"
            records.setdefault("verification_override_requests", []).append(
                {"path": path, "backend": backend, "action": action}
            )
            return {"profile_id": "profile-1", field: action == "validate"}
        if method == "PUT" and path.startswith("/staff/keys/") and path.endswith("/model-grants"):
            assert json_body is not None
            key_id = path.split("/")[3]
            key = next(record for record in records["keys"] if record["id"] == key_id)
            model_names = {str(model["id"]): str(model["nickname"]) for model in records["models"]}
            key["granted_models"] = [
                model_names[str(model_id)] for model_id in json_body["model_ids"]
            ]
            records.setdefault("grant_replace_requests", []).append((path, json_body))
            return None

        if (method, path) == ("PATCH", "/admin/models/model-1"):
            assert json_body is not None
            model = records["models"][0]
            defaults = model["request_defaults"]
            for key, value in json_body.items():
                if value is None:
                    defaults.pop(key, None)
                else:
                    defaults[key] = value
            records.setdefault("model_default_requests", []).append(json_body)
            return {"model": model}
        if (method, path) == ("POST", "/admin/models/model-1/clone"):
            assert json_body is not None
            record = {
                "id": "model-2",
                "nickname": json_body["nickname"],
                "state": "AVAILABLE",
                "huggingface_repo": "org/gemma",
                "artifact_path": "/models/gemma",
                "source_model_id": "model-1",
                "request_defaults": {"reasoning_effort": json_body["reasoning_effort"]},
                "request_limits": {"max_context_tokens": json_body["max_model_len"]},
                "registration_job": None,
            }
            records["models"].append(record)
            records.setdefault("clone_requests", []).append(json_body)
            return {"model": record, "profiles": [], "shared_artifact": True}
        if (method, path) == ("POST", "/admin/keys"):
            assert json_body is not None
            record = {
                "id": "key-2",
                "nickname": json_body["nickname"],
                "api_key": "rio_new_secret",
                "role": json_body["role"],
                "active": True,
                "unlimited": json_body["limit_tokens"] is None,
                "balance_tokens": json_body["limit_tokens"] or 0,
                "limit_tokens": json_body["limit_tokens"] or 0,
                "granted_models": json_body["models"],
            }
            records["keys"].append(record)
            return {"nickname": record["nickname"], "api_key": record["api_key"]}
        raise AssertionError((method, path, json_body))

    monkeypatch.setattr(cli_api, "_key_records", lambda: records["keys"])
    monkeypatch.setattr(cli_api, "_model_records", lambda: records["models"])
    monkeypatch.setattr(cli_api, "_request", request)
    monkeypatch.setattr(cli_api, "_base_url", lambda: "http://127.0.0.1:8000")
    monkeypatch.setattr(cli_api, "_api_key", lambda: "rio_admin")
    return records


@pytest.mark.asyncio
async def test_tui_loads_records_and_navigates(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.pause()

        assert app.query_one("#keys-table", DataTable).row_count == 1
        assert app.query_one("#models-table", DataTable).row_count == 1
        queue = app.query_one("#dashboard-requests-table", DataTable)
        assert queue.row_count == 1
        assert queue.get_row("request-1") == [
            "QUEUED",
            "researcher",
            "gemma",
            "24",
            "512",
            "2026-08-21T00:00:00+00:00",
        ]

        await pilot.click("#nav-keys")
        await pilot.pause()
        assert app.query_one("#content", ContentSwitcher).current == "keys"

        await pilot.click("#keys-create")
        await pilot.pause()
        assert len(app.screen_stack) == 2
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_tui_summarizes_usage_after_confirmation(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.click("#nav-maintenance")
        await pilot.pause()
        await pilot.click("#maintenance-summarize")
        await pilot.pause()
        assert len(app.screen_stack) == 2
        await pilot.click("#confirm-submit")
        await asyncio.sleep(0.05)
        await pilot.pause()

    assert management_backend["summary_requests"] == [{"path": "/admin/usage/summarize"}]


@pytest.mark.asyncio
async def test_clone_and_edit_model_inputs_are_focused_and_accept_text(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.click("#nav-models")
        await pilot.pause()
        assert app.query_one("#models-clone", Button).label == "Clone model"

        await pilot.click("#models-clone")
        await pilot.pause()
        clone_name = app.screen.query_one("#field-nickname", Input)
        assert app.screen.focused is clone_name
        await pilot.press("end", "x")
        assert clone_name.value == "gemma-clonex"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.click("#models-edit")
        await pilot.pause()
        temperature = app.screen.query_one("#field-temperature", Input)
        assert app.screen.focused is temperature
        await pilot.press("0")
        assert temperature.value == "0"


@pytest.mark.asyncio
async def test_tui_copies_selected_user_api_key_to_clipboard(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.click("#nav-keys")
        await pilot.pause()
        await pilot.click("#keys-copy")
        await pilot.pause()
        assert app.clipboard == "rio_secret"


@pytest.mark.asyncio
async def test_tui_replaces_access_from_user_and_model_checklists(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    management_backend["models"].append(
        {
            "id": "model-2",
            "nickname": "phi",
            "state": "AVAILABLE",
            "huggingface_repo": "org/phi",
            "artifact_path": "/models/phi",
            "request_defaults": {},
            "request_limits": {"max_context_tokens": 32_768},
            "registration_job": None,
        }
    )
    management_backend["keys"].append(
        {
            "id": "key-2",
            "nickname": "student",
            "api_key": "rio_student",
            "role": "user",
            "active": True,
            "unlimited": False,
            "balance_tokens": 900,
            "limit_tokens": 1_000,
            "granted_models": [],
        }
    )
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.click("#nav-keys")
        await pilot.pause()
        await pilot.click("#keys-access-update")
        await pilot.pause()
        granted = app.screen.query_one("#field-model-model-1", Checkbox)
        available = app.screen.query_one("#field-model-model-2", Checkbox)
        assert granted.value is True
        assert available.value is False
        granted.value = False
        available.value = True
        await pilot.click("#form-submit")
        await asyncio.sleep(0.05)
        await pilot.pause()
        assert management_backend["keys"][0]["granted_models"] == ["phi"]

        await pilot.click("#nav-models")
        await pilot.pause()
        await pilot.click("#models-user-access")
        await pilot.pause()
        researcher = app.screen.query_one("#field-user-key-1", Checkbox)
        student = app.screen.query_one("#field-user-key-2", Checkbox)
        assert researcher.value is False
        assert student.value is False
        student.value = True
        await pilot.click("#form-submit")
        await asyncio.sleep(0.05)
        await pilot.pause()
        assert management_backend["keys"][0]["granted_models"] == ["phi"]
        assert management_backend["keys"][1]["granted_models"] == ["gemma"]


@pytest.mark.asyncio
async def test_all_button_labels_fit_at_standard_terminal_size(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.pause()

        navigation = list(app.query("#sidebar Button").results(Button))
        assert all(button.region.bottom <= 23 for button in navigation)

        for page in (
            "dashboard",
            "keys",
            "models",
            "profiles",
            "maintenance",
            "system",
        ):
            app._navigate(page)
            await pilot.pause()
            buttons = app.query(f"#{page} .toolbar Button").results(Button)
            assert all(button.region.width >= len(str(button.label)) + 4 for button in buttons)


@pytest.mark.asyncio
async def test_tui_create_key_uses_same_management_route(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await app._create_key(
            {
                "nickname": "student",
                "role": "ta",
                "unlimited": False,
                "limit": "5000",
                "account_id": "",
                "grants": "gemma",
                "api_key": "",
            }
        )

        assert management_backend["keys"][-1]["nickname"] == "student"
        assert management_backend["keys"][-1]["role"] == "ta"
        assert management_backend["keys"][-1]["limit_tokens"] == 5_000
        assert app.query_one("#keys-table", DataTable).row_count == 2


@pytest.mark.asyncio
async def test_tui_restores_revoked_key(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    management_backend["keys"][0]["active"] = False
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await app._key_action(management_backend["keys"][0], "restore")

        assert management_backend["keys"][0]["active"] is True
        assert management_backend["restore_requests"] == ["/admin/keys/key-1/restore"]


@pytest.mark.asyncio
async def test_tui_clone_profile_uses_shared_model_route(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await app._clone_model_profile(
            management_backend["models"][0],
            {
                "nickname": "gemma-ext",
                "temperature": "",
                "top_p": "",
                "top_k": "",
                "min_p": "0.12",
                "presence_penalty": "-0.3",
                "repetition_penalty": "1.1",
                "reasoning_effort": "medium",
                "max_model_len": "1048576",
                "yarn_factor": "4",
                "yarn_original_max_model_len": "262144",
                "inherit_grants": True,
            },
        )

        assert management_backend["clone_requests"] == [
            {
                "nickname": "gemma-ext",
                "inherit_grants": True,
                "min_p": 0.12,
                "presence_penalty": -0.3,
                "repetition_penalty": 1.1,
                "yarn_factor": 4.0,
                "max_model_len": 1_048_576,
                "yarn_original_max_model_len": 262_144,
                "reasoning_effort": "medium",
            }
        ]
        assert app.query_one("#models-table", DataTable).row_count == 2


@pytest.mark.asyncio
async def test_tui_edits_model_generation_defaults(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await app._edit_model(
            management_backend["models"][0],
            {
                "temperature": "0.3",
                "top_p": "0.9",
                "top_k": "40",
                "min_p": "0.1",
                "presence_penalty": "-0.5",
                "repetition_penalty": "1.05",
                "reasoning_effort": "high",
            },
        )

        assert management_backend["model_default_requests"] == [
            {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.1,
                "presence_penalty": -0.5,
                "repetition_penalty": 1.05,
                "reasoning_effort": "high",
            }
        ]
        assert management_backend["models"][0]["request_defaults"]["min_p"] == 0.1


@pytest.mark.asyncio
async def test_clone_model_form_keeps_actions_and_input_text_visible(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.pause()

        app._open_clone_model_profile(management_backend["models"][0])
        await pilot.pause()
        nickname = app.screen.query_one("#field-nickname", Input)
        assert nickname.content_region.height >= 1

        for button_id in ("form-cancel", "form-submit"):
            button = app.screen.query_one(f"#{button_id}", Button)
            assert button.region.height > 0
            assert button.region.bottom <= app.size.height


def test_cli_clone_model_profile_posts_configured_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []
    monkeypatch.setattr(
        cli_api,
        "_model_record",
        lambda nickname: {"id": "source-model", "nickname": nickname},
    )

    def request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        calls.append((method, path, json_body))
        return {
            "model": {
                "nickname": "qwen-ext",
                "artifact_path": "/models/qwen",
                "request_defaults": {"reasoning_effort": "medium"},
            },
            "profiles": [{"id": "profile-2"}],
            "shared_artifact": True,
        }

    monkeypatch.setattr(cli_api, "_request", request)

    result = CliRunner().invoke(
        cli_api.app,
        [
            "models",
            "profile-clone",
            "qwen",
            "qwen-ext",
            "--reasoning-effort",
            "medium",
            "--max-model-len",
            "1048576",
            "--yarn-factor",
            "4",
            "--min-p",
            "0.1",
            "--presence-penalty",
            "-0.5",
            "--repetition-penalty",
            "1.05",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "POST",
            "/admin/models/source-model/clone",
            {
                "nickname": "qwen-ext",
                "reasoning_effort": "medium",
                "min_p": 0.1,
                "presence_penalty": -0.5,
                "repetition_penalty": 1.05,
                "max_model_len": 1_048_576,
                "yarn_factor": 4.0,
                "inherit_grants": True,
            },
        )
    ]
    assert "1 cloned placement profile" in result.stdout
    assert "Shared artifact: /models/qwen" in result.stdout


def test_no_argument_cli_launches_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def run_tui() -> None:
        calls.append(True)

    monkeypatch.setattr(tui_module, "run_tui", run_tui)

    result = CliRunner().invoke(cli_api.app, [])

    assert result.exit_code == 0
    assert calls == [True]


@pytest.mark.parametrize("mode", [None, *ServingMode])
def test_tui_can_handoff_to_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: ServingMode | None
) -> None:
    config = tmp_path / "alternate.toml"
    config.write_text('serving_mode = "queue"\n')
    served: list[Any] = []
    monkeypatch.setattr(tui_module, "run_tui", lambda: ServiceLaunch(config, mode))
    monkeypatch.setattr(cli_api, "create_app", lambda settings: settings)
    monkeypatch.setattr(cli_api.uvicorn, "run", lambda app, **kwargs: served.append(app))

    result = CliRunner().invoke(cli_api.app, ["interactive"])

    assert result.exit_code == 0, result.output
    assert len(served) == 1
    assert served[0].serving_mode == (mode or ServingMode.QUEUE)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, *ServingMode])
async def test_start_service_form_returns_selected_mode(
    management_backend: dict[str, list[dict[str, Any]]], mode: ServingMode | None
) -> None:
    app = RioTui()
    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await pilot.click("#dashboard-start-service")
        await pilot.pause()
        app.screen.query_one("#field-config", Input).value = "config.barra.toml"
        selector = app.screen.query_one("#field-mode", Select)
        assert selector.value == "configured"
        selector.value = mode.value if mode else "configured"
        await pilot.click("#form-submit")
        await pilot.pause()
    assert app.return_value == ServiceLaunch(Path("config.barra.toml"), mode)


@pytest.mark.asyncio
async def test_dashboard_refresh_preserves_table_scroll(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(80, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.pause()

        queue = app.query_one("#dashboard-requests-table", DataTable)
        assert queue.max_scroll_x > 0
        queue.scroll_to(x=queue.max_scroll_x, animate=False)
        await pilot.pause()
        scroll_x = queue.scroll_x

        app._render_dashboard(app.dashboard_payload)
        await pilot.pause()
        assert queue.scroll_x == scroll_x


@pytest.mark.asyncio
async def test_tui_exposes_manual_backend_verification_actions(
    management_backend: dict[str, list[dict[str, Any]]],
) -> None:
    app = RioTui()

    async with app.run_test(size=(130, 45)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.click("#nav-models")
        await pilot.pause()
        await pilot.click("#models-profiles")
        await pilot.pause()

        assert app.query_one("#profiles-verify-kvcached", Button).label == "Verify kvcached"
        assert (
            app.query_one("#profiles-toggle-normal-verification", Button).label
            == "Invalidate model"
        )
        assert (
            app.query_one("#profiles-toggle-kvcached-verification", Button).label
            == "Validate kvcached"
        )

        await pilot.click("#profiles-verify-kvcached")
        await pilot.pause()
        await pilot.click("#confirm-submit")
        await asyncio.sleep(0.05)
        await pilot.pause()

        await pilot.click("#profiles-toggle-normal-verification")
        await pilot.pause()
        await pilot.click("#confirm-submit")
        await asyncio.sleep(0.05)
        await pilot.pause()

        await pilot.click("#profiles-toggle-kvcached-verification")
        await pilot.pause()
        await pilot.click("#confirm-submit")
        await asyncio.sleep(0.05)
        await pilot.pause()

    assert management_backend["verification_requests"] == [
        {"path": "/admin/models/model-1/verify-kvcached"}
    ]
    assert management_backend["verification_override_requests"] == [
        {
            "path": "/admin/models/model-1/profiles/profile-1/verification/native/invalidate",
            "backend": "native",
            "action": "invalidate",
        },
        {
            "path": "/admin/models/model-1/profiles/profile-1/verification/kvcached/validate",
            "backend": "kvcached",
            "action": "validate",
        },
    ]
    assert management_backend["profiles"][0]["normal_verified"] is False
    assert management_backend["profiles"][0]["kvcached_verified"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("bulk", [False, True])
async def test_tui_trust_recovers_without_selecting_a_profile(
    management_backend: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    bulk: bool,
) -> None:
    original_request = cli_api._request
    requests = []

    def request(method, path, *, json_body=None):
        if method == "POST" and path.endswith(("/trust-available", "/trust-verification")):
            requests.append((path, json_body))
            return {"profiles_updated": 1, "data": [{"model_id": "model-1"}], "skipped": []}
        return original_request(method, path, json_body=json_body)

    monkeypatch.setattr(cli_api, "_request", request)
    app = RioTui()
    async with app.run_test(size=(130, 60)) as pilot:
        await pilot.pause()
        await pilot.click("#nav-models")
        await pilot.pause()
        button = "#models-trust-available" if bulk else "#models-trust"
        await pilot.click(button)
        await pilot.pause()
        assert requests == []
        await pilot.click("#confirm-cancel")
        await pilot.pause()
        assert requests == []
        await pilot.click(button)
        await pilot.pause()
        await pilot.click("#confirm-submit")
        await pilot.pause()
        await app.workers.wait_for_complete()
    path = "/admin/models/trust-available" if bulk else "/admin/models/model-1/trust-verification"
    assert requests == [(path, {})]
