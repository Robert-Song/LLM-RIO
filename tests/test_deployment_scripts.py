from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import final_deployment_test as deploy
import fire_all_native_revalidations as revalidate


def model(name="model", state="AVAILABLE", job="COMPLETED"):
    return {"nickname": name, "state": state, "registration_job": {"id": name, "state": job}}


@pytest.mark.parametrize("dry_run", [True, False])
def test_revalidation_includes_failed_and_previously_verified_models(monkeypatch, dry_run):
    monkeypatch.setattr(
        revalidate,
        "parse_args",
        lambda: SimpleNamespace(dry_run=dry_run, only_invalid=False, report=None),
    )
    calls = []

    def ctl(*args):
        calls.append(args)
        if args == ("models", "list", "--json"):
            return {
                "data": [
                    model(),
                    model("failed", "NEEDS_ADMIN_REVIEW", "FAILED"),
                    model("disabled", "DISABLED"),
                ]
            }
        if args == ("maintenance", "status"):
            return {"validation": {"requires_maintenance": True}}
        return {"data": []}

    monkeypatch.setattr(revalidate, "llmctl_json", ctl)
    submit = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(revalidate.subprocess, "run", submit)
    assert revalidate.main() == 0
    assert submit.call_count == (0 if dry_run else 2)
    assert (("maintenance", "drain") in calls) == (not dry_run)


@pytest.mark.parametrize("serving_mode", ["vllm-sleep", "queue"])
@pytest.mark.parametrize("job", ["RUNNING", "FAILED", "COMPLETED"])
def test_deployment_continue_requires_successful_jobs(monkeypatch, job, serving_mode):
    monkeypatch.setattr(
        deploy,
        "parse_args",
        lambda: SimpleNamespace(
            api_key="test",
            max_tokens=512,
            base_url="http://localhost:8003/v1",
            timeout=1,
            report=None,
            model=[],
            all_available=False,
            resume=True,
        ),
    )
    calls = []

    def ctl(*args):
        calls.append(args)
        if args == ("maintenance", "status"):
            return {
                "mode": "MAINTENANCE_READY",
                "serving_mode": serving_mode,
                "validation": {"requires_maintenance": True, "gpu_uuids": []},
            }
        return {"data": [model(job=job)]}

    monkeypatch.setattr(deploy, "llmctl_json", ctl)
    response = SimpleNamespace(
        model="model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="LLM-RIO deployment test passed."),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(completion_tokens=8),
    )
    create = Mock(return_value=response)
    monkeypatch.setattr(
        deploy,
        "OpenAI",
        lambda **kw: SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(data=[SimpleNamespace(id="model")])
            ),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
    )
    assert deploy.main() == (0 if job == "COMPLETED" else 2)
    assert (("maintenance", "resume") in calls) == (job == "COMPLETED")
    assert create.call_count == (1 if job == "COMPLETED" else 0)


@pytest.mark.parametrize(
    "text,finish,tokens",
    [("", "stop", 1), ("wrong", "stop", 1), ("answer", "length", 1), ("answer", "stop", 0)],
)
def test_deployment_does_not_pass_http_success_with_invalid_completion(
    monkeypatch, text, finish, tokens
):
    monkeypatch.setattr(
        deploy,
        "parse_args",
        lambda: SimpleNamespace(
            api_key="test",
            max_tokens=512,
            base_url="http://localhost:8003/v1",
            timeout=1,
            report=None,
            model=[],
            all_available=False,
            resume=False,
        ),
    )
    monkeypatch.setattr(
        deploy,
        "llmctl_json",
        lambda *args: (
            {"mode": "ACTIVE", "validation": {"requires_maintenance": True, "gpu_uuids": []}}
            if args[0] == "maintenance"
            else {"data": [model()]}
        ),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(completion_tokens=tokens),
    )
    monkeypatch.setattr(
        deploy,
        "OpenAI",
        lambda **kw: SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(data=[SimpleNamespace(id="model")])
            ),
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)),
        ),
    )
    assert deploy.main() == 1
