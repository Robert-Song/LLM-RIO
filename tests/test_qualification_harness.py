"""The hardware gate must reject mismatched or unsettled accounting evidence."""

from scripts.qualify import accounting_matches


def test_qualification_accounting_requires_exact_settled_usage() -> None:
    requests = [
        {"model": "first", "completion_tokens": 0},
        {"model": "second", "completion_tokens": 12},
    ]
    rows = [
        {
            "model": item["model"],
            "token_usage": {"prompt_tokens": 3, "completion_tokens": item["completion_tokens"]},
            "completion_status": "COMPLETED",
            "error_code": None,
            "accepted_count": 1,
            "completion_count": 1,
        }
        for item in requests
    ]
    assert accounting_matches(requests, list(reversed(rows)))
    assert not accounting_matches(requests, rows[:1])
    assert not accounting_matches(requests, [{**rows[0], "completion_status": "FAILED"}, rows[1]])
    assert not accounting_matches(requests, [{**rows[0], "completion_count": 2}, rows[1]])
    assert not accounting_matches(
        requests, [{**rows[0], "token_usage": {"completion_tokens": 1}}, rows[1]]
    )
