from __future__ import annotations

from llm_rio import admin_client as client_api
from llm_rio.commands.common import (
    _print,
    maintenance_app,
)
from llm_rio.commands.common import app as app


@maintenance_app.command("drain")
def maintenance_drain() -> None:
    _print(client_api.request("POST", "/admin/maintenance", json_body={"mode": "drain"}))


@maintenance_app.command("status")
def maintenance_status() -> None:
    _print(client_api.request("GET", "/admin/maintenance"))


@maintenance_app.command("resume")
def maintenance_resume() -> None:
    _print(client_api.request("POST", "/admin/maintenance", json_body={"mode": "active"}))
