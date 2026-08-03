from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_rio.api.routes_admin import router as admin_router
from llm_rio.api.routes_inference import router as inference_router
from llm_rio.api.routes_staff import router as staff_router
from llm_rio.config import Settings
from llm_rio.domain import MachineInventory, Role
from llm_rio.errors import RioError
from llm_rio.inventory import discover_inventory
from llm_rio.profiles import ProfileRepository
from llm_rio.recovery import terminate_recorded_workers
from llm_rio.registration import RegistrationManager
from llm_rio.runtime import ResidencyScheduler
from llm_rio.security import issue_api_key
from llm_rio.storage import Database
from llm_rio.validation import ProfileValidator
from llm_rio.workers import WorkerSupervisor

InventoryProvider = Callable[[str, list[str]], MachineInventory]

access_logger = logging.getLogger("uvicorn.error")


async def _ensure_initial_admin(database: Database, default_quota_tokens: int) -> None:
    if await database.key_count() != 0:
        return
    key_id = str(uuid.uuid4())
    account_id = str(uuid.uuid4())
    api_key, prefix = issue_api_key(key_id)
    await database.create_key(
        key_id=key_id,
        nickname="admin",
        role=Role.ADMIN,
        account_id=account_id,
        account_nickname="admin-account",
        prefix=prefix,
        api_key=api_key,
        limit_tokens=default_quota_tokens,
        unlimited=True,
    )
    access_logger.warning("FIRST STARTUP: created initial administrator 'admin'")
    access_logger.warning("INITIAL ADMIN API KEY: %s", api_key)
    access_logger.warning(
        "Create additional admins with: llmctl keys create NAME --role admin"
    )


def create_app(
    settings: Settings | None = None,
    inventory_provider: InventoryProvider = discover_inventory,
) -> FastAPI:
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_directories()
        logging.basicConfig(
            level=getattr(logging, resolved_settings.log_level),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        database = Database(resolved_settings.database_path)
        await database.open()
        worker_client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
        )
        inventory = await asyncio.to_thread(
            inventory_provider,
            resolved_settings.machine_id,
            resolved_settings.managed_gpu_uuids,
        )
        recovery = await terminate_recorded_workers(database)
        for item in recovery:
            await database.record_event("STARTUP_WORKER_RECONCILIATION", payload=item)
        await database.recover_orphaned_state()
        previous_fingerprint = await database.set_machine_fingerprint(inventory.fingerprint)
        if previous_fingerprint and previous_fingerprint != inventory.fingerprint:
            await database.record_event(
                "MACHINE_FINGERPRINT_CHANGED",
                payload={"previous": previous_fingerprint, "current": inventory.fingerprint},
            )
        await _ensure_initial_admin(database, resolved_settings.default_quota_tokens)

        profiles = ProfileRepository(database, inventory.fingerprint)
        supervisor = WorkerSupervisor(resolved_settings, database)
        scheduler = ResidencyScheduler(
            settings=resolved_settings,
            database=database,
            inventory=inventory,
            profiles=profiles,
            supervisor=supervisor,
        )
        validator = ProfileValidator(resolved_settings, inventory, scheduler)
        registration = RegistrationManager(
            settings=resolved_settings,
            database=database,
            inventory=inventory,
            profile_repository=profiles,
            validator=validator,
        )
        app.state.settings = resolved_settings
        app.state.database = database
        app.state.worker_client = worker_client
        app.state.inventory = inventory
        app.state.profiles = profiles
        app.state.supervisor = supervisor
        app.state.scheduler = scheduler
        app.state.registration = registration
        await scheduler.start()
        await registration.resume()
        try:
            yield
        finally:
            await registration.close()
            await scheduler.close()
            await worker_client.aclose()
            await database.close()

    app = FastAPI(
        title="LLM-RIO",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    @app.middleware("http")
    async def forge_style_access_log(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed_seconds = time.perf_counter() - started
            status_text = "OK" if status_code < 400 else "ERROR"
            client_host = request.client.host if request.client else "unknown"
            client_port = request.client.port if request.client else 0
            model = getattr(request.state, "model", "N/A")
            key_nickname = getattr(request.state, "key_nickname", "anonymous")
            request_id = getattr(
                request.state,
                "request_id",
                request.headers.get("X-Request-ID", "N/A"),
            )
            queue_wait_ms = getattr(request.state, "queue_wait_ms", None)
            queue_text = f" - Queue: {queue_wait_ms}ms" if queue_wait_ms is not None else ""
            access_logger.info(
                '%s:%s - "%s %s HTTP/%s" %s %s'
                " - Model: %s - Key: %s - Request: %s%s (%.2fs)",
                client_host,
                client_port,
                request.method,
                request.url.path,
                request.scope.get("http_version", "1.1"),
                status_code,
                status_text,
                model,
                key_nickname,
                request_id,
                queue_text,
                elapsed_seconds,
            )

    @app.exception_handler(RioError)
    async def rio_error_handler(request: Request, exc: RioError) -> JSONResponse:
        headers = {"Retry-After": "30"} if exc.status_code == 503 else {}
        return JSONResponse(exc.openai_body(), status_code=exc.status_code, headers=headers)

    @app.exception_handler(sqlite3.IntegrityError)
    async def integrity_error_handler(
        request: Request, exc: sqlite3.IntegrityError
    ) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": (
                        "The requested nickname, grant, or related record "
                        "conflicts with existing state"
                    ),
                    "type": "state_conflict",
                    "code": "state_conflict",
                }
            },
            status_code=409,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # exc.errors() can embed non-serializable objects (e.g. ValueError in ctx)
        # from pydantic field validators; coerce everything to JSON-safe values.
        details = []
        for error in exc.errors():
            safe: dict[str, Any] = {}
            for key, value in error.items():
                if key == "ctx" and isinstance(value, dict):
                    safe[key] = {
                        str(k): (
                            str(v)
                            if not isinstance(v, str | int | float | bool | type(None))
                            else v
                        )
                        for k, v in value.items()
                    }
                else:
                    safe[key] = value
            details.append(safe)
        return JSONResponse(
            {
                "error": {
                    "message": "Request validation failed",
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                    "details": details,
                }
            },
            status_code=422,
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        mode = await request.app.state.database.service_mode()
        return {
            "status": "ok",
            "mode": mode.value,
            "machine_id": request.app.state.inventory.machine_id,
            "managed_gpus": len(request.app.state.inventory.gpus),
        }

    app.include_router(inference_router)
    app.include_router(staff_router)
    app.include_router(admin_router)
    return app


app = create_app()

