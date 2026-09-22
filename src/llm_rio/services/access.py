from __future__ import annotations

import uuid

from llm_rio.api.schemas import CreateKeyRequest, KeySecretResponse
from llm_rio.errors import RioError
from llm_rio.security import issue_api_key, token_prefix
from llm_rio.storage import Database


async def create_key(database: Database, body: CreateKeyRequest) -> KeySecretResponse:
    if body.models:
        available_models = {str(model["nickname"]) for model in await database.list_models()}
        missing_models = sorted(set(body.models) - available_models)
        if missing_models:
            raise RioError(
                "model_not_found",
                "One or more model nicknames do not exist",
                status_code=404,
                details={"missing_models": missing_models},
            )
    key_id = str(uuid.uuid4())
    if body.quota_account_id:
        account = await database.fetchone(
            "SELECT id, nickname FROM quota_accounts WHERE id = ?", (body.quota_account_id,)
        )
        if account is None:
            raise RioError("account_not_found", "Quota account was not found", status_code=404)
        account_id = account["id"]
        account_nickname = account["nickname"]
    else:
        account_id = str(uuid.uuid4())
        account_nickname = body.quota_account_nickname or body.nickname
    if body.api_key is None:
        token, prefix = issue_api_key(key_id)
    else:
        token = body.api_key
        prefix = token_prefix(token)
    limit_tokens = body.limit_tokens if body.limit_tokens is not None else 0
    unlimited = body.limit_tokens is None
    await database.create_key(
        key_id=key_id,
        nickname=body.nickname,
        role=body.role,
        account_id=account_id,
        account_nickname=account_nickname,
        prefix=prefix,
        api_key=token,
        limit_tokens=limit_tokens,
        unlimited=unlimited,
    )
    if body.models:
        await database.update_model_access(
            key_id=key_id, model_nicknames=body.models, mode="replace"
        )
    return KeySecretResponse(id=key_id, nickname=body.nickname, api_key=token)
