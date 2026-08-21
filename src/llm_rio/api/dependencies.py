from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_rio.domain import Role
from llm_rio.errors import AuthenticationError, AuthorizationError
from llm_rio.security import Principal, token_prefix

bearer = HTTPBearer(auto_error=False)


async def current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("A Bearer API key is required")
    token = credentials.credentials
    principal = await request.app.state.database.authenticate(token_prefix(token), token)
    if principal is None:
        raise AuthenticationError()
    request.state.key_nickname = principal.nickname
    request.state.key_role = principal.role.value
    return principal


def require_roles(*roles: Role) -> Callable[..., Principal]:
    async def dependency(
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> Principal:
        if principal.role not in roles:
            raise AuthorizationError()
        return principal

    return dependency


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]
StaffPrincipal = Annotated[
    Principal, Depends(require_roles(Role.TA, Role.ADMIN))
]
AdminPrincipal = Annotated[Principal, Depends(require_roles(Role.ADMIN))]

