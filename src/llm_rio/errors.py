from __future__ import annotations

from typing import Any


class RioError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def openai_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.code,
                "code": self.code,
                **self.details,
            }
        }


class AuthenticationError(RioError):
    def __init__(self, message: str = "Invalid or inactive API key") -> None:
        super().__init__("invalid_api_key", message, status_code=401)


class AuthorizationError(RioError):
    def __init__(self, message: str = "This operation is not permitted") -> None:
        super().__init__("permission_denied", message, status_code=403)


class QuotaExceededError(RioError):
    def __init__(self, available: int, requested: int) -> None:
        super().__init__(
            "quota_exceeded",
            "The quota account has insufficient token credit",
            status_code=429,
            details={"available_tokens": available, "requested_tokens": requested},
        )


class QueueFullError(RioError):
    def __init__(self) -> None:
        super().__init__("queue_full", "The model queue is at capacity", status_code=429)


class MaintenanceError(RioError):
    def __init__(self) -> None:
        super().__init__(
            "service_maintenance",
            "The service is draining or in maintenance",
            status_code=503,
        )

