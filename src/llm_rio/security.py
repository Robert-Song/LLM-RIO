from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken

from llm_rio.domain import Role
from llm_rio.errors import AuthenticationError

_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)


@dataclass(frozen=True, slots=True)
class Principal:
    key_id: str
    nickname: str
    role: Role
    quota_account_id: str
    unlimited: bool


def issue_api_key(key_id: str) -> tuple[str, str]:
    """Return a lab API key and its searchable prefix."""
    public_id = key_id.replace("-", "")[:12]
    secret = secrets.token_urlsafe(32)
    token = f"rio_{public_id}_{secret}"
    prefix = token[:24]
    return token, prefix


def hash_api_key(token: str) -> str:
    return _hasher.hash(token)


def verify_api_key(stored_hash: str, token: str) -> bool:
    try:
        return _hasher.verify(stored_hash, token)
    except (InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def token_prefix(token: str) -> str:
    if not token.startswith("rio_") or len(token) < 24:
        raise AuthenticationError()
    return token[:24]


def hash_idempotency_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def default_key_vault_path(database_path: Path) -> Path:
    return database_path.with_name(f".{database_path.stem}-api-key-vault")


class ApiKeyVault:
    """Encrypt recoverable API keys with an automatically created host-local key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = self.path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(key + b"\n")
        os.chmod(self.path, 0o600)
        return key

    def encrypt(self, api_key: str) -> str:
        return self._fernet.encrypt(api_key.encode()).decode()

    def decrypt(self, encrypted_api_key: str) -> str:
        try:
            return self._fernet.decrypt(encrypted_api_key.encode()).decode()
        except InvalidToken as exc:
            raise RuntimeError(
                f"Cannot decrypt an API key with the vault at {self.path}"
            ) from exc

