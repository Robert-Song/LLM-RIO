"""Tests for API-key issuance, hashing, verification, and the encrypted vault."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rio.domain import Role
from llm_rio.errors import AuthenticationError
from llm_rio.security import (
    ApiKeyVault,
    default_key_vault_path,
    hash_api_key,
    hash_idempotency_key,
    issue_api_key,
    token_prefix,
    verify_api_key,
)


class TestIssueApiKey:
    def test_token_format(self) -> None:
        token, prefix = issue_api_key("key-1234567890ab")
        assert token.startswith("rio_")
        assert len(prefix) == 24
        assert token.startswith(prefix)
        assert len(token) > 24

    def test_unique_tokens(self) -> None:
        first, _ = issue_api_key("key-1234567890ab")
        second, _ = issue_api_key("key-1234567890ab")
        assert first != second

    def test_public_id_is_derived_from_key_id(self) -> None:
        token, _ = issue_api_key("key-1234567890ab")
        # key_id with dashes stripped, first 12 chars: "key1234567890ab"[:12] == "key123456789"
        assert token.startswith("rio_key123456789_")


class TestHashVerify:
    def test_round_trip(self) -> None:
        token, _ = issue_api_key("key-1234567890ab")
        hashed = hash_api_key(token)
        assert verify_api_key(hashed, token) is True

    def test_wrong_token_rejected(self) -> None:
        token, _ = issue_api_key("key-1234567890ab")
        hashed = hash_api_key(token)
        assert verify_api_key(hashed, "rio_wrong_token_value") is False

    def test_malformed_hash_rejected(self) -> None:
        assert verify_api_key("not-an-argon2-hash", "rio_anything") is False

    def test_hashes_are_salted(self) -> None:
        token, _ = issue_api_key("key-1234567890ab")
        assert hash_api_key(token) != hash_api_key(token)


class TestTokenPrefix:
    def test_valid_prefix(self) -> None:
        token, _ = issue_api_key("key-1234567890ab")
        assert token_prefix(token) == token[:24]

    def test_non_rio_token_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            token_prefix("sk-abcdefghijklmnopqrstuvwx")

    def test_short_token_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            token_prefix("rio_short")


class TestIdempotencyHash:
    def test_deterministic(self) -> None:
        assert hash_idempotency_key("abc") == hash_idempotency_key("abc")

    def test_different_inputs_differ(self) -> None:
        assert hash_idempotency_key("abc") != hash_idempotency_key("abd")

    def test_sha256_hex(self) -> None:
        assert len(hash_idempotency_key("abc")) == 64


class TestApiKeyVault:
    def test_vault_round_trip(self, tmp_path: Path) -> None:
        vault = ApiKeyVault(tmp_path / "vault")
        secret = "rio_abcdefghijkl_very-secret-value"
        encrypted = vault.encrypt(secret)
        assert encrypted != secret
        assert vault.decrypt(encrypted) == secret

    def test_vault_key_is_persistent(self, tmp_path: Path) -> None:
        path = tmp_path / "vault"
        first = ApiKeyVault(path)
        secret = "rio_persistent_secret_value_123"
        encrypted = first.encrypt(secret)
        second = ApiKeyVault(path)
        assert second.decrypt(encrypted) == secret

    def test_vault_created_with_restrictive_permissions(self, tmp_path: Path) -> None:
        path = tmp_path / "vault"
        ApiKeyVault(path)
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600

    def test_wrong_vault_cannot_decrypt(self, tmp_path: Path) -> None:
        first = ApiKeyVault(tmp_path / "vault-one")
        encrypted = first.encrypt("rio_secret_value_123456")
        second = ApiKeyVault(tmp_path / "vault-two")
        with pytest.raises(RuntimeError):
            second.decrypt(encrypted)


class TestDefaultKeyVaultPath:
    def test_path_derived_from_database(self) -> None:
        path = default_key_vault_path(Path("/data/state/llm-rio.db"))
        assert path == Path("/data/state/.llm-rio-api-key-vault")

    def test_role_values(self) -> None:
        assert Role.ADMIN.value == "admin"
        assert Role.TA.value == "ta"
        assert Role.USER.value == "user"
