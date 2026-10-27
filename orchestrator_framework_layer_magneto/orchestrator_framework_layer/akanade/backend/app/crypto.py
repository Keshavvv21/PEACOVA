"""Encryption at rest for OAuth tokens (app/oauth_store.py) — the one
place this codebase stores a secret whose theft grants live write access
to a real person's real external account, a materially higher-value
target than the plaintext LLM API keys admin_providers.py already stores.
Fernet (AES-128-CBC + HMAC, via the `cryptography` package) rather than
anything hand-rolled.
"""
from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from .config import settings

# Fixed, checked-in fallback — same "dev-secret-change-me" shape as
# JWT_SECRET's default (config.py). Only ever used when
# OAUTH_TOKEN_ENCRYPTION_KEY is unset, i.e. local/test runs with no real
# OAuth connections at stake; a real deploy MUST set the env var.
_DEV_FALLBACK_KEY = b"jN3f1ZbYw2C9hT0v6qk8pR4sL7dA1eM5xU9oI2cVfE0="


@lru_cache
def _fernet() -> Fernet:
    key = settings().OAUTH_TOKEN_ENCRYPTION_KEY.encode() or _DEV_FALLBACK_KEY
    return Fernet(key)


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as e:
        raise ValueError("could not decrypt stored value — wrong or rotated "
                         "OAUTH_TOKEN_ENCRYPTION_KEY") from e
