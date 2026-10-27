"""Coverage for auth.py's short-TTL user-validation cache
(_cached_real_app_user) — the fix for every authenticated request
(VS Code, mobile, web) doing a live HTTP round trip to ak-backend's
GET /api/users/me just to validate a bearer token, on every single call.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app import auth


@pytest.fixture(autouse=True)
def clear_user_cache():
    auth._user_cache.clear()
    yield
    auth._user_cache.clear()


def test_repeated_calls_within_ttl_hit_the_cache_not_the_real_app(monkeypatch):
    calls = []

    def fake_real_app_user(token):
        calls.append(token)
        return {"sub": "u1", "name": "Test User", "role": "Member"}

    monkeypatch.setattr(auth, "_real_app_user", fake_real_app_user)
    for _ in range(5):
        user = auth._cached_real_app_user("tok-a")
        assert user["sub"] == "u1"
    assert len(calls) == 1


def test_different_tokens_are_cached_separately(monkeypatch):
    calls = []

    def fake_real_app_user(token):
        calls.append(token)
        return {"sub": token, "name": "x", "role": "Member"}

    monkeypatch.setattr(auth, "_real_app_user", fake_real_app_user)
    a = auth._cached_real_app_user("tok-a")
    b = auth._cached_real_app_user("tok-b")
    assert a["sub"] == "tok-a" and b["sub"] == "tok-b"
    assert len(calls) == 2


def test_expiry_forces_a_fresh_real_app_call(monkeypatch):
    calls = []

    def fake_real_app_user(token):
        calls.append(token)
        return {"sub": "u1", "name": "x", "role": "Member"}

    monkeypatch.setattr(auth, "_real_app_user", fake_real_app_user)

    fake_now = [1000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: fake_now[0])

    auth._cached_real_app_user("tok-a")
    assert len(calls) == 1

    fake_now[0] += auth._USER_CACHE_TTL_SECONDS + 1
    auth._cached_real_app_user("tok-a")
    assert len(calls) == 2


def test_a_rejected_token_is_never_cached(monkeypatch):
    calls = []

    def fake_real_app_user(token):
        calls.append(token)
        raise HTTPException(401, "real app rejected this token")

    monkeypatch.setattr(auth, "_real_app_user", fake_real_app_user)

    for _ in range(3):
        with pytest.raises(HTTPException):
            auth._cached_real_app_user("bad-token")
    # Every call retried the real app — a failure must never be cached,
    # since a token could legitimately succeed moments later (e.g. a
    # race with the real app's own token refresh).
    assert len(calls) == 3
