"""app/oauth/state.py, app/crypto.py, app/oauth_store.py, and
app/routers/oauth.py — the per-user 'Connect your account' OAuth flow
for GitHub/Linear/Google. See test_user_mcp_client.py for the per-user
MCP tool-discovery/dispatch side this feature enables once connected.
"""
from __future__ import annotations

import datetime as dt

import httpx
import jwt as pyjwt
import pytest
from fastapi import HTTPException

from app import crypto, oauth_store
from app.config import settings
from app.oauth.state import sign_state, verify_state
from tests.conftest import auth


def _user_id_for(token: str) -> str:
    return pyjwt.decode(token, settings().JWT_SECRET, algorithms=[settings().JWT_ALG])["sub"]


# ── state (CSRF) ──────────────────────────────────────────────────────

def test_state_round_trips_for_the_provider_it_was_signed_for():
    token = sign_state("user-1", "github")
    assert verify_state(token, "github") == "user-1"


def test_state_rejects_a_different_provider_than_it_was_signed_for():
    token = sign_state("user-1", "github")
    with pytest.raises(HTTPException):
        verify_state(token, "linear")


def test_state_rejects_a_tampered_token():
    token = sign_state("user-1", "github")
    with pytest.raises(HTTPException):
        verify_state(token + "x", "github")


def test_state_rejects_an_expired_token():
    # Minted already-expired by hand, rather than sleeping 10 real minutes.
    now = dt.datetime.now(dt.timezone.utc)
    token = pyjwt.encode(
        {"sub": "user-1", "provider": "github", "typ": "oauth_state",
         "iat": now - dt.timedelta(minutes=20), "exp": now - dt.timedelta(minutes=10)},
        settings().JWT_SECRET, algorithm=settings().JWT_ALG)
    with pytest.raises(HTTPException):
        verify_state(token, "github")


# ── crypto ────────────────────────────────────────────────────────────

def test_encrypt_decrypt_round_trips():
    assert crypto.decrypt(crypto.encrypt("a-real-token")) == "a-real-token"


def test_encrypted_value_does_not_contain_the_plaintext():
    ciphertext = crypto.encrypt("super-secret-token")
    assert "super-secret-token" not in ciphertext


# ── oauth_store ───────────────────────────────────────────────────────

def test_save_get_list_delete_connection():
    oauth_store.save_connection("u1", "github", access_token="tok-1", refresh_token="ref-1",
                                expires_in=3600, scopes=["repo"])
    row = oauth_store.get_connection("u1", "github")
    assert row is not None
    assert crypto.decrypt(row["access_token"]) == "tok-1"
    assert crypto.decrypt(row["refresh_token"]) == "ref-1"

    conns = oauth_store.list_connections("u1")
    assert any(c["SK"] == "PROVIDER#github" for c in conns)

    oauth_store.delete_connection("u1", "github")
    assert oauth_store.get_connection("u1", "github") is None


def test_get_valid_access_token_returns_none_with_no_connection():
    assert oauth_store.get_valid_access_token("nobody", "github") is None


def test_get_valid_access_token_returns_the_stored_token_when_not_expired():
    oauth_store.save_connection("u2", "linear", access_token="tok-2", refresh_token=None,
                                expires_in=None, scopes=["read", "write"])
    assert oauth_store.get_valid_access_token("u2", "linear") == "tok-2"
    oauth_store.delete_connection("u2", "linear")


def test_get_valid_access_token_refreshes_an_expired_token(monkeypatch):
    oauth_store.save_connection("u3", "google", access_token="old-tok", refresh_token="ref-3",
                                expires_in=-10, scopes=["gmail"])  # already expired

    def fake_post(url, data=None, headers=None, timeout=None):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "ref-3"

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"access_token": "new-tok", "expires_in": 3600}
        return R()

    monkeypatch.setattr(oauth_store.httpx, "post", fake_post)
    token = oauth_store.get_valid_access_token("u3", "google")
    assert token == "new-tok"
    oauth_store.delete_connection("u3", "google")


def test_get_valid_access_token_returns_none_when_refresh_fails(monkeypatch):
    oauth_store.save_connection("u4", "google", access_token="old-tok", refresh_token="ref-4",
                                expires_in=-10, scopes=["gmail"])

    def fake_post(url, data=None, headers=None, timeout=None):
        raise httpx.ConnectError("boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(oauth_store.httpx, "post", fake_post)
    assert oauth_store.get_valid_access_token("u4", "google") is None
    oauth_store.delete_connection("u4", "google")


def test_get_valid_access_token_returns_none_when_expired_with_no_refresh_token():
    oauth_store.save_connection("u4b", "github", access_token="old-tok", refresh_token=None,
                                expires_in=-10, scopes=["repo"])
    assert oauth_store.get_valid_access_token("u4b", "github") is None
    oauth_store.delete_connection("u4b", "github")


def test_exchange_code_success(monkeypatch):
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"access_token": "tok", "expires_in": 3600, "scope": "repo"}
        return R()

    monkeypatch.setattr(oauth_store.httpx, "post", fake_post)
    body = oauth_store.exchange_code("github", "a-code", "http://localhost:8001/api/oauth/github/callback")
    assert body["access_token"] == "tok"
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["data"]["code"] == "a-code"


def test_exchange_code_raises_when_provider_returns_an_error(monkeypatch):
    def fake_post(url, data=None, headers=None, timeout=None):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"error": "bad_verification_code"}
        return R()

    monkeypatch.setattr(oauth_store.httpx, "post", fake_post)
    with pytest.raises(ValueError):
        oauth_store.exchange_code("github", "bad-code", "http://x/callback")


def test_revoke_connection_deletes_locally_even_if_the_remote_call_fails(monkeypatch):
    oauth_store.save_connection("u5", "github", access_token="tok-5", refresh_token=None,
                                expires_in=None, scopes=["repo"])

    def fake_request(method, url, **kw):
        raise httpx.ConnectError("boom", request=httpx.Request(method, url))

    monkeypatch.setattr(oauth_store.httpx, "request", fake_request)
    oauth_store.revoke_connection("u5", "github")
    assert oauth_store.get_connection("u5", "github") is None


# ── router ────────────────────────────────────────────────────────────

def test_authorize_requires_login(client):
    r = client.get("/api/oauth/github/authorize")
    assert r.status_code == 401


def test_authorize_404s_for_an_unknown_provider(client, member_token):
    r = client.get("/api/oauth/bitbucket/authorize", headers=auth(member_token))
    assert r.status_code == 404


def test_authorize_409s_when_the_provider_has_no_client_id_configured(client, member_token):
    # Tests run with every OAuth provider's client id/secret unset (see
    # conftest's blanket "every rung unconfigured" convention) — this IS
    # the unconfigured case, not special setup.
    r = client.get("/api/oauth/github/authorize", headers=auth(member_token))
    assert r.status_code == 409


def test_authorize_returns_a_real_looking_url_once_configured(client, member_token, monkeypatch):
    monkeypatch.setattr(settings(), "GITHUB_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(settings(), "GITHUB_OAUTH_CLIENT_SECRET", "test-secret")
    r = client.get("/api/oauth/github/authorize", headers=auth(member_token))
    assert r.status_code == 200
    url = r.json()["url"]
    assert url.startswith("https://github.com/login/oauth/authorize?")
    assert "client_id=test-client-id" in url
    assert "state=" in url


def test_callback_rejects_a_tampered_state(client):
    r = client.get("/api/oauth/github/callback?code=abc&state=not-a-real-token",
                   follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "error=invalid_state" in r.headers["location"]


def test_callback_redirects_with_error_when_the_provider_denies_consent(client):
    r = client.get("/api/oauth/github/callback?error=access_denied", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "error=access_denied" in r.headers["location"]


def test_callback_saves_a_connection_on_success(client, member_token, monkeypatch):
    from app.routers import oauth as oauth_router
    user_id = _user_id_for(member_token)
    state = sign_state(user_id, "github")
    monkeypatch.setattr(
        oauth_router.oauth_store, "exchange_code",
        lambda provider, code, redirect_uri: {"access_token": "tok-x", "refresh_token": None,
                                              "expires_in": None, "scope": "repo"})
    r = client.get(f"/api/oauth/github/callback?code=abc&state={state}", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "connected=github" in r.headers["location"]

    row = oauth_store.get_connection(user_id, "github")
    assert row is not None
    oauth_store.delete_connection(user_id, "github")


def test_callback_redirects_with_error_when_token_exchange_fails(client, member_token, monkeypatch):
    from app.routers import oauth as oauth_router
    state = sign_state(_user_id_for(member_token), "github")

    def boom(provider, code, redirect_uri):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(oauth_router.oauth_store, "exchange_code", boom)
    r = client.get(f"/api/oauth/github/callback?code=abc&state={state}", follow_redirects=False)
    assert "error=token_exchange_failed" in r.headers["location"]


def test_list_and_disconnect_connections(client, member_token):
    user_id = _user_id_for(member_token)
    oauth_store.save_connection(user_id, "linear", access_token="tok", refresh_token=None,
                                expires_in=None, scopes=["read", "write"])

    r = client.get("/api/oauth/connections", headers=auth(member_token))
    assert r.status_code == 200
    body = r.json()
    assert any(c["provider"] == "linear" for c in body)
    assert not any("access_token" in c for c in body)  # never leaks the token

    r = client.delete("/api/oauth/connections/linear", headers=auth(member_token))
    assert r.status_code == 204
    assert oauth_store.get_connection(user_id, "linear") is None
