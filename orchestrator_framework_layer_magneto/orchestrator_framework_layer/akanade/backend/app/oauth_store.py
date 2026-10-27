"""Per-user OAuth connection storage + token lifecycle (exchange, refresh,
revoke). Reuses app/db.py's existing store() factory (PK/SK document
store, no new persistence layer) and app/crypto.py for encryption at
rest — see app/oauth/providers.py for the provider registry this drives
against, and app/routers/oauth.py for the HTTP surface.

Row shape (PK USER#{user_id}, SK PROVIDER#{name}):
  {access_token: <encrypted>, refresh_token: <encrypted or None>,
   expires_at: <iso8601 or None (never expires, e.g. GitHub's classic
   OAuth App tokens)>, scopes: [str], connected_at: iso8601}

Every function here is "optional, additive, never a hard failure" for
its CALLER — a bad/expired/unrefreshable token degrades to "this
provider's tools just aren't available this turn," the same philosophy
mcp_client.py already applies to an unreachable static MCP server.
"""
from __future__ import annotations

import datetime as dt
import logging

import httpx

from . import crypto
from .db import store
from .oauth.providers import get_provider

logger = logging.getLogger(__name__)

_TABLE = "OAuthConnections"


def _pk(user_id: str) -> str:
    return f"USER#{user_id}"


def _sk(provider: str) -> str:
    return f"PROVIDER#{provider}"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_connection(user_id: str, provider: str, *, access_token: str,
                    refresh_token: str | None, expires_in: int | None,
                    scopes: list[str]) -> None:
    expires_at = None
    if expires_in:
        expires_at = (dt.datetime.now(dt.timezone.utc) +
                      dt.timedelta(seconds=expires_in)).isoformat()
    row = {
        "access_token": crypto.encrypt(access_token),
        "refresh_token": crypto.encrypt(refresh_token) if refresh_token else None,
        "expires_at": expires_at,
        "scopes": scopes,
        "connected_at": _now_iso(),
    }
    store(_TABLE).put(_pk(user_id), _sk(provider), row)


def get_connection(user_id: str, provider: str) -> dict | None:
    return store(_TABLE).get(_pk(user_id), _sk(provider))


def list_connections(user_id: str) -> list[dict]:
    return store(_TABLE).scan(_pk(user_id), "PROVIDER#")


def delete_connection(user_id: str, provider: str) -> None:
    store(_TABLE).delete(_pk(user_id), _sk(provider))


def _is_expired(row: dict) -> bool:
    if not row.get("expires_at"):
        return False  # never expires (e.g. GitHub classic OAuth App tokens)
    expires_at = dt.datetime.fromisoformat(row["expires_at"])
    # A minute of slack so a token doesn't expire mid-call.
    return dt.datetime.now(dt.timezone.utc) >= expires_at - dt.timedelta(minutes=1)


def exchange_code(provider_name: str, code: str, redirect_uri: str) -> dict:
    """Authorization Code → token. Raises on a real failure — the caller
    (the callback route) is expected to turn that into a user-facing
    ?error= redirect, not swallow it silently."""
    p = get_provider(provider_name)
    headers = {"Accept": "application/json"}  # GitHub defaults to form-encoded without this
    data = {"client_id": p.client_id, "client_secret": p.client_secret,
            "code": code, "redirect_uri": redirect_uri, "grant_type": "authorization_code"}
    r = httpx.post(p.token_url, data=data, headers=headers, timeout=10)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise ValueError(f"{provider_name} rejected the authorization code: {body}")
    return body


def _refresh(provider_name: str, refresh_token: str) -> dict | None:
    p = get_provider(provider_name)
    headers = {"Accept": "application/json"}
    data = {"client_id": p.client_id, "client_secret": p.client_secret,
            "refresh_token": refresh_token, "grant_type": "refresh_token"}
    try:
        r = httpx.post(p.token_url, data=data, headers=headers, timeout=10)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise ValueError(body["error"])
        return body
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Refreshing %s token failed (%s) — connection needs reconnecting", provider_name, e)
        return None


def get_valid_access_token(user_id: str, provider: str) -> str | None:
    row = get_connection(user_id, provider)
    if not row:
        return None
    if not _is_expired(row):
        return crypto.decrypt(row["access_token"])
    refresh_token_enc = row.get("refresh_token")
    if not refresh_token_enc:
        logger.warning("%s token for user %s expired with no refresh_token — reconnect required",
                       provider, user_id)
        return None
    refreshed = _refresh(provider, crypto.decrypt(refresh_token_enc))
    if not refreshed:
        return None
    new_access = refreshed["access_token"]
    # Some providers omit refresh_token on a refresh response (it's still
    # valid — reuse the one already stored rather than dropping it).
    new_refresh = refreshed.get("refresh_token") or crypto.decrypt(refresh_token_enc)
    save_connection(user_id, provider, access_token=new_access, refresh_token=new_refresh,
                    expires_in=refreshed.get("expires_in"), scopes=row.get("scopes", []))
    return new_access


def revoke_connection(user_id: str, provider: str) -> None:
    """Best-effort revoke with the provider, then always deletes the local
    row regardless of whether the remote revoke succeeded — a failed
    revoke call must never block the user from disconnecting locally."""
    row = get_connection(user_id, provider)
    if row:
        p = get_provider(provider)
        try:
            access_token = crypto.decrypt(row["access_token"])
            if provider == "github":
                # GitHub's real revoke shape: DELETE to a REST path with
                # Basic auth (client_id:client_secret), not a token-revoke
                # POST — see docs.github.com/en/rest/apps/oauth-applications.
                httpx.request(
                    "DELETE", f"https://api.github.com/applications/{p.client_id}/token",
                    auth=(p.client_id, p.client_secret), json={"access_token": access_token},
                    timeout=10)
            elif p.revoke_url:
                httpx.post(p.revoke_url, data={"token": access_token}, timeout=10)
            # Linear publishes no public revoke endpoint — local
            # disconnect only, honestly, rather than guessing one.
        except httpx.HTTPError as e:
            logger.warning("Revoking %s token for user %s failed (%s) — deleting local "
                           "connection anyway", provider, user_id, e)
    delete_connection(user_id, provider)
