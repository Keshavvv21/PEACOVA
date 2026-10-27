"""Per-user 'Connect your account' OAuth 2.0 — GitHub/Linear/Google. See
app/oauth_store.py (token storage/lifecycle), app/oauth/providers.py
(the static provider registry), app/oauth/state.py (the signed `state`
CSRF mechanism the callback relies on since it can't carry a bearer
token), and app/agent/user_mcp_client.py (where a connection actually
becomes usable tools in a chat turn).
"""
from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse

from .. import oauth_store
from ..agent import user_mcp_client
from ..auth import current_user
from ..config import settings
from ..oauth.providers import all_providers, get_provider
from ..oauth.state import sign_state, verify_state

router = APIRouter(prefix="/api/oauth", tags=["oauth"])


def _redirect_uri(provider: str) -> str:
    return f"{settings().OAUTH_REDIRECT_BASE_URL}/api/oauth/{provider}/callback"


def _require_known_provider(provider: str) -> None:
    if provider not in all_providers():
        raise HTTPException(404, f"unknown provider {provider!r}")


@router.get("/{provider}/authorize")
def authorize(provider: str, user: dict = Depends(current_user)):
    _require_known_provider(provider)
    p = get_provider(provider)
    if not p.configured:
        raise HTTPException(
            409, f"{provider} OAuth isn't configured on this deployment yet "
                f"(missing client id/secret) — ask an admin to set it up.")
    params = {
        "client_id": p.client_id,
        "redirect_uri": _redirect_uri(provider),
        "response_type": "code",
        "scope": " ".join(p.scopes),
        "state": sign_state(user["sub"], provider),
        **p.extra_authorize_params,
    }
    return {"url": f"{p.authorize_url}?{urlencode(params)}"}


@router.get("/{provider}/callback")
def callback(provider: str, code: str | None = Query(None), state: str | None = Query(None),
            error: str | None = Query(None)):
    _require_known_provider(provider)
    target = f"{settings().FRONTEND_BASE_URL}/account/integrations"
    if error:
        return RedirectResponse(f"{target}?error={error}")
    if not code or not state:
        return RedirectResponse(f"{target}?error=missing_code_or_state")
    try:
        user_id = verify_state(state, provider)
    except HTTPException:
        return RedirectResponse(f"{target}?error=invalid_state")
    p = get_provider(provider)
    try:
        token_body = oauth_store.exchange_code(provider, code, _redirect_uri(provider))
    except Exception:  # noqa: BLE001 — any exchange failure is just "connection failed"
        return RedirectResponse(f"{target}?error=token_exchange_failed")
    oauth_store.save_connection(
        user_id, provider,
        access_token=token_body["access_token"],
        refresh_token=token_body.get("refresh_token"),
        expires_in=token_body.get("expires_in"),
        scopes=(token_body.get("scope", "").split(",") if "," in token_body.get("scope", "")
                else token_body.get("scope", "").split()) or p.scopes,
    )
    user_mcp_client.invalidate(user_id)
    return RedirectResponse(f"{target}?connected={provider}")


@router.get("/connections")
def list_connections(user: dict = Depends(current_user)):
    rows = oauth_store.list_connections(user["sub"])
    # Never return access_token/refresh_token, same "set, not the value"
    # rule admin_providers.py's _to_out() already follows for api_key_set.
    return [{"provider": r["SK"].removeprefix("PROVIDER#"), "connected_at": r["connected_at"],
            "scopes": r.get("scopes", [])} for r in rows]


@router.delete("/connections/{provider}", status_code=204)
def disconnect(provider: str, user: dict = Depends(current_user)):
    _require_known_provider(provider)
    oauth_store.revoke_connection(user["sub"], provider)
    user_mcp_client.invalidate(user["sub"])
