"""JWT auth + role dependencies — the require_role pattern the design
reuses (§2.1, §11). The bot never gets its own privileges: tools carry
the calling user's JWT into the same service functions the REST router
uses, so every role check fires identically for a chat turn and a click.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time
from typing import Callable

import httpx
import jwt
from fastapi import Depends, HTTPException, Request

from .config import settings
from .models import Role

ROLE_ORDER: dict[Role, int] = {"Member": 0, "Manager": 1, "Admin": 2, "Executive": 3}

# Real, observed cost this closes: in REAL_APP_API_BASE_URL mode (this
# deployment), EVERY authenticated request — every chat turn from VS
# Code, mobile, or the web app, not just login — did a live, synchronous
# HTTP round trip to ak-backend's own GET /api/users/me just to validate
# the bearer token, before any actual work (an LLM call, a session
# load, ...) even started. A short-TTL in-process cache keyed by a hash
# of the token (never the raw token itself, as a small extra precaution
# against it lingering in memory dumps/logs) turns that into one real
# validation per ~30s of a user's activity instead of one per request —
# single-process deployment (bare `uvicorn app.main:app`, no multi-
# worker), so no shared cache (Redis or otherwise) is needed for this to
# work; see auth's own module docstring / README discussion on when
# Redis would actually start earning its keep here (multiple backend
# replicas needing to share this cache).
_USER_CACHE_TTL_SECONDS = 30.0
# Capped, not unbounded — same "prune under real growth, don't chase
# every individual expiry" convention as elsewhere in this codebase
# (framework/agents/coding_agent.py's _PENDING_TOOL_CALLS, etc.). A
# lightly-trafficked deployment never gets remotely close to this.
_USER_CACHE_MAX_ENTRIES = 500
_user_cache: dict[str, tuple[float, dict]] = {}


def _prune_user_cache(now: float) -> None:
    if len(_user_cache) <= _USER_CACHE_MAX_ENTRIES:
        return
    for key in [k for k, (expiry, _) in _user_cache.items() if expiry <= now]:
        del _user_cache[key]

# The real app's global roles (Admin > Manager2 > Manager1 > Member) mapped
# onto this bot's four-tier hierarchy above, preserving relative order —
# used only when REAL_APP_API_BASE_URL is configured (see current_user()).
_REAL_ROLE_MAP: dict[str, Role] = {
    "Admin": "Executive", "Manager2": "Admin", "Manager1": "Manager", "Member": "Member",
}


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def make_token(user_id: str, name: str, role: Role) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return jwt.encode(
        {"sub": user_id, "name": name, "role": role,
         "iat": now, "exp": now + dt.timedelta(hours=settings().JWT_TTL_HOURS)},
        settings().JWT_SECRET, algorithm=settings().JWT_ALG)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings().JWT_SECRET,
                          algorithms=[settings().JWT_ALG])
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"invalid token: {e}")


def _real_app_user(token: str) -> dict:
    """REAL_APP_API_BASE_URL mode: the incoming token is the REAL app's own
    JWT, not this bot's — it's never decoded here (the two apps' secrets
    differ and shouldn't be shared). Instead it's validated the same way
    any other client would: call the real app's own GET /api/users/me with
    it. A rejected/expired token 401s there and surfaces as a 401 here too;
    a real name/role means the whole rest of the turn — the "signed in as"
    line, every _need() role check — reflects the actual signed-in person,
    not a fixed demo account."""
    base = settings().REAL_APP_API_BASE_URL
    try:
        r = httpx.get(f"{base}/api/users/me",
                      headers={"Authorization": f"Bearer {token}"}, timeout=5)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(401, f"real app rejected this token: {e}")
    me = r.json()
    return {"sub": me["user_id"], "name": me["name"],
           "role": _REAL_ROLE_MAP.get(me["role"], "Member"),
           # Carried along so tools.py's real-data calls can forward the
           # SAME token to the real app's other endpoints — this bot never
           # gets its own elevated credential, same rule as demo mode.
           "_real_token": token}


def _cached_real_app_user(token: str) -> dict:
    now = time.monotonic()
    key = hashlib.sha256(token.encode()).hexdigest()
    cached = _user_cache.get(key)
    if cached and cached[0] > now:
        return cached[1]
    user = _real_app_user(token)
    _prune_user_cache(now)
    _user_cache[key] = (now + _USER_CACHE_TTL_SECONDS, user)
    return user


def current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = auth.removeprefix("Bearer ")
    if settings().REAL_APP_API_BASE_URL:
        return _cached_real_app_user(token)
    return decode_token(token)


def require_role(minimum: Role) -> Callable:
    """Dependency: caller's role must be >= minimum in the hierarchy."""
    def dep(user: dict = Depends(current_user)) -> dict:
        if ROLE_ORDER[user["role"]] < ROLE_ORDER[minimum]:
            raise HTTPException(403, f"requires {minimum}+ role")
        return user
    return dep
