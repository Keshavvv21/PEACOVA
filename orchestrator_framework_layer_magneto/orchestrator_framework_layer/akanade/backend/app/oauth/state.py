"""Signed `state` param for the OAuth authorize/callback round trip.

The callback (app/routers/oauth.py's GET /api/oauth/{provider}/callback)
is hit by a raw browser redirect from the provider — it CANNOT carry an
Authorization bearer header, so `state` is the only thing standing
between "this callback belongs to the user who clicked Connect" and any
attacker who can guess/observe a callback URL. A JWT (reusing PyJWT,
already a dependency — see auth.py) signed with the same JWT_SECRET,
scoped with its own `typ` claim so it can never be mistaken for (or
replayed as) a real login token, short-lived, and tied to the specific
provider it was issued for so a state minted for "github" can't be
replayed against the "google" callback.
"""
from __future__ import annotations

import datetime as dt

import jwt
from fastapi import HTTPException

from ..config import settings

_TYP = "oauth_state"
_TTL_MINUTES = 10


def sign_state(user_id: str, provider: str) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return jwt.encode(
        {"sub": user_id, "provider": provider, "typ": _TYP,
         "iat": now, "exp": now + dt.timedelta(minutes=_TTL_MINUTES)},
        settings().JWT_SECRET, algorithm=settings().JWT_ALG)


def verify_state(token: str, provider: str) -> str:
    """Returns the user_id the state was minted for, or raises 400 —
    every failure mode (expired, tampered, wrong provider, not an
    oauth_state token at all) is treated identically as "reject the
    callback," never partially trusted."""
    try:
        claims = jwt.decode(token, settings().JWT_SECRET, algorithms=[settings().JWT_ALG])
    except jwt.PyJWTError:
        raise HTTPException(400, "invalid or expired OAuth state")
    if claims.get("typ") != _TYP or claims.get("provider") != provider:
        raise HTTPException(400, "invalid OAuth state")
    return claims["sub"]
