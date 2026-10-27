"""This dashboard's own login — deliberately independent of Algo Akanade's
Users table/JWT secret (a real "separate UI and password", not a thin
wrapper over the same account system). One admin identity, configured
via DASHBOARD_ADMIN_USER/DASHBOARD_ADMIN_PASSWORD (config.py); same
sha256+PyJWT shape Algo Akanade's own app/auth.py uses, so anyone who already
knows that codebase recognizes this instantly, just pointed at its own
secret/table-free single account instead of a Users store.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac

import jwt
from fastapi import Header, HTTPException

from .config import settings


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def verify_login(username: str, password: str) -> bool:
    s = settings()
    return (hmac.compare_digest(username, s.DASHBOARD_ADMIN_USER)
           and hmac.compare_digest(_hash(password), _hash(s.DASHBOARD_ADMIN_PASSWORD)))


def make_token(username: str) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    s = settings()
    return jwt.encode(
        {"sub": username, "iat": now, "exp": now + dt.timedelta(hours=s.DASHBOARD_JWT_TTL_HOURS)},
        s.DASHBOARD_JWT_SECRET, algorithm="HS256")


def current_admin(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, settings().DASHBOARD_JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid or expired token")
    return payload["sub"]
