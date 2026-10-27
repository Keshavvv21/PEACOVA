"""JWT auth + role dependencies — the require_role pattern the design
reuses (§2.1, §11). The bot never gets its own privileges: tools carry
the calling user's JWT into the same service functions the REST router
uses, so every role check fires identically for a chat turn and a click.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from typing import Callable

import jwt
from fastapi import Depends, HTTPException, Request

from .config import settings
from .models import Role

ROLE_ORDER: dict[Role, int] = {"Member": 0, "Manager": 1, "Admin": 2, "Executive": 3}


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


def current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    return decode_token(auth.removeprefix("Bearer "))


def require_role(minimum: Role) -> Callable:
    """Dependency: caller's role must be >= minimum in the hierarchy."""
    def dep(user: dict = Depends(current_user)) -> dict:
        if ROLE_ORDER[user["role"]] < ROLE_ORDER[minimum]:
            raise HTTPException(403, f"requires {minimum}+ role")