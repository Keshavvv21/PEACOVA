"""The Optimization Dashboard's API — a small, standalone FastAPI service
(its own login, see auth.py) that turns Algo Akanade's optimization event log
and the Algo Engine's live stats into the numbers an admin actually wants
to see: is the RL-driven optimization layer helping, is it getting
better day over day, and is the GPU tier real. `uvicorn app.main:app`.

See README_OPTIMIZATION_DASHBOARD.md (one level up) for the full
architecture and how to run this alongside Algo Akanade + Algo Engine.
"""
from __future__ import annotations

import datetime as dt

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import events_store
from .auth import current_admin, make_token, verify_login
from .engine_client import KERNEL_TIERS, fetch_engine_stats

app = FastAPI(title="Optimization Dashboard", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LoginIn(BaseModel):
    username: str
    password: str


class LoginOut(BaseModel):
    access_token: str
    username: str


@app.post("/api/login", response_model=LoginOut)
def login(body: LoginIn):
    if not verify_login(body.username, body.password):
        raise HTTPException(401, "bad credentials")
    return LoginOut(access_token=make_token(body.username), username=body.username)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
async def summary(_: str = Depends(current_admin)):
    """The dashboard's landing view: today so far, the GPU/kernel panel,
    and provider_optimizer's persisted RL posteriors (see
    events_store.provider_optimizer_posteriors's docstring for why that's
    the one optimizer whose raw posterior survives a restart)."""
    today = dt.date.today().isoformat()
    return {
        "date": today,
        "today": events_store.summarize(events_store.list_events_for_date(today)),
        "recent_dates": events_store.list_recent_dates(),
        "provider_rl_posteriors": events_store.provider_optimizer_posteriors(),
        "gpu": {
            "kernel_tiers": KERNEL_TIERS,
            "live": await fetch_engine_stats(),
        },
    }


@app.get("/api/daily")
def daily(date: str | None = None, _: str = Depends(current_admin)):
    """`date` defaults to today (UTC). Returns that day's rollup plus the
    delta against the day before — the "daily improvement" view."""
    target = date or dt.date.today().isoformat()
    try:
        dt.date.fromisoformat(target)
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    return events_store.daily_report(target)


@app.get("/api/events")
def events(date: str | None = None, _: str = Depends(current_admin)):
    target = date or dt.date.today().isoformat()
    try:
        dt.date.fromisoformat(target)
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    return {"date": target, "events": events_store.list_events_for_date(target)}
