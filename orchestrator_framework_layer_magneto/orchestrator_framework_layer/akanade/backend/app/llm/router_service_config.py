"""The LOCAL (ak-chatbot-side) on/off switch for calling ak-router at
all — the first of the plan's two independent flags. Deliberately
separate from ak-router's own RouterConfig (deterministic/
thompson_sampling): this one has to be readable with NO network call,
so a decision to skip ak-router entirely never depends on ak-router
itself being reachable. Same DynamoDB-style single-row pattern as every
other admin toggle in this codebase (admin_providers.py's Providers
rows, provider_budget.py's per-provider budgets).

Defaults to DISABLED — a fresh deployment (or one that's never had an
admin touch this) behaves EXACTLY as it did before ak-router existed:
TierClient.chat() uses the fixed CHAINS order, zero new network
dependency, zero behavior change. This is the plan's own rollout step 1
("ship with ROUTER_SERVICE_ENABLED=false initially... verify the
fallback path first with zero behavior change to production") made the
actual default, not just a documented intention.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..db import store

_PK = "ROUTER_SERVICE_CONFIG"
_SK = "SINGLETON"


def is_enabled() -> bool:
    row = store("RouterServiceConfig").get(_PK, _SK)
    return bool((row or {}).get("enabled", False))


def set_enabled(enabled: bool, updated_by: str) -> dict:
    row = {"enabled": enabled, "updated_by": updated_by,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    store("RouterServiceConfig").put(_PK, _SK, row)
    return row


def get_row() -> dict:
    row = store("RouterServiceConfig").get(_PK, _SK)
    return row or {"enabled": False, "updated_by": None, "updated_at": None}
