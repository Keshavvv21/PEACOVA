"""Admin-toggled shadow signal for Algo aKANADE's OWN secondary pipeline
(deterministic.py + thompson.py) — NOT the primary/RouteLLM "Generic
Architecture" arm (generic_architecture.py), which already has its own
full mf/bert/sw_ranking/random cascade and needs nothing from this file.

deterministic.py orders candidates by expected cost alone; thompson.py
picks between exactly two arms by learned historical performance —
neither one has ever asked "how hard does THIS prompt actually look?"
This module adds that one missing signal, reusing the SAME lazily-loaded
"bert" classifier generic_architecture.py already loads for the primary
arm (see bert_tier_hint()) rather than loading a second copy of the
checkpoint into this process.

Same promotion gate shape as thompson.py's own live/shadow switch:
OFF by default, computed and logged for later comparison once an admin
opts in, and even then it NEVER reorders ordered_ids on its own — see
routes.py's /route handler. Real routing stays exactly as
deterministic/robust as it already was; this only ever adds a column
to compare against, until/unless a human decides otherwise.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .db import store

_CONFIG_PK = "BERT_SHADOW_CONFIG"
_CONFIG_SK = "SINGLETON"


def is_enabled() -> bool:
    row = store("BertShadowConfig").get(_CONFIG_PK, _CONFIG_SK)
    return bool((row or {}).get("enabled", False))


def set_enabled(enabled: bool, updated_by: str) -> dict:
    row = {"enabled": enabled, "updated_by": updated_by,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    store("BertShadowConfig").put(_CONFIG_PK, _CONFIG_SK, row)
    return row


def get_row() -> dict:
    row = store("BertShadowConfig").get(_CONFIG_PK, _CONFIG_SK)
    return row or {"enabled": False, "updated_by": None, "updated_at": None}


def hint(prompt: str | None) -> str | None:
    """The bert classifier's tier hint for this prompt ("default" or
    "escalation"), or None if the signal is off, no prompt was given,
    "bert" isn't available, or the call failed for any reason — this
    is a comparison signal, never a real decision, so it must never
    raise into routes.py's real /route hot path."""
    if not prompt or not is_enabled():
        return None
    try:
        from . import generic_architecture
        return generic_architecture.bert_tier_hint(prompt)
    except Exception:
        return None
