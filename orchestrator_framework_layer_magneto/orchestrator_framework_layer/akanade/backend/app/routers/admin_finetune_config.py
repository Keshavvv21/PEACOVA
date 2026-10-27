"""What SHOULD get trained tonight — one row per domain (see
models.FinetuneConfigIn), editable here instead of hand-editing shell-
script env vars on the deploy box. deploy/nightly_finetune.sh fetches
every enabled row via GET .../enabled at 03:30 UTC and runs the SLM
Generation Layer's pipeline once per row — this is the "choose any
base model, run multiple domains, configurable from the UI" surface
(README_SLM_GENERATION_LAYER.md §5). Mutating (PUT/DELETE) is Executive-
role only, same bar as admin_providers.py/admin_slm.py — this is part of
the same "which model ends up answering" surface. Listing is Admin-role
(admits the real app's Manager2/"Senior Manager", per auth.py's
_REAL_ROLE_MAP) — nothing here is a secret, unlike admin_providers.py's
api_key.

Keyed by `domain` (None normalized to the literal string "general" for
the URL path) rather than a generated id — PUT is a genuine upsert:
there is at most ONE config per domain, mirroring
convert_and_register_ollama_model.sh's own "one provider per domain,
ever" invariant.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_role
from ..db import store
from ..models import FinetuneConfigIn, FinetuneConfigOut

router = APIRouter(prefix="/api/admin/finetune-config", tags=["admin-finetune-config"])

_GENERAL_KEY = "general"


def _key_for(domain: str | None) -> str:
    return domain or _GENERAL_KEY


def _to_out(row: dict) -> FinetuneConfigOut:
    # .get() with FinetuneConfigIn's OWN field defaults, not direct
    # indexing — a real bug this surfaced: a row written with only a
    # subset of fields (e.g. peft_analyzer.py's own tests, which store a
    # bare {id, enabled, scrapeTopics} row into this SAME shared table
    # purely to exercise scrape-topic lookup, never through this
    # router's own upsert_config()) raised a bare KeyError here and 500'd
    # every OTHER caller of GET ""/"" /enabled for the rest of the
    # process — one incomplete row should never take down the whole
    # listing.
    return FinetuneConfigOut(
        id=row["id"], domain=row.get("domain"),
        base_model=row.get("baseModel", FinetuneConfigIn.model_fields["base_model"].default),
        ollama_base_tag=row.get("ollamaBaseTag", FinetuneConfigIn.model_fields["ollama_base_tag"].default),
        serve_backend=row.get("serveBackend", "ollama"),
        max_minutes=row.get("maxMinutes", FinetuneConfigIn.model_fields["max_minutes"].default),
        enabled=row.get("enabled", True),
        scrape_topics=row.get("scrapeTopics", []),
        created_at=row.get("createdAt", ""), updated_at=row.get("updatedAt", ""))


@router.get("", response_model=list[FinetuneConfigOut])
def list_configs(_: dict = Depends(require_role("Admin"))):
    rows = store("FinetuneConfigs").scan("FINETUNECFG#")
    return [_to_out(r) for r in sorted(rows, key=lambda r: r["id"])]


@router.get("/enabled", response_model=list[FinetuneConfigOut])
def list_enabled_configs(_: dict = Depends(require_role("Admin"))):
    """What deploy/nightly_finetune.sh actually fetches each run. A
    caller that finds this empty (e.g. a fresh deploy, nobody's
    configured anything yet) falls back to its own FINETUNE_BASE_MODEL/
    OLLAMA_BASE_TAG env-var defaults — see that script's own comment —
    so an unconfigured deployment keeps training the single
    general-purpose model it always did, not nothing."""
    rows = store("FinetuneConfigs").scan("FINETUNECFG#")
    return [_to_out(r) for r in sorted(rows, key=lambda r: r["id"]) if r.get("enabled", True)]


@router.put("/{key}", response_model=FinetuneConfigOut)
def upsert_config(key: str, payload: FinetuneConfigIn, _: dict = Depends(require_role("Executive"))):
    expected_key = _key_for(payload.domain)
    if key != expected_key:
        raise HTTPException(400, f"URL key {key!r} doesn't match domain {payload.domain!r} "
                                 f"(expected {expected_key!r}) — PUT /general for the "
                                 f"general-purpose model, /<domain> for a domain variant.")
    existing = store("FinetuneConfigs").get(f"FINETUNECFG#{key}", "META")
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": key,
        "domain": payload.domain,
        "baseModel": payload.base_model,
        "ollamaBaseTag": payload.ollama_base_tag,
        "serveBackend": payload.serve_backend,
        "maxMinutes": payload.max_minutes,
        "enabled": payload.enabled,
        "scrapeTopics": payload.scrape_topics,
        "createdAt": existing["createdAt"] if existing else now,
        "updatedAt": now,
    }
    store("FinetuneConfigs").put(f"FINETUNECFG#{key}", "META", row)
    return _to_out(row)


@router.delete("/{key}", status_code=204)
def delete_config(key: str, _: dict = Depends(require_role("Executive"))):
    if not store("FinetuneConfigs").get(f"FINETUNECFG#{key}", "META"):
        raise HTTPException(404, "No finetune config for that key")
    store("FinetuneConfigs").delete(f"FINETUNECFG#{key}", "META")
