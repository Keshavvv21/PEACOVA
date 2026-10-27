"""Admin-managed LLM providers — add a new model (name, base URL, model id,
API key) via the UI and it's live on the very next chat turn, no redeploy;
toggle it on/off the same way. See llm/router.py's _db_provider_entries()
for how these get spliced into the normal tier chains alongside the static
EXTRA_LLM_PROVIDERS env var — same mechanism, just admin-editable at runtime.

Mutating this (create/update/delete) is Executive-role only (the real app's
actual Admin — see auth.py's _REAL_ROLE_MAP): it stores real third-party API
keys, a step above the Admin-level actions (create_user, etc.) elsewhere in
this bot. Listing/reading is Admin-role (which _REAL_ROLE_MAP already maps
the real app's Manager2/"Senior Manager" onto) — safe to view-gate one level
lower than write, since ProviderOut never serializes the raw api_key at all
(only api_key_set: bool), so a Senior Manager can see WHICH models exist and
whether they're configured without ever seeing a secret.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_role
from ..db import store
from ..framework import model_benchmark_status
from ..models import ProviderIn, ProviderOut, ProviderUpdate
from .admin_model_benchmarks import run_benchmark_sweep

router = APIRouter(prefix="/api/admin/providers", tags=["admin-providers"])


def _to_out(row: dict) -> ProviderOut:
    return ProviderOut(
        id=row["id"],
        name=row["name"],
        base_url=row["baseUrl"],
        model=row.get("model", ""),
        api_key_set=bool(row.get("apiKey")),
        tiers=row.get("tiers") or ["realtime", "bulk"],
        enabled=row.get("enabled", True),
        capability_tier=row.get("capabilityTier", 0.3),
        created_at=row["createdAt"],
    )


@router.post("", response_model=ProviderOut, status_code=201)
def create_provider(payload: ProviderIn, user: dict = Depends(require_role("Executive"))):
    provider_id = uuid.uuid4().hex[:12]
    row = {
        "id": provider_id,
        "name": payload.name,
        "baseUrl": payload.base_url,
        "model": payload.model,
        "apiKey": payload.api_key,
        "tiers": payload.tiers,
        "enabled": True,
        "capabilityTier": payload.capability_tier,
        "createdBy": user["sub"],
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    store("Providers").put(f"PROVIDER#{provider_id}", "META", row)
    # Auto-trigger a benchmark sweep for the new model — the "automatically
    # tests new models against our workload" requirement. run_benchmark_sweep()
    # starts its own detached daemon thread and returns immediately — this
    # request (and, just as importantly, any test that calls this endpoint)
    # is never held up waiting on a real, multi-minute harness run. Best-
    # effort: a missing eval venv degrades to a failed job row (see
    # admin_model_benchmarks.py's own docstring), never blocks provider
    # creation itself.
    job_id = model_benchmark_status.start_job(model_source="ours", triggered_by="auto:create_provider")
    run_benchmark_sweep(job_id, "ours")
    return _to_out(row)


@router.get("", response_model=list[ProviderOut])
def list_providers(_: dict = Depends(require_role("Admin"))):
    rows = store("Providers").scan("PROVIDER#")
    return [_to_out(r) for r in sorted(rows, key=lambda r: r["createdAt"])]


@router.patch("/{provider_id}", response_model=ProviderOut)
def update_provider(provider_id: str, payload: ProviderUpdate,
                    _: dict = Depends(require_role("Executive"))):
    row = store("Providers").get(f"PROVIDER#{provider_id}", "META")
    if not row:
        raise HTTPException(404, "Provider not found")
    changes = payload.model_dump(exclude_unset=True)
    if "base_url" in changes:
        row["baseUrl"] = changes.pop("base_url")
    if "api_key" in changes:
        row["apiKey"] = changes.pop("api_key")
    if "capability_tier" in changes:
        row["capabilityTier"] = changes.pop("capability_tier")
    row.update(changes)
    store("Providers").put(f"PROVIDER#{provider_id}", "META", row)
    return _to_out(row)


@router.delete("/{provider_id}", status_code=204)
def delete_provider(provider_id: str, _: dict = Depends(require_role("Executive"))):
    if not store("Providers").get(f"PROVIDER#{provider_id}", "META"):
        raise HTTPException(404, "Provider not found")
    store("Providers").delete(f"PROVIDER#{provider_id}", "META")
