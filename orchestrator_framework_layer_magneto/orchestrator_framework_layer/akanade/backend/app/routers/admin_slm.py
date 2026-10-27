"""Local, inspectable version history for the SLM Generation Layer
(README_SLM_GENERATION_LAYER.md) — every nightly
convert_and_register_ollama_model.sh run reports its outcome here
(sanity check / regression gate / whether it was actually registered),
success or failure, so the Admin dashboard can show which
Algo-Akanade-SLM version is live and what happened on prior nights,
without reaching into the deploy box's filesystem. Recording a new
version (POST) is Executive-role only, same bar as admin_providers.py —
this is part of the same "which model is actually answering" surface.
Reading is Admin-role (admits the real app's Manager2/"Senior Manager",
per auth.py's _REAL_ROLE_MAP) — version history isn't a secret the way
admin_providers.py's api_key is.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from ..auth import require_role
from ..db import store
from ..models import SlmVersionIn, SlmVersionOut

router = APIRouter(prefix="/api/admin/slm-versions", tags=["admin-slm-versions"])


def _to_out(row: dict) -> SlmVersionOut:
    return SlmVersionOut(
        id=row["id"], tag=row["tag"], base_tag=row["baseTag"], quant_type=row["quantType"],
        sanity_check=row["sanityCheck"], regression_gate=row.get("regressionGate"),
        registered=row.get("registered", False), domain=row.get("domain"),
        created_at=row["createdAt"])


@router.post("", response_model=SlmVersionOut, status_code=201)
def record_slm_version(payload: SlmVersionIn, user: dict = Depends(require_role("Executive"))):
    version_id = uuid.uuid4().hex[:12]
    row = {
        "id": version_id,
        "tag": payload.tag,
        "baseTag": payload.base_tag,
        "quantType": payload.quant_type,
        "sanityCheck": payload.sanity_check,
        "regressionGate": payload.regression_gate,
        "registered": payload.registered,
        "domain": payload.domain,
        "createdBy": user["sub"],
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    store("SlmVersions").put(f"SLMVERSION#{version_id}", "META", row)
    return _to_out(row)


@router.get("", response_model=list[SlmVersionOut])
def list_slm_versions(_: dict = Depends(require_role("Admin"))):
    """Newest first, so the Admin dashboard's top row per `domain`
    (None = the general-purpose provider) is always the current
    candidate for THAT domain — `registered=True` on the first row for
    a given domain is the version actually answering chat turns right
    now; a run further down with the same domain that's also
    `registered=True` was superseded by a later night, not still live
    (registration is upsert-by-name PER DOMAIN, see the shell script —
    only ONE provider named "algo-akanade-slm[-<domain>]" ever exists
    for any given domain)."""
    rows = store("SlmVersions").scan("SLMVERSION#")
    return [_to_out(r) for r in sorted(rows, key=lambda r: r["createdAt"], reverse=True)]
