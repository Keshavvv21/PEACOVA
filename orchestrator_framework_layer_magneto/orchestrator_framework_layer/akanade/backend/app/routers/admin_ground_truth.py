"""Ground Truth Validation — a daily record of comparing the freshly
fine-tuned candidate's own answers against a reference answer Groq (a
strong hosted model) generated on the spot for the same prompt, drawn
from that night's own new curated training data. See
app.agent.ground_truth_validator for who actually produces these rows
(run inside the always-on chatbot image, in-process — it calls
`create_run` below directly, no HTTP round trip needed) and
app.models.GroundTruthRunItem's docstring for why the score is a cheap
embedding cosine similarity, not an LLM judge.

Recording a run (POST) is Executive-role only, same bar as
admin_eval_reports.py. Reading runs is Admin-role — match scores and
prompt/answer text aren't secrets the way admin_providers.py's api_key
is.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_role
from ..db import store
from ..models import GroundTruthRunIn, GroundTruthRunOut, GroundTruthRunSummaryOut

router = APIRouter(prefix="/api/admin/ground-truth-runs", tags=["admin-ground-truth-runs"])


def _to_out(row: dict) -> GroundTruthRunOut:
    return GroundTruthRunOut(
        id=row["id"], domain=row.get("domain"), provider=row["provider"],
        items=row.get("items") or [], avg_match_score=row["avgMatchScore"],
        created_at=row["createdAt"])


def _to_summary(row: dict) -> GroundTruthRunSummaryOut:
    items = row.get("items") or []
    return GroundTruthRunSummaryOut(
        id=row["id"], domain=row.get("domain"), provider=row["provider"],
        n_items=len(items), avg_match_score=row["avgMatchScore"], created_at=row["createdAt"])


def create_run(payload: GroundTruthRunIn) -> GroundTruthRunOut:
    """The actual persistence logic, split out from the POST handler
    below so app.agent.ground_truth_validator (running in-process inside
    the same app image) can call this directly — no HTTP round trip or
    admin token needed for a module that already lives inside the
    process that would otherwise be calling its own API."""
    run_id = uuid.uuid4().hex[:12]
    items = [item.model_dump() for item in payload.items]
    avg_match_score = round(sum(i["match_score"] for i in items) / len(items), 4) if items else 0.0
    row = {
        "id": run_id,
        "domain": payload.domain,
        "provider": payload.provider,
        "items": items,
        "avgMatchScore": avg_match_score,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    store("GroundTruthRuns").put(f"GTRUN#{run_id}", "META", row)
    return _to_out(row)


@router.post("", response_model=GroundTruthRunOut, status_code=201)
def record_ground_truth_run(payload: GroundTruthRunIn, _: dict = Depends(require_role("Executive"))):
    return create_run(payload)


@router.get("", response_model=list[GroundTruthRunSummaryOut])
def list_ground_truth_runs(_: dict = Depends(require_role("Admin"))):
    """Newest first, summaries only — GET .../{id} for the full
    per-prompt breakdown of one specific day's run."""
    rows = store("GroundTruthRuns").scan("GTRUN#")
    return [_to_summary(r) for r in sorted(rows, key=lambda r: r["createdAt"], reverse=True)]


@router.get("/{run_id}", response_model=GroundTruthRunOut)
def get_ground_truth_run(run_id: str, _: dict = Depends(require_role("Admin"))):
    row = store("GroundTruthRuns").get(f"GTRUN#{run_id}", "META")
    if not row:
        raise HTTPException(404, "No ground-truth validation run with that id")
    return _to_out(row)
