"""The daily verification report for the SLM Generation Layer — every
regression-gate run (scripts/eval/regression_gate.py, invoked from
convert_and_register_ollama_model.sh right after a candidate passes its
sanity check) reports here, win or lose, so the Admin dashboard can
show exactly which test prompts tonight's candidate was actually run
against and how it scored on each — the same "a real report an
operator can read without SSHing into the deploy box" role
admin_security.py's daily digest already plays for security events.
Recording a new report (POST) is Executive-role only, same bar as
admin_slm.py — this is the detailed evidence behind that router's own
`regression_gate` field. Reading reports is Admin-role (admits the real
app's Manager2/"Senior Manager", per auth.py's _REAL_ROLE_MAP) — eval
scores and prompt results aren't secrets the way admin_providers.py's
api_key is.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_role
from ..db import store
from ..models import SlmEvalItemResult, SlmEvalReportIn, SlmEvalReportOut, SlmEvalReportSummaryOut

router = APIRouter(prefix="/api/admin/slm-eval-reports", tags=["admin-slm-eval-reports"])


def _to_out(row: dict) -> SlmEvalReportOut:
    return SlmEvalReportOut(
        id=row["id"], tag=row["tag"], base_tag=row["baseTag"], domain=row.get("domain"),
        tolerance=row["tolerance"], old_score=row["oldScore"], new_score=row["newScore"],
        passed=row["passed"],
        old_by_category=row.get("oldByCategory") or {}, new_by_category=row.get("newByCategory") or {},
        items=[SlmEvalItemResult(**item) for item in row.get("items") or []],
        created_at=row["createdAt"])


def _to_summary(row: dict) -> SlmEvalReportSummaryOut:
    items = row.get("items") or []
    return SlmEvalReportSummaryOut(
        id=row["id"], tag=row["tag"], base_tag=row["baseTag"], domain=row.get("domain"),
        tolerance=row["tolerance"], old_score=row["oldScore"], new_score=row["newScore"],
        passed=row["passed"], n_items=len(items),
        n_item_errors=sum(1 for i in items if i.get("error")), created_at=row["createdAt"])


@router.post("", response_model=SlmEvalReportOut, status_code=201)
def record_eval_report(payload: SlmEvalReportIn, _: dict = Depends(require_role("Executive"))):
    report_id = uuid.uuid4().hex[:12]
    row = {
        "id": report_id,
        "tag": payload.tag,
        "baseTag": payload.base_tag,
        "domain": payload.domain,
        "tolerance": payload.tolerance,
        "oldScore": payload.old_score,
        "newScore": payload.new_score,
        "passed": payload.passed,
        "oldByCategory": payload.old_by_category,
        "newByCategory": payload.new_by_category,
        "items": [item.model_dump() for item in payload.items],
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    store("SlmEvalReports").put(f"SLMEVALREPORT#{report_id}", "META", row)
    return _to_out(row)


@router.get("", response_model=list[SlmEvalReportSummaryOut])
def list_eval_reports(_: dict = Depends(require_role("Admin"))):
    """Newest first, summaries only (no `items` — see
    SlmEvalReportSummaryOut's own docstring) — GET .../{id} for the
    full per-prompt breakdown of one specific night's report."""
    rows = store("SlmEvalReports").scan("SLMEVALREPORT#")
    return [_to_summary(r) for r in sorted(rows, key=lambda r: r["createdAt"], reverse=True)]


@router.get("/{report_id}", response_model=SlmEvalReportOut)
def get_eval_report(report_id: str, _: dict = Depends(require_role("Admin"))):
    row = store("SlmEvalReports").get(f"SLMEVALREPORT#{report_id}", "META")
    if not row:
        raise HTTPException(404, "No eval report with that id")
    return _to_out(row)
