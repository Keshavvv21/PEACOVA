"""Admin controls for ak-router — the two independent flags from the
plan (Deterministic Baseline Router + Router Service + Model
Benchmarking): whether to call ak-router at all, and (only relevant
while that's on) which strategy it runs live. Same Executive-write/
Admin-read role split as admin_providers.py — flipping either is a real
production routing-behavior change, not a cosmetic setting.

/router-strategy is a thin proxy to ak-router's own /admin/strategy —
that service deliberately has no auth of its own (see its routes.py
docstring: only reachable on the internal ak-net network), so THIS is
where the real role check happens before the request ever leaves
ak-chatbot.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response

from ..auth import require_role
from ..config import settings
from ..llm import coding_pipeline, primary_pipeline, router_service_config

router = APIRouter(prefix="/api/admin/router", tags=["admin-router"])

_RANGES = {"today": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}


@router.get("/service-config")
def get_service_config(_: dict = Depends(require_role("Admin"))) -> dict:
    return router_service_config.get_row()


@router.patch("/service-config")
def patch_service_config(payload: dict, user: dict = Depends(require_role("Executive"))) -> dict:
    if "enabled" not in payload or not isinstance(payload["enabled"], bool):
        raise HTTPException(422, "expected {\"enabled\": bool}")
    return router_service_config.set_enabled(payload["enabled"], user["sub"])


# --- Primary pipeline blue/green switch (llm/primary_pipeline.py) — which
# pipeline answers REAL live chat traffic: "routellm" (primary/default,
# the CEO memo's own generic architecture) or "akanade" (secondary,
# Algo aKANADE's own chain).
# Local to this process (no ak-router proxy needed), same RBAC split as
# everything else here — this is the single biggest lever in the whole
# admin surface, so it stays Executive-only to flip. ---

@router.get("/primary-pipeline")
def get_primary_pipeline(_: dict = Depends(require_role("Admin"))) -> dict:
    return primary_pipeline.get_pipeline_row()


@router.patch("/primary-pipeline")
def patch_primary_pipeline(payload: dict, user: dict = Depends(require_role("Executive"))) -> dict:
    if payload.get("pipeline") not in primary_pipeline.ROUTELLM_MODES + ("akanade",):
        raise HTTPException(422, "pipeline must be \"routellm\", \"routellm_only\", or \"akanade\"")
    return primary_pipeline.set_pipeline(payload["pipeline"], user["sub"])


@router.get("/strategy")
def get_strategy(_: dict = Depends(require_role("Admin"))) -> dict:
    try:
        resp = httpx.get(f"{settings().ROUTER_SERVICE_URL}/admin/strategy",
                         timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ak-router unreachable: {e}") from e


@router.patch("/strategy")
def patch_strategy(payload: dict, user: dict = Depends(require_role("Executive"))) -> dict:
    if payload.get("strategy") not in ("deterministic", "thompson_sampling"):
        raise HTTPException(422, "strategy must be \"deterministic\" or \"thompson_sampling\"")
    try:
        resp = httpx.patch(f"{settings().ROUTER_SERVICE_URL}/admin/strategy",
                           json={"strategy": payload["strategy"], "updated_by": user["sub"]},
                           timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ak-router unreachable: {e}") from e


# --- Coding Agent (VS Code) Secondary Pipeline — Escalation tier on/off
# (llm/coding_pipeline.py). Admin-write here, unlike the primary-pipeline
# switch above (Executive-only): this is a single scoped, per-agent cost
# lever, not the biggest switch in the whole admin surface, so a manager-
# tier admin (auth.py's _REAL_ROLE_MAP: a real "Manager2" role resolves to
# effective "Admin") can flip it without needing Executive access. ---

@router.get("/coding-escalation")
def get_coding_escalation(_: dict = Depends(require_role("Admin"))) -> dict:
    return coding_pipeline.get_escalation_config_row()


@router.patch("/coding-escalation")
def patch_coding_escalation(payload: dict, user: dict = Depends(require_role("Admin"))) -> dict:
    if "enabled" not in payload or not isinstance(payload["enabled"], bool):
        raise HTTPException(422, "expected {\"enabled\": bool}")
    return coding_pipeline.set_escalation_enabled(payload["enabled"], user["sub"])


# --- "GENERIC ARCHITECTURE" comparison arm (generic_architecture.py,
# RouteLLM's calibrated difficulty classifier) — same RBAC shape as
# everything above: triggering a real pick/outcome-record is Executive
# (same tier as record_slm_version/record_eval_report's own convention —
# it writes real data and, once real model keys are configured, spends
# real money), viewing is Admin. ---

@router.post("/generic-architecture/route")
def generic_route(payload: dict | None = None, user: dict = Depends(require_role("Executive"))) -> dict:
    try:
        resp = httpx.post(f"{settings().ROUTER_SERVICE_URL}/generic-architecture/route",
                          json=payload or {}, timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ak-router unreachable: {e}") from e


@router.post("/generic-architecture/escalate")
def generic_escalate(payload: dict | None = None, user: dict = Depends(require_role("Executive"))) -> dict:
    try:
        resp = httpx.post(f"{settings().ROUTER_SERVICE_URL}/generic-architecture/escalate",
                          json=payload or {}, timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ak-router unreachable: {e}") from e


@router.post("/generic-architecture/outcome")
def generic_outcome(payload: dict, user: dict = Depends(require_role("Executive"))) -> dict:
    try:
        resp = httpx.post(f"{settings().ROUTER_SERVICE_URL}/generic-architecture/outcome",
                          json=payload, timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        return {"status": "recorded"}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ak-router unreachable: {e}") from e


# --- Routing Comparison dashboard: Actual vs. the two shadow-only bandits
# (Thompson Sampling / UCB, both from thompson.py's single shared posterior).
# Shadow picks are hypothetical — the arm was never actually called for that
# request — so the only honest per-policy metric here is agreement-with-the-
# real-live-pick, never a fabricated "would-have-succeeded" outcome. Read-only,
# Admin-gated same as everything else in this file (Admin covers Senior
# Manager too, via auth.py's _REAL_ROLE_MAP: Manager2 -> Admin). ---

def _fetch_routing_comparison(range_key: str) -> dict:
    if range_key not in _RANGES:
        raise HTTPException(422, f"range must be one of {sorted(_RANGES)}")
    since = (datetime.now(timezone.utc) - _RANGES[range_key]).isoformat()
    try:
        resp = httpx.get(f"{settings().ROUTER_SERVICE_URL}/admin/routing-outcomes",
                         params={"since": since, "limit": 5000},
                         timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        rows = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ak-router unreachable: {e}") from e

    actual = [{"timestamp": r.get("timestamp"), "tier": r.get("tier"), "chosen": r.get("chosen"),
              "ok": r.get("ok"), "latencyMs": r.get("latencyMs"), "costUsd": r.get("costUsd")}
             for r in rows]
    thompson = [{"timestamp": r.get("timestamp"), "tier": r.get("tier"),
                "shadowPick": r.get("shadowPick"), "actualPick": r.get("chosen"),
                "agree": r.get("shadowPick") == r.get("chosen")}
               for r in rows if r.get("shadowPick") is not None]
    ucb = [{"timestamp": r.get("timestamp"), "tier": r.get("tier"),
           "ucbPick": r.get("ucbPick"), "actualPick": r.get("chosen"),
           "agree": r.get("ucbPick") == r.get("chosen")}
          for r in rows if r.get("ucbPick") is not None]

    def _agreement_pct(table: list[dict]) -> float | None:
        return round(100 * sum(1 for t in table if t["agree"]) / len(table), 1) if table else None

    summary = {
        "range": range_key, "total": len(actual),
        "thompsonCount": len(thompson), "ucbCount": len(ucb),
        "thompsonAgreementPct": _agreement_pct(thompson),
        "ucbAgreementPct": _agreement_pct(ucb),
    }
    return {"summary": summary, "actual": actual, "thompson": thompson, "ucb": ucb}


@router.get("/routing-comparison")
def get_routing_comparison(range: str = "7d", _: dict = Depends(require_role("Admin"))) -> dict:
    return _fetch_routing_comparison(range)


@router.get("/routing-comparison/report.pdf")
def get_routing_comparison_report(range: str = "7d", _: dict = Depends(require_role("Admin"))) -> Response:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    data = _fetch_routing_comparison(range)
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="Routing Comparison Report")
    story = [Paragraph(f"Routing Comparison Report — {range}", styles["Title"]),
            Paragraph(f"Generated {datetime.now(timezone.utc).isoformat()}", styles["Normal"]),
            Spacer(1, 12)]

    s = data["summary"]
    story.append(Paragraph(
        f"Total requests: {s['total']} &nbsp;|&nbsp; "
        f"Thompson agreement: {s['thompsonAgreementPct']}% ({s['thompsonCount']} shadow picks) &nbsp;|&nbsp; "
        f"UCB agreement: {s['ucbAgreementPct']}% ({s['ucbCount']} shadow picks)", styles["Normal"]))
    story.append(Spacer(1, 12))

    def _table(title: str, rows: list[dict], columns: list[str]) -> None:
        story.append(Paragraph(title, styles["Heading2"]))
        if not rows:
            story.append(Paragraph("No data in this range.", styles["Normal"]))
            story.append(Spacer(1, 12))
            return
        table_data = [columns] + [[str(r.get(c, "")) for c in columns] for r in rows[:200]]
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    _table("Actual", data["actual"], ["timestamp", "tier", "chosen", "ok", "latencyMs", "costUsd"])
    _table("Thompson Sampling (shadow)", data["thompson"],
          ["timestamp", "tier", "shadowPick", "actualPick", "agree"])
    _table("UCB (shadow)", data["ucb"], ["timestamp", "tier", "ucbPick", "actualPick", "agree"])

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="routing-comparison-{range}.pdf"'})
