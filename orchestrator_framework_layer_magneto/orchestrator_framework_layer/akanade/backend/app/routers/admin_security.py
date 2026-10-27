"""Admin-only visibility into the security_guard.py flag layer — real
logged events (security_events.py) and an LLM-summarized daily digest
(security_report.py). Admin-role only (not Executive-only like
admin_providers.py's API-key management — this is read visibility into
what users are experiencing, not a credential-management surface, so
the lower bar in auth.py's ROLE_ORDER applies, matching AdminUsers.tsx's
own admin-visibility bar on the frontend).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from .. import security_events, security_report
from ..auth import require_role
from ..framework.agents.security_findings import list_recent_scans

router = APIRouter(prefix="/api/admin/security", tags=["admin-security"])


@router.get("/events")
def list_events(days: int = Query(7, ge=1, le=90), _: dict = Depends(require_role("Admin"))):
    return security_events.list_recent_events(days)


@router.get("/daily-report")
def daily_report(date: str | None = Query(None, description="YYYY-MM-DD, default today (UTC)"),
                 refresh: bool = Query(False, description="Regenerate even if cached"),
                 _: dict = Depends(require_role("Admin"))):
    return security_report.generate_daily_report(date, force=refresh)


@router.get("/findings")
def list_findings(days: int = Query(7, ge=1, le=90), _: dict = Depends(require_role("Admin"))):
    """Real threats/vulnerabilities identified by the three scanning
    agents (code-security-agent, supply-chain-agent, cloud-config-agent)
    — every scan run, including ones that came back clean
    (highest_severity: "none"), newest first. Distinct from /events
    above, which is about flagged risky USER PROMPTS, not scanned
    code/dependencies/config."""
    scans = list_recent_scans(days)
    total_findings = sum(len(s["findings"]) for s in scans)
    by_severity: dict[str, int] = {}
    for s in scans:
        for f in s["findings"]:
            by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1
    return {
        "scans": scans,
        "summary": {
            "total_scans": len(scans),
            "total_findings": total_findings,
            "by_severity": by_severity,
            "clean_scans": sum(1 for s in scans if s["highest_severity"] == "none"),
        },
    }
