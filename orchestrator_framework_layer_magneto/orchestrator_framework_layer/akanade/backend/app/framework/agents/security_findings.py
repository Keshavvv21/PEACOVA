"""Shared shape for the three security agents (code_security_agent.py,
supply_chain_agent.py, cloud_config_agent.py) — every one of them does
REAL deterministic detection first (regex/pattern checks over real
file content, or a real query against a real vulnerability database),
and only THEN asks an LLM to summarize/prioritize what was actually
found. The LLM never invents a finding — it's grounded in this list,
the same "grounded ONLY in the provided excerpts, never invent facts"
rule document_pipeline.py already applies to summaries.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ...db import store as db_store

Severity = str  # "critical" | "high" | "medium" | "low" | "info"

_TABLE = "SecurityScans"

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    message: str
    location: str = ""       # file:line, or a package name, or a resource identifier
    snippet: str = ""        # the actual matched text, for a human to verify — never fabricated


def sort_by_severity(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: _SEVERITY_RANK.get(f.severity, 0), reverse=True)


def format_findings_for_prompt(findings: list[Finding]) -> str:
    """The ONLY thing the LLM ever sees about what was found — a plain
    listing of the real Finding objects, so its summary can't drift from
    what was actually detected."""
    if not findings:
        return "No findings."
    lines = []
    for f in sort_by_severity(findings):
        loc = f" ({f.location})" if f.location else ""
        snippet = f" — `{f.snippet}`" if f.snippet else ""
        lines.append(f"[{f.severity.upper()}] {f.rule_id}{loc}: {f.message}{snippet}")
    return "\n".join(lines)


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    scanned: list[str] = field(default_factory=list)   # what was actually checked
    skipped: list[str] = field(default_factory=list)    # what couldn't be checked, and why — never silent


def log_scan(agent_id: str, scan: ScanResult) -> None:
    """Persists every scan run (code_security_agent.py, supply_chain_
    agent.py, cloud_config_agent.py) for the Security Dashboard's
    "identified threats/vulnerabilities" view — before this, a scan's
    findings only ever existed in that one AgentResult.raw, gone the
    moment the caller moved on. A scan with zero findings is still
    logged (as "clean") — the dashboard needs to show what WAS checked
    and came back safe, not just what tripped a rule. Same fail-soft
    convention as security_events.log_event()/bandit_persistence.py: a
    storage hiccup must never break the scan reply it's describing."""
    now = datetime.now(timezone.utc)
    row = {
        "id": uuid.uuid4().hex[:12],
        "agent_id": agent_id,
        "occurred_at": now.isoformat(),
        "scanned": scan.scanned,
        "skipped": scan.skipped,
        "findings": [f.__dict__ for f in sort_by_severity(scan.findings)],
        "highest_severity": (sort_by_severity(scan.findings)[0].severity
                            if scan.findings else "none"),
    }
    try:
        db_store(_TABLE).put(now.strftime("%Y-%m-%d"), f"SCAN#{now.isoformat()}#{row['id']}", row)
    except Exception:  # noqa: BLE001
        pass


def _strip_storage_keys(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("PK", "SK")}


def list_recent_scans(days: int = 7) -> list[dict]:
    """Newest first — the Security Dashboard's real data source for
    "threats identified" across all three scanning agents."""
    from datetime import timedelta

    rows: list[dict] = []
    today = datetime.now(timezone.utc).date()
    for i in range(days):
        date_key = (today - timedelta(days=i)).isoformat()
        rows.extend(db_store(_TABLE).scan(date_key, "SCAN#"))
    return sorted((_strip_storage_keys(r) for r in rows), key=lambda r: r["occurred_at"], reverse=True)
