"""SupplyChainAgent — checks a dependency manifest's PINNED versions
against OSV.dev (Open Source Vulnerabilities, osv.dev — Google-run,
public, free, no API key), the same real vulnerability database
`pip-audit`/`osv-scanner` use. Real findings from a real database, not
an LLM guessing which packages might be risky — the LLM only phrases a
summary of what OSV.dev actually returned (see security_findings.py).

`task.metadata["path"]` points at a real requirements.txt-shaped file
under settings().RAG_INGEST_DIR — the same "a REST caller supplies a
relative path, not a trusted server operator" guard
document_pipeline.py's ingest_path() already enforces for RAG (reused
via document_pipeline.resolve_within(), not reimplemented); with no
path, `task.text` is parsed directly as pasted manifest content. An
unpinned dependency (no exact version) is reported as SKIPPED, not
silently ignored — OSV.dev needs an exact version to check, and
guessing one would be worse than saying so honestly.
"""
from __future__ import annotations

import re

import httpx

from ...config import settings
from ...llm.router import get_llm_client
from ..agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from ..document_pipeline import UnsafePathError, resolve_within
from ..task import Task, TaskType
from .security_findings import Finding, ScanResult, format_findings_for_prompt, log_scan

_OSV_URL = "https://api.osv.dev/v1/query"
_TIMEOUT = 10.0

# requirements.txt-shaped: `name==1.2.3`, `name >= 1.2.3`, `name~=1.2.3` —
# only an EXACT pin (`==`) is checkable against a specific known version;
# any other operator or a bare name is reported as skipped, not guessed at.
_PINNED_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)\s*(?:#.*)?$")
_UNPINNED_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~]+)?")


def parse_requirements(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns (pinned [(name, version), ...], skipped [reason, ...])."""
    pinned: list[tuple[str, str]] = []
    skipped: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(("-r", "-e", "--")):
            continue
        m = _PINNED_LINE.match(stripped)
        if m:
            pinned.append((m.group(1), m.group(2)))
            continue
        m2 = _UNPINNED_LINE.match(stripped)
        name = m2.group(1) if m2 else stripped
        skipped.append(f"{name}: not exactly pinned (need `==<version>` to check against OSV.dev)")
    return pinned, skipped


def query_osv(name: str, version: str, *, ecosystem: str = "PyPI") -> list[dict]:
    """One real OSV.dev lookup — a network call, so this is the one
    function tests monkeypatch rather than hitting the real API. Returns
    the raw `vulns` list from OSV's response, [] if none or on any
    request failure (never raises — a supply-chain check that crashes
    the whole scan over one flaky network call is worse than reporting
    that package as skipped)."""
    try:
        r = httpx.post(_OSV_URL, json={"package": {"name": name, "ecosystem": ecosystem},
                                       "version": version}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("vulns", [])
    except (httpx.HTTPError, ValueError):
        return []


def scan_dependencies(text: str) -> ScanResult:
    result = ScanResult()
    pinned, skipped = parse_requirements(text)
    result.skipped = skipped
    for name, version in pinned:
        result.scanned.append(f"{name}=={version}")
        vulns = query_osv(name, version)
        for v in vulns:
            severity = "high"  # OSV's own severity field is inconsistent across advisories;
            # treat every real match as at least "high" rather than guess-parsing CVSS text.
            result.findings.append(Finding(
                rule_id=v.get("id", "OSV-UNKNOWN"), severity=severity,
                message=(v.get("summary") or v.get("details") or "Known vulnerability (see OSV.dev).")[:300],
                location=f"{name}=={version}",
                snippet=v.get("id", "")))
    return result


SYSTEM_PROMPT = {"role": "system", "content": (
    "You summarize dependency vulnerability findings from OSV.dev (a real "
    "vulnerability database) for a developer. You are given a fixed list of "
    "findings — never invent, add, or omit any finding; never claim a "
    "dependency is safe unless it was actually checked AND came back clean. "
    "Group by package, note which packages were skipped and why (unpinned "
    "versions can't be checked), and recommend upgrading to a fixed version "
    "where OSV.dev's data mentions one.")}


class SupplyChainAgent(Agent):
    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="supply-chain-agent",
            name="Supply Chain Agent",
            description="Checks a requirements.txt-shaped dependency manifest's "
                       "EXACTLY PINNED versions against OSV.dev (a real, public "
                       "vulnerability database) — not an LLM guessing which "
                       "packages might be risky. Unpinned dependencies are "
                       "reported as skipped, never silently ignored.",
            capabilities=frozenset({"security_scan", "supply_chain_scan", "dependency_scan"}),
            supported_tasks=frozenset({TaskType.CODE_REVIEW, TaskType.GENERIC}),
            model="configurable (via llm/router.py's provider chain) — summary only",
            tools=("osv_dev_lookup",),
            rag_enabled=False,
            cost_per_call_usd=0.003,
            latency_profile=LatencyProfile(p50_ms=800, p95_ms=5000),  # dominated by real OSV.dev round trips
            version="1.0.0",
            reliability_score=0.7,
            status=AgentStatus.ACTIVE,
        )

    def understand(self, task: Task) -> dict:
        return {"path": task.metadata.get("path"), "text": task.text}

    def plan(self, task: Task, understanding: dict) -> dict:
        return understanding

    def execute(self, task: Task, plan: dict) -> dict:
        if plan.get("path"):
            try:
                safe_path = resolve_within(plan["path"], settings().RAG_INGEST_DIR)
                text = safe_path.read_text(encoding="utf-8", errors="ignore")
            except (UnsafePathError, OSError) as e:
                return {"scan": ScanResult(skipped=[f"{plan['path']}: {e}"])}
        else:
            text = plan["text"]
        return {"scan": scan_dependencies(text)}

    def validate(self, task: Task, result: dict) -> dict:
        return result  # OSV.dev lookups are already real — nothing to retry

    def respond(self, task: Task, validated: dict) -> AgentResult:
        scan: ScanResult = validated["scan"]
        log_scan(self.metadata.agent_id, scan)
        findings_text = format_findings_for_prompt(scan.findings)
        client = get_llm_client("bulk")
        messages = [SYSTEM_PROMPT,
                   {"role": "user", "content": f"Checked: {', '.join(scan.scanned) or 'nothing'}\n"
                                              f"Skipped: {', '.join(scan.skipped) or 'nothing'}\n\n"
                                              f"Findings:\n{findings_text}"}]
        res = client.chat("supply_chain_summary", messages)
        text = res.text if not res.error else findings_text
        return AgentResult(
            text=text, ok=not res.error, confidence=0.9 if scan.scanned else 0.4,
            raw={"findings": [f.__dict__ for f in scan.findings],
                "scanned": scan.scanned, "skipped": scan.skipped, "provider": res.provider})

    def evaluate(self, task: Task, result: AgentResult) -> float:
        if not result.ok:
            return 0.0
        return result.confidence if result.raw.get("scanned") else result.confidence * 0.5
