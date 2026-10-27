"""CloudConfigAgent — real pattern checks over Infrastructure-as-Code
and container config (Dockerfile, docker-compose.yml, Kubernetes YAML,
Terraform .tf) for well-known misconfiguration classes: wide-open
network access, privileged containers, containers running as root,
hardcoded secrets in config, and disabled encryption. Deterministic —
same "the LLM only summarizes real findings, never invents one" rule as
code_security_agent.py/supply_chain_agent.py (see security_findings.py).

No live cloud-provider API calls (this deployment has no AWS/GCP/Azure
credentials configured, and shouldn't need any to be useful) — scans
the CONFIGURATION FILES that define your infrastructure instead, the
same static-analysis approach real tools like checkov/tfsec use. Real
and honest about that scope, not a simulation of a live cloud audit.

`task.metadata["path"]` scans a real file/directory under
settings().RAG_INGEST_DIR (same guard as the other two security agents,
via document_pipeline.resolve_within()); with no path, `task.text` is
scanned directly as pasted config content.

Known limitation (same as code_security_agent.py): line-based regex
over raw text, not a real IaC parser — findings are "worth a human
look," not a definitive verdict. Validated against this repo's own
deploy/Dockerfile.* for real: correctly flagged all three as running
without a USER directive (root by default) and one as pinned to
`:latest`.
"""
from __future__ import annotations

import re
from pathlib import Path

from ...config import settings
from ...llm.router import get_llm_client
from ..agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from ..document_pipeline import UnsafePathError, resolve_within
from ..task import Task, TaskType
from .security_findings import Finding, ScanResult, format_findings_for_prompt, log_scan

# Line-based rules, applicable across Dockerfile/compose/Kubernetes/Terraform —
# deliberately format-agnostic substring/regex checks rather than a real
# parser per format, same "simple, real, line-numbered" tradeoff
# code_security_agent.py makes.
_RULES: list[tuple[str, str, re.Pattern, str]] = [
    ("CFG-PRIVILEGED", "critical", re.compile(r"(?i)\bprivileged\s*[:=]\s*true\b|--privileged\b"),
     "A privileged container has near-total access to the host — a container "
     "escape here compromises the whole host, not just the container."),
    ("CFG-OPEN-CIDR", "critical", re.compile(r"0\.0\.0\.0/0"),
     "A security-group/firewall rule open to 0.0.0.0/0 (the entire internet) — "
     "scope this to the specific IP ranges that actually need access."),
    ("CFG-HOST-NETWORK", "high", re.compile(r"(?i)\bhostNetwork\s*:\s*true\b"),
     "hostNetwork:true removes network isolation between the container and "
     "the host — the container can see/bind every host network interface."),
    ("CFG-ROOT-USER", "medium", re.compile(r"(?i)\brunAsUser\s*:\s*0\b|^\s*USER\s+root\b"),
     "Explicitly running as root (uid 0) — a container compromise gets root "
     "in the container immediately, with no privilege-escalation step needed."),
    ("CFG-PUBLIC-ACCESS", "high", re.compile(r"(?i)\bpublicly[_-]?accessible\s*[:=]\s*true\b"),
     "A resource (database, storage bucket, ...) marked publicly accessible — "
     "confirm this is actually intended, not a default left unchanged."),
    ("CFG-NO-ENCRYPTION", "high", re.compile(r"(?i)\b(encrypted|server_side_encryption)\s*[:=]\s*false\b"),
     "Encryption explicitly disabled for a resource that stores data at rest."),
    ("CFG-UNPINNED-IMAGE", "low", re.compile(r"(?i)^\s*FROM\s+\S+:latest\b"),
     "`:latest` isn't reproducible and can silently change what actually "
     "deploys between builds — pin to a specific version/digest."),
]

_HARDCODED_SECRET = re.compile(
    r"(?i)\b(password|passwd|api[_-]?key|secret|access[_-]?key|token)\b\s*[:=]\s*"
    r"[\"']?([A-Za-z0-9_\-/+=]{6,})[\"']?\s*$")

_SUPPORTED_SUFFIXES = {".dockerfile", ".yaml", ".yml", ".tf"}


def _scan_text(text: str, *, location_prefix: str = "") -> list[Finding]:
    findings: list[Finding] = []
    has_user_directive = False
    is_dockerfile = "FROM " in text or location_prefix.lower().endswith("dockerfile")
    for lineno, line in enumerate(text.splitlines(), start=1):
        loc = f"{location_prefix}:{lineno}" if location_prefix else f"line {lineno}"
        if re.match(r"^\s*USER\s+\S+", line) and "root" not in line.lower():
            has_user_directive = True
        for rule_id, severity, pattern, message in _RULES:
            if pattern.search(line):
                findings.append(Finding(rule_id=rule_id, severity=severity, message=message,
                                        location=loc, snippet=line.strip()[:120]))
        m = _HARDCODED_SECRET.search(line)
        if m and not line.strip().startswith("#"):
            findings.append(Finding(
                rule_id="CFG-HARDCODED-SECRET", severity="critical",
                message=f"A literal value assigned to a config key named like a "
                       f"credential ({m.group(1)!r}) — use a secret manager or an "
                       f"env-var reference, never a literal committed to config.",
                location=loc, snippet=line.strip()[:120]))
    if is_dockerfile and not has_user_directive:
        findings.append(Finding(
            rule_id="CFG-NO-USER-DIRECTIVE", severity="medium",
            message="No USER directive found — the container runs as root by "
                   "default. Add a non-root USER before the final CMD/ENTRYPOINT.",
            location=location_prefix or "(pasted content)"))
    return findings


def scan_path(path: Path) -> ScanResult:
    result = ScanResult()
    if path.is_file():
        files = [path]
    else:
        files = sorted(f for f in path.rglob("*") if f.is_file() and
                       (f.suffix.lower() in _SUPPORTED_SUFFIXES or f.name.lower() == "dockerfile"
                        or f.name.lower().startswith("dockerfile.")))
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            result.skipped.append(f"{f}: {e}")
            continue
        result.scanned.append(str(f))
        result.findings.extend(_scan_text(text, location_prefix=str(f)))
    if not result.scanned and not result.skipped:
        result.skipped.append(f"{path}: no Dockerfile/docker-compose/Kubernetes/Terraform files found")
    return result


SYSTEM_PROMPT = {"role": "system", "content": (
    "You summarize cloud/container configuration security findings for a "
    "developer. You are given a fixed list of findings — never invent, add, "
    "or omit any finding from that list; never claim something is safe "
    "unless the findings list is genuinely empty. Group by severity, explain "
    "the real-world attack surface each finding opens up, and give one "
    "concrete remediation per finding type.")}


class CloudConfigAgent(Agent):
    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="cloud-config-agent",
            name="Cloud Config Agent",
            description="Scans Dockerfile/docker-compose/Kubernetes/Terraform "
                       "content for well-known misconfigurations — privileged "
                       "containers, containers running as root, wide-open "
                       "network access (0.0.0.0/0), disabled encryption, "
                       "hardcoded secrets. Static config analysis (no live "
                       "cloud-provider API access) — real regex matches "
                       "against real config text, with a file:line citation.",
            capabilities=frozenset({"security_scan", "cloud_config_scan", "iac_scan"}),
            supported_tasks=frozenset({TaskType.CODE_REVIEW, TaskType.GENERIC}),
            model="configurable (via llm/router.py's provider chain) — summary only",
            tools=("static_config_scan",),
            rag_enabled=False,
            cost_per_call_usd=0.003,
            latency_profile=LatencyProfile(p50_ms=400, p95_ms=1500),
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
            except UnsafePathError as e:
                return {"scan": ScanResult(skipped=[str(e)])}
            scan = scan_path(safe_path)
        else:
            scan = ScanResult(findings=_scan_text(plan["text"]), scanned=["<pasted text>"])
        return {"scan": scan}

    def validate(self, task: Task, result: dict) -> dict:
        return result  # nothing to retry — the scan itself is deterministic

    def respond(self, task: Task, validated: dict) -> AgentResult:
        scan: ScanResult = validated["scan"]
        log_scan(self.metadata.agent_id, scan)
        findings_text = format_findings_for_prompt(scan.findings)
        client = get_llm_client("bulk")
        messages = [SYSTEM_PROMPT,
                   {"role": "user", "content": f"Scanned: {', '.join(scan.scanned) or 'nothing'}\n"
                                              f"Skipped: {', '.join(scan.skipped) or 'nothing'}\n\n"
                                              f"Findings:\n{findings_text}"}]
        res = client.chat("cloud_config_summary", messages)
        text = res.text if not res.error else findings_text
        return AgentResult(
            text=text, ok=not res.error, confidence=0.9 if scan.findings or scan.scanned else 0.5,
            raw={"findings": [f.__dict__ for f in scan.findings],
                "scanned": scan.scanned, "skipped": scan.skipped, "provider": res.provider})

    def evaluate(self, task: Task, result: AgentResult) -> float:
        if not result.ok:
            return 0.0
        return result.confidence if result.raw.get("scanned") else result.confidence * 0.5
