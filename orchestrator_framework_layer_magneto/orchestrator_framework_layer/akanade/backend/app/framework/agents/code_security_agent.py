"""CodeSecurityAgent — real static pattern checks for well-known Python
vulnerability classes (hardcoded secrets, dangerous eval/exec, shell
injection, insecure deserialization, unsafe YAML loading, weak hashing),
line-numbered against real source text. Deterministic and dependency-
free by design — no LLM call decides WHETHER something is a finding;
the LLM only phrases a grounded summary of findings that already exist
(see security_findings.py's module docstring), so this can never
hallucinate a vulnerability that isn't actually in the code.

`task.metadata["path"]` scans a real file/directory under
settings().RAG_INGEST_DIR — the exact same "a REST caller supplies a
relative path, not a trusted server operator" guard
document_pipeline.py's ingest_path() already enforces for RAG (reused,
not reimplemented, via document_pipeline.resolve_within()); a path
outside that directory is reported as skipped, never read. With no
path, `task.text` is scanned directly as pasted code — useful for a
snippet a user pastes into chat without it ever touching disk.

Known limitation, confirmed by scanning this very file with itself:
these are line-based regex checks over RAW TEXT, not a real AST
analysis (unlike CodingAgent's actual ast.parse) — a rule's own
docstring/message text that merely MENTIONS a pattern (e.g. this file's
own comments discussing "eval()" or "shell=True" as prose) can trigger
a false positive the same way a real string literal or comment
containing that text would. Real tools (bandit, semgrep) avoid this
with a real parser; this stays a lightweight, dependency-free
line-scanner on purpose, so findings should be read as "worth a human
look," not as a definitive verdict.
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

# Each rule: (id, severity, compiled pattern, human message). Patterns are
# intentionally simple/line-based — a real cross-file dataflow analysis
# is a much bigger tool; this catches the well-known, common, single-line
# patterns real secret scanners and linters (bandit, gitleaks) key on too.
_RULES: list[tuple[str, str, re.Pattern, str]] = [
    ("PY-EVAL-EXEC", "high", re.compile(r"\b(eval|exec)\s*\("),
     "eval()/exec() on anything not fully trusted is arbitrary code execution."),
    ("PY-SHELL-TRUE", "high", re.compile(r"shell\s*=\s*True"),
     "subprocess/os call with shell=True — shell-metacharacter injection risk "
     "if any part of the command comes from user input."),
    ("PY-OS-SYSTEM", "high", re.compile(r"\bos\.system\s*\("),
     "os.system() runs through a shell — same injection risk as shell=True; "
     "prefer subprocess.run([...], shell=False)."),
    ("PY-PICKLE-LOAD", "medium", re.compile(r"\bpickle\.(loads?|Unpickler)\b"),
     "pickle deserializes arbitrary objects — loading untrusted pickle data "
     "can execute arbitrary code."),
    ("PY-YAML-UNSAFE-LOAD", "high", re.compile(r"\byaml\.load\s*\((?!.*Loader\s*=\s*yaml\.SafeLoader)"),
     "yaml.load() without Loader=yaml.SafeLoader can construct arbitrary "
     "Python objects from the YAML — use yaml.safe_load() instead."),
    ("PY-WEAK-HASH", "medium", re.compile(r"\bhashlib\.(md5|sha1)\s*\("),
     "MD5/SHA1 are cryptographically broken — fine for a checksum, never for "
     "passwords or anything security-sensitive (use bcrypt/scrypt/argon2)."),
]

_HARDCODED_SECRET = re.compile(
    r"(?i)\b(password|passwd|api[_-]?key|secret|access[_-]?key|token)\b\s*[:=]\s*"
    r"[\"']([^\"']{4,})[\"']")


def _scan_text(text: str, *, location_prefix: str = "") -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        loc = f"{location_prefix}:{lineno}" if location_prefix else f"line {lineno}"
        for rule_id, severity, pattern, message in _RULES:
            if pattern.search(line):
                findings.append(Finding(rule_id=rule_id, severity=severity, message=message,
                                        location=loc, snippet=line.strip()[:120]))
        m = _HARDCODED_SECRET.search(line)
        if m:
            findings.append(Finding(
                rule_id="PY-HARDCODED-SECRET", severity="critical",
                message=f"A literal string assigned to a variable named like a "
                       f"credential ({m.group(1)!r}) — real secrets belong in an "
                       f"env var or secret store, never committed as a literal.",
                location=loc, snippet=line.strip()[:120]))
    return findings


_SUPPORTED_SUFFIXES = {".py"}


def scan_path(path: Path) -> ScanResult:
    result = ScanResult()
    files = [path] if path.is_file() else sorted(f for f in path.rglob("*") if f.is_file())
    for f in files:
        if f.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            result.skipped.append(f"{f}: {e}")
            continue
        result.scanned.append(str(f))
        result.findings.extend(_scan_text(text, location_prefix=str(f)))
    if not result.scanned and not result.skipped:
        result.skipped.append(f"{path}: no .py files found")
    return result


SYSTEM_PROMPT = {"role": "system", "content": (
    "You summarize static code-security scan findings for a developer. "
    "You are given a fixed list of findings — never invent, add, or omit "
    "any finding from that list; never claim something is safe unless "
    "the findings list is genuinely empty. Group by severity, explain "
    "impact in plain language, and give one concrete remediation per "
    "finding type.")}


class CodeSecurityAgent(Agent):
    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="code-security-agent",
            name="Code Security Agent",
            description="Scans Python code for well-known vulnerability classes — "
                       "hardcoded secrets, eval/exec, shell injection, insecure "
                       "deserialization, unsafe YAML loading, weak hashing. "
                       "Deterministic pattern checks, not an LLM guess: every "
                       "reported finding is a real regex match against real "
                       "source text, with a file:line citation.",
            capabilities=frozenset({"security_scan", "code_review", "vulnerability_scan"}),
            supported_tasks=frozenset({TaskType.CODE_REVIEW, TaskType.GENERIC}),
            model="configurable (via llm/router.py's provider chain) — summary only",
            tools=("static_pattern_scan",),
            rag_enabled=False,
            cost_per_call_usd=0.003,
            latency_profile=LatencyProfile(p50_ms=400, p95_ms=1500),
            version="1.0.0",
            reliability_score=0.75,
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
        res = client.chat("code_security_summary", messages)
        text = res.text if not res.error else findings_text  # raw findings, never lost on an LLM failure
        return AgentResult(
            text=text, ok=not res.error, confidence=0.9 if scan.findings or scan.scanned else 0.5,
            raw={"findings": [f.__dict__ for f in scan.findings],
                "scanned": scan.scanned, "skipped": scan.skipped, "provider": res.provider})

    def evaluate(self, task: Task, result: AgentResult) -> float:
        # A scan that actually checked something is a stronger signal
        # than one that silently skipped everything.
        if not result.ok:
            return 0.0
        return result.confidence if result.raw.get("scanned") else result.confidence * 0.5
