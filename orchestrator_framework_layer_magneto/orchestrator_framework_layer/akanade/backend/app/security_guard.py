"""Cross-cutting security classification — 🟢 SAFE / 🟡 NEEDS_REVIEW /
🔴 RISK — applied to every real text surface a chat turn touches: the
user's own prompt, content read FROM an external system (Gmail/Calendar/
Chat — see google_tools.py), and the assistant's own generated reply
going back OUT. Same "deterministic detection first, never an LLM
guessing" discipline as app/framework/agents/code_security_agent.py's
rule-based scanning — a security flag needs to be reproducible for
automated pass/fail testing, which a second LLM call classifying its own
sibling's output would not reliably be.

This is DELIBERATELY narrower than a real production threat-detection
system: pattern/heuristic matching over plain text, not a trained
classifier, an LLM judge, or real malware/URL reputation lookups. Treat
a RISK/NEEDS_REVIEW flag as "worth a human's attention," not a
guarantee of what's actually inside — same honesty the code-security
agents already apply to their own regex-vs-AST limitations. See
README.md's "Security Prompts" section for the documented, tested
behavior this module is expected to produce.

Distinct from llm/guardrails.py's input_guard(): that one is a hard
pre-execution refusal gate for a fixed prompt-injection pattern list
(refuses the whole turn on a match). This module additionally FLAGS
(without necessarily blocking) content from every other surface, and is
designed to run repeatedly across a turn — it's the "how risky was
everything I just touched" signal, not a single admission gate. Its
patterns deliberately overlap input_guard's for user-prompt injection
attempts, so a turn input_guard refuses is still correctly reported as
🔴 RISK to the caller (see native_agent.assess_security).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SAFE = "safe"
NEEDS_REVIEW = "needs_review"
RISK = "risk"

_ORDER = {SAFE: 0, NEEDS_REVIEW: 1, RISK: 2}

FLAG_EMOJI = {SAFE: "🟢", NEEDS_REVIEW: "🟡", RISK: "🔴"}
FLAG_LABEL = {SAFE: "Safe", NEEDS_REVIEW: "Needs Review", RISK: "Security Risk"}


@dataclass
class SecurityAssessment:
    flag: str = SAFE
    findings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    flag: str
    pattern: re.Pattern
    message: str
    # Which `source` value(s) this rule applies to — None = every source.
    # A phrase that's merely suspicious coming from the USER (e.g. asking
    # "ignore previous instructions" — could be a curious test) is
    # materially more dangerous found INSIDE untrusted external content
    # (an email/tool response actively trying to command the agent) — see
    # the two separate embedded-injection rules below for that distinction.
    sources: frozenset[str] | None = None


_UNIVERSAL_SOURCES = None  # applies everywhere

_RULES: list[_Rule] = [
    # ── prompt injection ──────────────────────────────────────────────
    _Rule("SEC-INJECTION-USER", NEEDS_REVIEW,
         re.compile(r"(?i)\b(ignore|disregard)\s+(all\s+)?(previous|prior|above|earlier)\s+"
                    r"instructions?\b|\byou\s+are\s+now\s+(in\s+)?(unrestricted|dan|jailbroken|"
                    r"developer\s+mode)\b|\breveal\s+(your\s+)?(system\s+prompt|api[_ -]?key|"
                    r"secret|password|jwt)\b"),
         "Prompt-injection-style phrasing in the user's own message — same pattern class "
         "llm/guardrails.py's input_guard() refuses on.",
         frozenset({"user_prompt"})),
    _Rule("SEC-INJECTION-EMBEDDED", RISK,
         re.compile(r"(?i)\b(ignore|disregard)\s+(all\s+)?(previous|prior|above|earlier)\s+"
                    r"instructions?\b|\bsystem\s*:\s|\bnew\s+instructions?\s*:\s|\byou\s+must\s+"
                    r"now\b|\byour\s+(real|true|actual)\s+(task|goal|instructions?)\s+is\b"),
         "Content read FROM an external source (email/calendar/chat/tool response) is trying "
         "to issue new instructions to the assistant — a classic indirect prompt-injection "
         "attempt via untrusted data, more severe than the same phrasing from the user "
         "themselves since the user never actually said this.",
         frozenset({"email_content", "calendar_event", "chat_message", "tool_response"})),

    # ── credential / secret exfiltration requests ────────────────────
    _Rule("SEC-CRED-EXFIL-REQUEST", RISK,
         re.compile(r"(?i)\b(?:(?:show|print|reveal|give|send)\s+(?:me\s+)?(?:your\s+|the\s+)?|"
                    r"what(?:\s+is|'s)\s+your\s+)(?:oauth\s+)?(?:access\s+)?(?:token|"
                    r"api[_ -]?key|client\s+secret|password|credentials?)\b"),
         "A request to expose an OAuth token, API key, or other credential — these must "
         "never be surfaced in a chat reply regardless of who's asking."),
    _Rule("SEC-CRED-IN-CONTENT", RISK,
         re.compile(r"(?i)\b(access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
                    r"api[_ -]?key)\s*[:=]\s*\S{8,}"),
         "What looks like a literal credential/token value appearing in content — should "
         "never be echoed back verbatim."),

    # ── data exfiltration ─────────────────────────────────────────────
    _Rule("SEC-EXFIL-FORWARD", RISK,
         re.compile(r"(?i)\bforward\s+(all\s+)?(my\s+)?(emails?|inbox|messages?)\s+to\b|"
                    r"\bsend\s+(all\s+)?(my\s+)?(emails?|contacts?|calendar)\s+to\s+"
                    r"[\w.+-]+@[\w-]+\.[\w.-]+"),
         "A request to bulk-forward the user's own mailbox/contacts/calendar to a third "
         "party — a classic data-exfiltration pattern, distinct from sending one normal "
         "email the user is clearly composing on purpose."),
    _Rule("SEC-PII-OUTBOUND", NEEDS_REVIEW,
         re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]*?){13,16}\b"),
         "Content that looks like a Social Security Number or card-number-shaped sequence "
         "of digits — worth a human's review before it goes out anywhere."),

    # ── phishing / social engineering ─────────────────────────────────
    _Rule("SEC-PHISHING-URGENCY", NEEDS_REVIEW,
         re.compile(r"(?is)\b(urgent(ly)?|immediately|act\s+now|verify\s+your\s+account|"
                    r"suspended|confirm\s+your\s+(password|identity|payment))\b.{0,80}"
                    r"\b(click|log\s*in|sign\s*in|password|verify)\b"),
         "Urgency language combined with a request to click/log in/verify — the classic "
         "phishing shape."),
    _Rule("SEC-SOCIAL-ENGINEERING", NEEDS_REVIEW,
         re.compile(r"(?is)\bthis\s+is\s+(your\s+)?(ceo|boss|manager|it\s+support|admin)\b.{0,80}"
                    r"\b(wire|transfer|gift\s*cards?|password|urgent)\b"),
         "Impersonation-plus-pressure phrasing (\"this is your CEO\" + a money/credential "
         "ask) — a classic social-engineering / business-email-compromise shape."),

    # ── suspicious URLs ────────────────────────────────────────────────
    _Rule("SEC-URL-IP-LITERAL", NEEDS_REVIEW,
         re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}(?:[:/]\S*)?"),
         "A raw IP-address URL — legitimate services almost never link via bare IP."),
    _Rule("SEC-URL-SHORTENER", NEEDS_REVIEW,
         re.compile(r"(?i)https?://(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|ow\.ly|"
                    r"buff\.ly)/\S+"),
         "A link-shortener URL — hides the real destination, common in phishing."),
    _Rule("SEC-URL-PUNYCODE", NEEDS_REVIEW,
         re.compile(r"https?://[\w.-]*xn--[\w.-]+"),
         "A punycode (xn--) domain — often used for homograph/look-alike domain spoofing."),

    # ── unauthorized-action language ──────────────────────────────────
    _Rule("SEC-UNAUTHORIZED-ACTION", NEEDS_REVIEW,
         re.compile(r"(?i)\bwithout\s+(asking|confirmation|telling|notifying)\b|"
                    r"\bdo(n'?t|\s+not)\s+(ask|confirm|tell\s+(the\s+)?user)\b|"
                    r"\bskip\s+(the\s+)?confirmation\b|\bauto[- ]?approve\b"),
         "Explicit language asking the assistant to bypass its own confirmation step — "
         "the confirmation gate for a write action is never skippable by request."),
]


def classify(text: str, *, source: str = "user_prompt") -> SecurityAssessment:
    """Runs every rule scoped to `source` (or scoped to every source)
    against `text`, returning the WORST flag found plus every matching
    rule's message. Never raises — worst case (no text, empty string) is
    just SAFE with no findings, same "never blocks on its own absence"
    rule every optional layer in this codebase follows."""
    if not text:
        return SecurityAssessment()
    findings: list[str] = []
    worst = SAFE
    for rule in _RULES:
        if rule.sources is not None and source not in rule.sources:
            continue
        if rule.pattern.search(text):
            findings.append(f"{rule.rule_id}: {rule.message}")
            if _ORDER[rule.flag] > _ORDER[worst]:
                worst = rule.flag
    return SecurityAssessment(flag=worst, findings=findings)


def aggregate(assessments: list[SecurityAssessment]) -> SecurityAssessment:
    """Worst-of across every surface checked this turn, findings from
    every assessment concatenated (order-preserving, de-duplicated)."""
    worst = SAFE
    findings: list[str] = []
    seen: set[str] = set()
    for a in assessments:
        if _ORDER[a.flag] > _ORDER[worst]:
            worst = a.flag
        for f in a.findings:
            if f not in seen:
                seen.add(f)
                findings.append(f)
    return SecurityAssessment(flag=worst, findings=findings)
