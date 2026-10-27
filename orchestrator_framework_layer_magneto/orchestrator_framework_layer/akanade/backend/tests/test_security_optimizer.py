"""app/security_optimizer.py — the RL layer that learns from REAL
confirmation outcomes on top of security_guard.py's deterministic
rules. Never touches RISK (that's covered by security_precheck's hard
block, tested in test_security_integration.py) — only whether a 🟡
NEEDS_REVIEW rule earns a confidence note over repeated real feedback.
"""
from __future__ import annotations

from app.agent import native_agent
from app.security_optimizer import SecurityOptimizer, security_optimizer


def test_confidence_note_is_none_for_a_never_seen_rule_source_pair():
    opt = SecurityOptimizer()
    assert opt.confidence_note("SEC-DOES-NOT-EXIST", "user_prompt") is None


def test_repeated_approvals_learn_toward_a_fine_note():
    opt = SecurityOptimizer()
    finding = "SEC-UNAUTHORIZED-ACTION: some message"
    for _ in range(8):
        opt.record_feedback([finding], source="user_prompt", approved=True)
    note = opt.confidence_note("SEC-UNAUTHORIZED-ACTION", "user_prompt")
    assert note is not None
    assert "fine" in note


def test_repeated_rejections_learn_toward_a_real_concern_note():
    opt = SecurityOptimizer()
    finding = "SEC-PHISHING-URGENCY: some message"
    for _ in range(8):
        opt.record_feedback([finding], source="email_content", approved=False)
    note = opt.confidence_note("SEC-PHISHING-URGENCY", "email_content")
    assert note is not None
    assert "concern" in note


def test_arms_are_isolated_per_source():
    opt = SecurityOptimizer()
    finding = "SEC-UNAUTHORIZED-ACTION: some message"
    for _ in range(8):
        opt.record_feedback([finding], source="user_prompt", approved=False)
    # The SAME rule on a DIFFERENT source has no learned history yet.
    assert opt.confidence_note("SEC-UNAUTHORIZED-ACTION", "tool_response") is None


def test_record_feedback_ignores_findings_with_no_recognizable_rule_id():
    opt = SecurityOptimizer()
    opt.record_feedback(["not a real finding string"], source="user_prompt", approved=False)
    # No crash, and nothing learned for any real rule id.
    assert opt.confidence_note("SEC-UNAUTHORIZED-ACTION", "user_prompt") is None


# ── native_agent.record_security_feedback / process-wide singleton ────

def test_record_security_feedback_is_a_noop_for_empty_findings():
    # Just needs to not raise — most confirmations (a plain destructive
    # PM action) carry no security findings at all.
    native_agent.record_security_feedback([], approved=True)


def test_record_security_feedback_reaches_the_shared_singleton():
    finding = "SEC-UNAUTHORIZED-ACTION: test singleton wiring"
    for _ in range(8):
        native_agent.record_security_feedback([finding], approved=False)
    assert security_optimizer.confidence_note("SEC-UNAUTHORIZED-ACTION", "user_prompt") is not None


def test_security_confidence_note_helper_returns_empty_string_when_nothing_learned():
    # A brand-new (name, args) pair with no NEEDS_REVIEW findings at all
    # yields no note — the confirmation prompt gets no parenthetical.
    assert native_agent._security_confidence_note("send_email",
                                                   {"to": "x@y.com", "subject": "hi",
                                                    "body": "ordinary text"}) == ""
