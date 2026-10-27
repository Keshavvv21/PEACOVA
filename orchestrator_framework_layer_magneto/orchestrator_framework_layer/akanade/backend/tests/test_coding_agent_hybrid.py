"""Coverage for "Auto Hybrid" in the Coding Agent (coding_agent.py's
_resolve_hybrid/_HYBRID_LOCAL_PROVIDER) — reuses llm/hybrid_router.py's
existing local-vs-hosted RL bandit, previously only ever wired into the
PM/Trading chat pipeline, instead of building a parallel mechanism.
"""
from __future__ import annotations

from tests.conftest import auth


def test_hybrid_preferred_provider_resolves_to_ollama_coder_and_records_outcome(client, exec_token, monkeypatch):
    from app.framework.agents import coding_agent
    from app.llm.hybrid_router import HybridSelection
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_preferred = {}
    recorded = {}

    def fake_choose(self, message, tier, local_provider="ollama"):
        assert local_provider == "ollama-coder"
        return HybridSelection(arm="local", context=[0.0, 0.0, 0.0])

    def fake_preferred_provider_for(self, selection, local_provider="ollama"):
        return local_provider if selection.arm == "local" else None

    def fake_record_outcome(self, selection, *, success, latency_ms=0.0):
        recorded["selection"] = selection
        recorded["success"] = success

    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "choose", fake_choose)
    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "preferred_provider_for", fake_preferred_provider_for)
    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "record_outcome", fake_record_outcome)

    def fake_init(self, tier, preferred=None, preferred_timeout=None, exclude_rule_based=False,
                 explicit_cascade=None):
        seen_preferred["value"] = preferred
        self.tier = tier
        self.preferred = preferred
        self.log = []
        self.tool_selections = []

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="plain answer, no tool call", provider="ollama-coder")

    monkeypatch.setattr(TierClient, "__init__", fake_init)
    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "explain this function",
        "session_id": "hybrid-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "preferred_provider": "hybrid",
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert seen_preferred["value"] == "ollama-coder"
    assert recorded["success"] is True


def test_plain_default_turn_now_also_uses_hybrid_router_not_hardcoded_ollama_coder(
    client, exec_token, monkeypatch
):
    """Real gap found live: a plain default turn (no preferred_provider
    at all — the common case for every normal Coding Agent request)
    used to hard-prefer "ollama-coder" unconditionally, completely
    bypassing hybrid_router's complexity-aware local-vs-hosted choice
    (previously reachable only via the explicit "hybrid" opt-in string
    covered by the test above). Confirmed live: a trivial read_file
    tool-call decision on that hard-coded CPU model took 100+ seconds.
    This confirms the SAME hybrid_router.choose() call now also fires
    for a caller who never set preferred_provider at all."""
    from app.framework.agents import coding_agent
    from app.llm.hybrid_router import HybridSelection
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_preferred = {}

    def fake_choose(self, message, tier, local_provider="ollama"):
        assert local_provider == "ollama-coder"
        return HybridSelection(arm="hosted", context=[0.0, 0.0, 0.0])

    def fake_preferred_provider_for(self, selection, local_provider="ollama"):
        return local_provider if selection.arm == "local" else None

    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "choose", fake_choose)
    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "preferred_provider_for",
                        fake_preferred_provider_for)
    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "record_outcome",
                        lambda self, selection, *, success, latency_ms=0.0: None)

    def fake_init(self, tier, preferred=None, preferred_timeout=None, exclude_rule_based=False,
                 explicit_cascade=None):
        seen_preferred["value"] = preferred
        self.tier = tier
        self.preferred = preferred
        self.log = []
        self.tool_selections = []

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="a fast hosted answer, no tool call", provider="groq")

    monkeypatch.setattr(TierClient, "__init__", fake_init)
    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "read fibonacci.py",
        "session_id": "plain-default-hybrid-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    # hybrid_router picked "hosted" here — preferred stays None (the
    # normal chain decides), never hard-forced to "ollama-coder".
    assert seen_preferred["value"] is None


def test_non_hybrid_pick_never_touches_hybrid_router(client, exec_token, monkeypatch):
    from app.framework.agents import coding_agent
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    called = {"choose": False, "record": False}

    def fake_choose(self, message, tier, local_provider="ollama"):
        called["choose"] = True
        raise AssertionError("hybrid_router.choose should never run for an explicit non-hybrid pick")

    def fake_record_outcome(self, selection, *, success, latency_ms=0.0):
        called["record"] = True

    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "choose", fake_choose)
    monkeypatch.setattr(coding_agent.hybrid_router.__class__, "record_outcome", fake_record_outcome)

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="plain answer", provider="groq")

    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "explain this function",
        "session_id": "non-hybrid-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "preferred_provider": "groq",
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert called == {"choose": False, "record": False}


def test_coding_agent_always_uses_the_secondary_pipelines_own_cascade(client, exec_token, monkeypatch):
    """A plain default turn (no preferred_provider, no "hybrid" opt-in)
    must always resolve to the Secondary Pipeline's own cascade
    (llm/coding_pipeline.effective_cascade()) — never hybrid_router's
    local/hosted bandit, which only ever applies when a caller explicitly
    opts in with "hybrid" (see test_hybrid_router.py equivalents above).
    WHICH tiers that cascade contains IS admin-toggleable (unlike the
    routing mechanism itself) — see coding_pipeline.set_escalation_enabled
    and its own admin endpoint (routers/admin_router_config.py) — so this
    checks both states explicitly rather than asserting one fixed list."""
    from app.llm import coding_pipeline
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen = {}

    def fake_init(self, tier, preferred=None, preferred_timeout=None, exclude_rule_based=False,
                 explicit_cascade=None):
        seen["preferred"] = preferred
        seen["explicit_cascade"] = explicit_cascade
        self.tier = tier
        self.preferred = preferred
        self.explicit_cascade = explicit_cascade
        self.log = []
        self.tool_selections = []

    def fake_chat(self, node, messages, tools=None):
        assert self.explicit_cascade == coding_pipeline.effective_cascade()
        return LLMResult(text="answer", provider="glm-5.3-flash")

    monkeypatch.setattr(TierClient, "__init__", fake_init)
    monkeypatch.setattr(TierClient, "chat", fake_chat)

    coding_pipeline.set_escalation_enabled(False, "test")
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "explain this function (secondary pipeline cascade test, escalation off)",
        "session_id": "secondary-pipeline-cascade-off",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert seen["preferred"] is None
    assert seen["explicit_cascade"] == [coding_pipeline.CODING_CASCADE_ORDER[0]]

    coding_pipeline.set_escalation_enabled(True, "test")
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "explain this function (secondary pipeline cascade test, escalation on)",
        "session_id": "secondary-pipeline-cascade-on",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert seen["explicit_cascade"] == list(coding_pipeline.CODING_CASCADE_ORDER)
    coding_pipeline.set_escalation_enabled(False, "test")  # restore the default for any test after this one
