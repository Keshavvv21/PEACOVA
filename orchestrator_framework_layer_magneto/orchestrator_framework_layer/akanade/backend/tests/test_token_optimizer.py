"""llm/token_optimizer.py — the RL-selected tool-schema reduction layer
for local (Ollama) calls specifically, plus its guardrail hierarchy
(DESTRUCTIVE/MCP tools never trimmed) and the real wiring into
llm/router.py's TierClient.chat(). Mirrors test_llm_optimization.py's
conventions for the sibling optimization_router.
"""
from __future__ import annotations

from app.llm.providers import OpenAICompatible
from app.llm.router import TierClient
from app.llm.token_optimizer import (Selection, TokenOptimizer,
                                     optimize_tools_for_provider, token_optimizer)


def _tool(name: str, description: str = "") -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []}}


# ── optimize_tools_for_provider() — the actual hook TierClient.chat calls ──

def test_non_ollama_provider_gets_the_exact_same_tool_list():
    tools = [_tool("list_projects"), _tool("delete_task")]
    result, selection = optimize_tools_for_provider("groq-llama-3.3-70b", [], tools)
    assert result == tools
    assert selection is None


def test_no_tools_passed_is_a_no_op_even_for_ollama():
    result, selection = optimize_tools_for_provider("ollama-qwen2.5:1.5b", [], None)
    assert result is None
    assert selection is None


def test_ollama_provider_can_get_a_reduced_tool_list(monkeypatch):
    # Force the tightest reduction level so this test doesn't depend on
    # the bandit's exploration randomness picking "full" by chance.
    monkeypatch.setattr("app.llm.token_optimizer._ARM_IDS", ["top3"])
    monkeypatch.setattr("app.llm.token_optimizer._KEEP_N_BY_ARM", {"top3": 3})
    many_tools = [_tool(f"read_only_tool_{i}", "reads some data") for i in range(20)]
    result, selection = optimize_tools_for_provider("ollama-qwen2.5:1.5b", [], many_tools)
    assert len(result) == 3
    assert selection is not None
    assert selection.arm_id == "top3"
    assert selection.tokens_after < selection.tokens_before


# ── The guardrail hierarchy — the non-negotiable floor ──────────────────

def test_destructive_tools_are_never_trimmed_even_at_the_tightest_level(monkeypatch):
    monkeypatch.setattr("app.llm.token_optimizer._ARM_IDS", ["top3"])
    monkeypatch.setattr("app.llm.token_optimizer._KEEP_N_BY_ARM", {"top3": 3})
    monkeypatch.setattr("app.agent.tools.DESTRUCTIVE", frozenset({"delete_task"}))
    monkeypatch.setattr("app.agent.mcp_client.mcp_tool_names", lambda: frozenset())

    tools = [_tool(f"unrelated_read_tool_{i}") for i in range(20)] + [_tool("delete_task")]
    result, selection = optimize_tools_for_provider("ollama-qwen2.5:1.5b", [], tools)
    names = {t["name"] for t in result}
    assert "delete_task" in names             # survived the top-3 cut despite never ranking by relevance
    assert len(names) == 4                    # the 3 kept optional tools + the 1 required one


def test_mcp_tools_are_never_trimmed_either(monkeypatch):
    monkeypatch.setattr("app.llm.token_optimizer._ARM_IDS", ["top3"])
    monkeypatch.setattr("app.llm.token_optimizer._KEEP_N_BY_ARM", {"top3": 3})
    monkeypatch.setattr("app.agent.tools.DESTRUCTIVE", frozenset())
    monkeypatch.setattr("app.agent.mcp_client.mcp_tool_names", lambda: frozenset({"fetch"}))

    tools = [_tool(f"unrelated_read_tool_{i}") for i in range(20)] + [_tool("fetch")]
    result, _ = optimize_tools_for_provider("ollama-qwen2.5:1.5b", [], tools)
    assert "fetch" in {t["name"] for t in result}


# ── Relevance ranking — the message actually influences what survives ───

def test_relevance_ranking_favors_tools_matching_the_users_message(monkeypatch):
    monkeypatch.setattr("app.llm.token_optimizer._ARM_IDS", ["top3"])
    monkeypatch.setattr("app.llm.token_optimizer._KEEP_N_BY_ARM", {"top3": 3})
    monkeypatch.setattr("app.agent.tools.DESTRUCTIVE", frozenset())
    monkeypatch.setattr("app.agent.mcp_client.mcp_tool_names", lambda: frozenset())

    tools = [_tool("list_sprints", "shows sprint status and progress"),
            _tool("list_projects", "shows all projects"),
            _tool("archive_note", "archives a personal note"),
            _tool("list_notes", "shows personal notes")]
    messages = [{"role": "user", "content": "show me my sprint status please"}]
    result, _ = optimize_tools_for_provider("ollama-qwen2.5:1.5b", messages, tools)
    assert "list_sprints" in {t["name"] for t in result}


# ── TokenOptimizer — the bandit + reward + observability core ───────────

def test_token_optimizer_learns_to_prefer_the_arm_with_the_higher_reward():
    opt = TokenOptimizer(exploration_scale=0.2)
    context = [0.5, 0.5]
    for _ in range(40):
        opt._core.update("full", context, -0.4)   # e.g. kept failing (out of scope)
        opt._core.update("top3", context, 0.9)     # e.g. kept succeeding, saved tokens
    picks = [opt._core.select(["full", "top3"], context)[0] for _ in range(100)]
    assert picks.count("top3") / len(picks) > 0.75


def test_record_outcome_rewards_success_and_tokens_saved():
    opt = TokenOptimizer()
    good = Selection(arm_id="top3", context=[0.0, 0.0], tokens_before=100, tokens_after=20)
    bad = Selection(arm_id="full", context=[0.0, 0.0], tokens_before=100, tokens_after=100)
    opt.record_outcome(good, success=True)
    opt.record_outcome(bad, success=False)
    # Indirect check: the successful, token-saving arm's posterior mean
    # should score higher than the failed, no-savings one for this context.
    mean_good = opt._core.posterior_mean("top3") @ opt._core.features([0.0, 0.0])
    mean_bad = opt._core.posterior_mean("full") @ opt._core.features([0.0, 0.0])
    assert mean_good > mean_bad


def test_average_tokens_stats_tracks_before_after_across_calls():
    opt = TokenOptimizer()
    assert opt.average_tokens_stats()["n_calls"] == 0
    opt.select([], {"a": {"description": "x" * 400}})  # forces a non-trivial before/after
    stats = opt.average_tokens_stats()
    assert stats["n_calls"] == 1
    assert stats["avg_tokens_before"] > 0


# ── Real wiring into TierClient.chat() ───────────────────────────────────

def test_tier_client_reduces_tools_for_an_ollama_named_provider_only(monkeypatch):
    """Two providers in the same chain, one named like the real Ollama
    rung ("ollama-<model>", the actual convention from llm/router.py's
    _providers()) and one not — only the Ollama one's outgoing tools
    list should ever be touched."""
    seen_tools = {}

    def make_fake(name):
        class Fake:
            def __init__(self):
                self.name = name

            def available(self):
                return True

            def chat(self, messages, tools=None):
                seen_tools[name] = tools
                from app.llm.providers import LLMResult
                return LLMResult(text="ok", provider=name)
        return Fake()

    fake_ollama = make_fake("ollama-qwen2.5:1.5b")
    monkeypatch.setattr("app.llm.token_optimizer._ARM_IDS", ["top3"])
    monkeypatch.setattr("app.llm.token_optimizer._KEEP_N_BY_ARM", {"top3": 3})
    monkeypatch.setattr("app.agent.tools.DESTRUCTIVE", frozenset())
    monkeypatch.setattr("app.agent.mcp_client.mcp_tool_names", lambda: frozenset())
    monkeypatch.setattr("app.llm.router._effective_chains", lambda: {"realtime": ["ollama-fake"]})
    monkeypatch.setattr("app.llm.router._providers", lambda: {"ollama-fake": fake_ollama})

    many_tools = [_tool(f"tool_{i}") for i in range(20)]
    client = TierClient(tier="realtime")
    result = client.chat("test_node", [{"role": "user", "content": "hi"}], many_tools)

    assert result.text == "ok"
    assert len(seen_tools["ollama-qwen2.5:1.5b"]) == 3     # reduced
    assert len(client.tool_selections) == 1                # stashed, not yet finalized
    assert client.tool_selections[0].arm_id == "top3"


def test_tier_client_leaves_non_ollama_providers_tools_untouched(monkeypatch):
    seen_tools = {}

    class FakeGroq:
        name = "groq-llama-3.3-70b"

        def available(self):
            return True

        def chat(self, messages, tools=None):
            seen_tools["groq"] = tools
            from app.llm.providers import LLMResult
            return LLMResult(text="ok", provider=self.name)

    fake = FakeGroq()
    monkeypatch.setattr("app.llm.router._effective_chains", lambda: {"realtime": ["groq"]})
    monkeypatch.setattr("app.llm.router._providers", lambda: {"groq": fake})

    many_tools = [_tool(f"tool_{i}") for i in range(20)]
    client = TierClient(tier="realtime")
    client.chat("test_node", [{"role": "user", "content": "hi"}], many_tools)

    assert seen_tools["groq"] == many_tools     # completely unchanged
    assert client.tool_selections == []          # nothing to finalize for a non-local provider
