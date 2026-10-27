"""End-to-end: the new fields ChatReply/AgentReply gained this session
(provider, token_stats, sources_used) actually round-trip through the
real /api/chat/message endpoint — not just unit-tested in isolation.
Also the guardrail invariant: none of the RL optimizers (token, RAG)
change whether a destructive action still requires confirmation.
"""
import pytest

from tests.conftest import auth

TRACKS = ["native", "langgraph", "adk"]


def say(client, tok, msg, track, session="rag-surf"):
    r = client.post(f"/api/chat/message?framework={track}",
                    json={"message": msg, "session_id": f"{session}-{track}"},
                    headers=auth(tok))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("track", TRACKS)
def test_provider_is_surfaced_in_the_chat_response(client, exec_token, track):
    d = say(client, exec_token, "what is overdue in Compass OS", track)
    assert d["provider"]                          # non-empty for any real turn
    assert d["provider"] == d["tier_log"][-1]["provider"]  # matches the actual call that answered


@pytest.mark.parametrize("track", TRACKS)
def test_rag_sources_surfaced_for_a_question_matching_an_indexed_document(client, exec_token, track,
                                                                          monkeypatch):
    # rag_registry always has "pm-index" (Chroma-backed, real PM data)
    # registered alongside "default" — the RL store-selector's Thompson
    # Sampling exploration can legitimately pick "pm-index" for this query
    # by chance (it has no idea yet which store is relevant here), which
    # would correctly find nothing there and make this test flaky through
    # no fault of the production code. Scoping list_names() to just the
    # one store this test actually seeded removes that non-determinism
    # without touching how the real endpoint behaves.
    monkeypatch.setattr("app.framework.rag.rag_registry.list_names", lambda: ["default"])

    index = client.post("/api/v1/rag/index", json={
        "store": "default",
        "text": "The ALGO AKANADE release process requires sign-off from the platform "
               "lead before any nightly fine-tune gets promoted to production.",
    }, headers=auth(exec_token))
    assert index.status_code == 200

    d = say(client, exec_token, "what does the release process require?", track)
    assert d["sources_used"], f"expected sources_used to be populated, got reply: {d['reply']!r}"


@pytest.mark.parametrize("track", TRACKS)
def test_a_plain_command_does_not_trigger_rag_retrieval(client, exec_token, track):
    d = say(client, exec_token, f"create a task called 'No RAG needed {track}' in Compass OS", track)
    assert d["sources_used"] == []


# ── guardrail invariant: optimizers never affect confirmation gating ────

@pytest.mark.parametrize("track", TRACKS)
def test_destructive_action_still_requires_confirmation_regardless_of_optimizers(
        client, exec_token, track):
    """token_optimizer.py and rag_optimizer.py both only ever change WHAT
    gets sent to the model (a trimmed tool schema, extra retrieved
    context) — neither should ever affect whether requires_confirmation()
    still gates a destructive action. This is the same assertion
    test_destructive_requires_confirmation_and_no_cancels makes; kept
    here too as an explicit regression guard tied to THESE optimizers
    specifically, so a future change to either can't quietly weaken it."""
    say(client, exec_token,
        f"create a task called 'Guardrail check {track}' in Compass OS", track, "guardrail")
    d = say(client, exec_token, f"delete task Guardrail check {track}", track, "guardrail")
    assert d["pending_confirmation"] is True
