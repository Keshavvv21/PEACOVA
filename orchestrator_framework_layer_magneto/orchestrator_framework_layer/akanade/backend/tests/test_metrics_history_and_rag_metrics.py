"""Coverage for two new observability surfaces added for the ALGO AKANADE ops
dashboards (RL "improving over time" and RAG performance):

- GET /api/v1/metrics/history (framework/metrics_history.py) — a real
  time-series log, one row per FeedbackLoop.close()/submit_user_feedback()
  call, where GET /api/v1/metrics itself is always point-in-time-only.
- GET /api/v1/rag/metrics (framework/rag_optimizer.py's stats()/
  query_stats()) — RagOptimizer's per-(source, algorithm) bandit already
  accumulated real signal from every real chat turn; this just persists
  it (bandit_persistence.py) and exposes it for the first time.
"""
from __future__ import annotations

from tests.conftest import auth


def test_metrics_history_grows_across_orchestration_calls(client, exec_token):
    before = client.get("/api/v1/metrics/history", headers=auth(exec_token))
    assert before.status_code == 200, before.text
    before_count = len(before.json()["snapshots"])

    for i in range(2):
        r = client.post("/api/v1/orchestration/execute", json={
            "task": f"metrics history probe {i}",
            "session_id": f"metrics-history-probe-{i}",
            "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        }, headers=auth(exec_token))
        assert r.status_code == 200, r.text

    after = client.get("/api/v1/metrics/history", headers=auth(exec_token))
    assert after.status_code == 200, after.text
    snapshots = after.json()["snapshots"]
    assert len(snapshots) >= before_count + 2

    latest = snapshots[-1]
    assert "timestamp" in latest and "task_id" in latest
    assert isinstance(latest["reward_weights"], dict) and latest["reward_weights"]
    assert isinstance(latest["attributed_rewards"], dict)
    # Ascending by time — the natural order for a chart's x-axis.
    timestamps = [s["timestamp"] for s in snapshots]
    assert timestamps == sorted(timestamps)


def test_metrics_history_respects_limit(client, exec_token):
    r = client.get("/api/v1/metrics/history?limit=1", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert len(r.json()["snapshots"]) <= 1


def test_metrics_history_requires_admin(client, member_token):
    r = client.get("/api/v1/metrics/history", headers=auth(member_token))
    assert r.status_code == 403


def test_rag_metrics_reflects_real_usage(client, exec_token):
    """Rather than reaching into rag_optimizer internals, this drives a
    real RAG-backed PM chat turn (the seeded demo project) — the same
    path native_agent.py/langgraph_agent.py/adk_agent.py already call
    retrieve_with_optimizer()+record_outcome() on — then checks the
    metrics endpoint reflects at least one recorded call."""
    before = client.get("/api/v1/rag/metrics", headers=auth(exec_token))
    assert before.status_code == 200, before.text
    before_total = before.json()["query_coverage"]["total_queries"]

    r = client.post("/api/chat/message", json={
        "message": "what tasks are in compass os?",
        "session_id": "rag-metrics-probe",
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text

    after = client.get("/api/v1/rag/metrics", headers=auth(exec_token))
    assert after.status_code == 200, after.text
    body = after.json()
    assert isinstance(body["arms"], list)
    assert body["query_coverage"]["total_queries"] >= before_total + 1
    if body["arms"]:
        arm = body["arms"][0]
        assert {"store", "algorithm", "calls", "empty_hits", "successes",
               "hit_rate", "success_rate", "posterior_mean_weights"} <= arm.keys()


def test_rag_metrics_requires_admin(client, member_token):
    r = client.get("/api/v1/rag/metrics", headers=auth(member_token))
    assert r.status_code == 403


def test_routing_metrics_reflects_real_usage(client, exec_token):
    """Drives a real plain-auto chat turn (no provider specified at all)
    — the path native_agent.py's auto-also-tries-domain-SLM addition and
    provider_optimizer.record_outcome() both run on every single call —
    then checks the metrics endpoint reflects it."""
    before = client.get("/api/v1/metrics/routing", headers=auth(exec_token))
    assert before.status_code == 200, before.text

    r = client.post("/api/chat/message", json={
        "message": "hello, what can you help me with?",
        "session_id": "routing-metrics-probe",
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text

    after = client.get("/api/v1/metrics/routing", headers=auth(exec_token))
    assert after.status_code == 200, after.text
    body = after.json()
    assert isinstance(body["domain_routing"], list)
    assert isinstance(body["provider_routing"], list)
    if body["provider_routing"]:
        row = body["provider_routing"][0]
        assert {"tier", "provider", "calls", "wins", "win_rate",
               "posterior_mean_weights"} <= row.keys()
    if body["domain_routing"]:
        row = body["domain_routing"][0]
        assert {"domain", "calls", "wins", "win_rate",
               "posterior_mean_weights"} <= row.keys()


def test_routing_metrics_requires_admin(client, member_token):
    r = client.get("/api/v1/metrics/routing", headers=auth(member_token))
    assert r.status_code == 403
