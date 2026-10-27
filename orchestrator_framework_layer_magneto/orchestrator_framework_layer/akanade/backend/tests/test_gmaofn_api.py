"""ALGO AKANADE's REST API surface (/api/v1/*) — exercised through the real
bootstrapped agents (pm-agent, openclaw-agent, coding-agent; see
gmaofn/bootstrap.py, wired into app startup by app/main.py), through the
SAME TestClient/fixtures the rest of this suite uses, so a PM task
executed via POST /api/v1/orchestration/execute is exercising the exact
same real PM domain logic as POST /api/v1/chat/message — just through
ALGO AKANADE's generic entry point instead of the PM-specific one.
"""
from __future__ import annotations

from tests.conftest import auth


def test_list_agents_includes_the_three_bootstrapped_defaults(client, exec_token):
    r = client.get("/api/v1/agents", headers=auth(exec_token))
    assert r.status_code == 200
    ids = {a["agent_id"] for a in r.json()}
    assert {"pm-agent", "openclaw-agent", "coding-agent"} <= ids


def test_get_one_agent(client, exec_token):
    r = client.get("/api/v1/agents/coding-agent", headers=auth(exec_token))
    assert r.status_code == 200
    body = r.json()
    assert body["agent_id"] == "coding-agent"
    assert "code_generation" in body["capabilities"]


def test_get_unknown_agent_404s(client, exec_token):
    r = client.get("/api/v1/agents/no-such-agent", headers=auth(exec_token))
    assert r.status_code == 404


def test_register_update_and_delete_a_remote_agent(client, admin_token):
    register = client.post("/api/v1/agents/register", json={
        "agent_id": "test-remote-agent", "name": "Test Remote Agent",
        "description": "A webhook-backed agent for API lifecycle testing",
        "webhook_url": "http://localhost:9/does-not-need-to-exist-for-this-test",
        "capabilities": ["testing"],
    }, headers=auth(admin_token))
    assert register.status_code == 201, register.text
    assert register.json()["agent_id"] == "test-remote-agent"

    duplicate = client.post("/api/v1/agents/register", json={
        "agent_id": "test-remote-agent", "name": "dup", "description": "dup",
        "webhook_url": "http://localhost:9/dup"}, headers=auth(admin_token))
    assert duplicate.status_code == 409

    updated = client.put("/api/v1/agents/test-remote-agent", json={"status": "disabled"},
                         headers=auth(admin_token))
    assert updated.status_code == 200
    assert updated.json()["status"] == "disabled"

    deleted = client.delete("/api/v1/agents/test-remote-agent", headers=auth(admin_token))
    assert deleted.status_code == 204
    gone = client.get("/api/v1/agents/test-remote-agent", headers=auth(admin_token))
    assert gone.status_code == 404


def test_register_requires_admin(client, member_token):
    r = client.post("/api/v1/agents/register", json={
        "agent_id": "should-not-register", "name": "x", "description": "x",
        "webhook_url": "http://localhost:9/x"}, headers=auth(member_token))
    assert r.status_code == 403


def test_list_models_includes_generic_model_layer_providers(client, exec_token):
    r = client.get("/api/v1/models", headers=auth(exec_token))
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["providers"]}
    assert {"claude", "huggingface", "custom", "rule"} <= ids


def test_list_tools_includes_pm_and_mcp_tools(client, exec_token):
    r = client.get("/api/v1/tools", headers=auth(exec_token))
    assert r.status_code == 200
    tool_ids = {t["tool_id"] for t in r.json()["tools"]}
    assert "create_task" in tool_ids
    assert "get_current_datetime" in tool_ids  # the bundled MCP example server


def test_list_workflow_step_kinds(client, exec_token):
    r = client.get("/api/v1/workflows", headers=auth(exec_token))
    assert r.status_code == 200
    kinds = {s["kind"] for s in r.json()["step_kinds"]}
    assert {"sequential", "parallel", "consensus", "fallback", "retry", "hierarchical"} == kinds


def test_orchestration_execute_real_pm_prompt_end_to_end(client, exec_token):
    """The actual "make sure current prompts work" proof, through
    ALGO AKANADE's generic entry point rather than /api/chat/message —
    same underlying PM chatbot, same seeded Compass OS project."""
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "create a task called 'ALGO AKANADE API test' in Compass OS",
        "session_id": "gmaofn-api-test-1",
        "context": {"task_type": "project_management", "required_capabilities": ["project_management"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "pm-agent" in body["selected_agents"]
    assert "Created task" in body["result"]["text"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert "task_id" in body
    assert body["trace"]["events"]  # a non-empty orchestration trace


def test_orchestration_execute_coding_prompt_routes_to_coding_agent(client, exec_token):
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "Write a Python function that reverses a string",
        "session_id": "gmaofn-api-test-2",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert r.json()["selected_agents"] == ["coding-agent"]


def test_orchestration_then_feedback_then_task_lookup(client, exec_token):
    execute = client.post("/api/v1/orchestration/execute", json={
        "task": "create a task called 'Feedback loop test' in Compass OS",
        "session_id": "gmaofn-api-test-3",
        "context": {"task_type": "project_management", "required_capabilities": ["project_management"]},
    }, headers=auth(exec_token))
    task_id = execute.json()["task_id"]

    feedback = client.post("/api/v1/feedback", json={"task_id": task_id, "score": 0.95},
                           headers=auth(exec_token))
    assert feedback.status_code == 200, feedback.text
    assert "pm-agent" in feedback.json()["attributed_rewards"]

    lookup = client.get(f"/api/v1/tasks/{task_id}", headers=auth(exec_token))
    assert lookup.status_code == 200
    assert lookup.json()["outcome"]["user_feedback"] == 0.95

    evaluation = client.get(f"/api/v1/evaluations/{task_id}", headers=auth(exec_token))
    assert evaluation.status_code == 200


def test_feedback_for_unknown_task_404s(client, exec_token):
    r = client.post("/api/v1/feedback", json={"task_id": "no-such-task", "score": 0.5},
                    headers=auth(exec_token))
    assert r.status_code == 404


def test_rag_index_search_and_query_round_trip(client, exec_token):
    index = client.post("/api/v1/rag/index", json={
        "store": "default",
        "text": "ALGO AKANADE is a generic multi-agent orchestration framework. "
               "It routes tasks using Thompson Sampling and Shapley value credit assignment.",
    }, headers=auth(exec_token))
    assert index.status_code == 200
    assert index.json()["chunks_indexed"] >= 1

    search = client.post("/api/v1/rag/search", json={"store": "default", "query": "Thompson Sampling"},
                         headers=auth(exec_token))
    assert search.status_code == 200
    assert search.json()["hits"]

    query = client.post("/api/v1/rag/query", json={"store": "default", "query": "Shapley value"},
                        headers=auth(exec_token))
    assert query.status_code == 200
    assert "Shapley" in query.json()["context"]


def test_rag_unknown_store_404s(client, exec_token):
    r = client.post("/api/v1/rag/search", json={"store": "no-such-store", "query": "x"},
                    headers=auth(exec_token))
    assert r.status_code == 404


def test_metrics_requires_admin_and_reports_agents(client, admin_token, member_token):
    denied = client.get("/api/v1/metrics", headers=auth(member_token))
    assert denied.status_code == 403

    allowed = client.get("/api/v1/metrics", headers=auth(admin_token))
    assert allowed.status_code == 200
    body = allowed.json()
    assert "pm-agent" in body["agents"]
    assert "reward_weights" in body
    assert "network" in body
