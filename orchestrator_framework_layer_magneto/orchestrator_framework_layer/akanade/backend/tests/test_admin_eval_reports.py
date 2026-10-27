"""Tests for /api/admin/slm-eval-reports (app.routers.admin_eval_reports)
— the daily verification report scripts/eval/regression_gate.py
produces for every candidate it scores."""

from tests.conftest import auth

ITEMS = [
    {"item_id": "task-status-1", "category": "tool_use", "scoring_type": "deterministic",
     "score": 1.0, "answer": "Done"},
    {"item_id": "reasoning-1", "category": "reasoning", "scoring_type": "judged",
     "score": 0.4, "error": "timeout", "answer": ""},
]


def test_member_cannot_manage_eval_reports(client, member_token):
    r = client.get("/api/admin/slm-eval-reports", headers=auth(member_token))
    assert r.status_code == 403

    r = client.post("/api/admin/slm-eval-reports",
                    json={"tag": "algo-akanade-slm:20260828", "base_tag": "qwen2.5:1.5b",
                          "tolerance": 0.02, "old_score": 0.8, "new_score": 0.85, "passed": True},
                    headers=auth(member_token))
    assert r.status_code == 403


def test_record_and_list_a_report(client, exec_token):
    r = client.post("/api/admin/slm-eval-reports",
                    json={"tag": "algo-akanade-slm:20260828", "base_tag": "qwen2.5:1.5b",
                          "tolerance": 0.02, "old_score": 0.80, "new_score": 0.85, "passed": True,
                          "old_by_category": {"tool_use": 0.9, "reasoning": 0.7},
                          "new_by_category": {"tool_use": 0.95, "reasoning": 0.75},
                          "items": ITEMS},
                    headers=auth(exec_token))
    assert r.status_code == 201
    body = r.json()
    assert body["tag"] == "algo-akanade-slm:20260828"
    assert body["passed"] is True
    assert len(body["items"]) == 2
    report_id = body["id"]

    # List is summary-only — no `items` in that response shape at all.
    r = client.get("/api/admin/slm-eval-reports", headers=auth(exec_token))
    assert r.status_code == 200
    summary = next(s for s in r.json() if s["id"] == report_id)
    assert "items" not in summary
    assert summary["n_items"] == 2
    assert summary["n_item_errors"] == 1

    # Full detail, including items, via the id.
    r = client.get(f"/api/admin/slm-eval-reports/{report_id}", headers=auth(exec_token))
    assert r.status_code == 200
    assert len(r.json()["items"]) == 2


def test_a_failed_gate_report_is_still_recorded(client, exec_token):
    r = client.post("/api/admin/slm-eval-reports",
                    json={"tag": "algo-akanade-slm:20260827", "base_tag": "qwen2.5:1.5b",
                          "tolerance": 0.02, "old_score": 0.80, "new_score": 0.60, "passed": False,
                          "items": ITEMS},
                    headers=auth(exec_token))
    assert r.status_code == 201
    assert r.json()["passed"] is False


def test_domain_field_round_trips(client, exec_token):
    r = client.post("/api/admin/slm-eval-reports",
                    json={"tag": "algo-akanade-slm-legal:20260828", "base_tag": "qwen2.5:1.5b",
                          "domain": "legal", "tolerance": 0.02, "old_score": 0.8,
                          "new_score": 0.83, "passed": True, "items": []},
                    headers=auth(exec_token))
    assert r.json()["domain"] == "legal"


def test_get_unknown_report_404s(client, exec_token):
    r = client.get("/api/admin/slm-eval-reports/does-not-exist", headers=auth(exec_token))
    assert r.status_code == 404


def test_newest_report_listed_first(client, exec_token):
    for tag in ["algo-akanade-slm:20260101", "algo-akanade-slm:20260102"]:
        client.post("/api/admin/slm-eval-reports",
                    json={"tag": tag, "base_tag": "qwen2.5:1.5b", "tolerance": 0.02,
                          "old_score": 0.8, "new_score": 0.8, "passed": True, "items": []},
                    headers=auth(exec_token))
    r = client.get("/api/admin/slm-eval-reports", headers=auth(exec_token))
    tags = [v["tag"] for v in r.json()]
    assert tags.index("algo-akanade-slm:20260102") < tags.index("algo-akanade-slm:20260101")
