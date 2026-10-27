"""Tests for /api/admin/ground-truth-runs (app.routers.admin_ground_truth)
— the daily record comparing a freshly fine-tuned candidate's answers
against Groq-generated reference answers for that night's own new
training prompts."""

from tests.conftest import auth

ITEMS = [
    {"prompt": "What is a Kafka consumer group?", "groq_answer": "A group of consumers sharing partitions.",
     "model_answer": "A group of consumers sharing partitions.", "match_score": 0.97},
    {"prompt": "What is a race condition?", "groq_answer": "Unsynchronized concurrent access to shared state.",
     "model_answer": "It's when a program crashes.", "match_score": 0.12},
]


def test_member_cannot_manage_ground_truth_runs(client, member_token):
    r = client.get("/api/admin/ground-truth-runs", headers=auth(member_token))
    assert r.status_code == 403

    r = client.post("/api/admin/ground-truth-runs",
                    json={"domain": "legal", "provider": "algo-akanade-slm-legal", "items": ITEMS},
                    headers=auth(member_token))
    assert r.status_code == 403


def test_record_and_list_a_run(client, exec_token):
    r = client.post("/api/admin/ground-truth-runs",
                    json={"domain": "legal", "provider": "algo-akanade-slm-legal", "items": ITEMS},
                    headers=auth(exec_token))
    assert r.status_code == 201
    body = r.json()
    assert body["domain"] == "legal"
    assert body["provider"] == "algo-akanade-slm-legal"
    assert len(body["items"]) == 2
    assert body["avg_match_score"] == round((0.97 + 0.12) / 2, 4)
    run_id = body["id"]

    # List is summary-only — no `items` in that response shape at all.
    r = client.get("/api/admin/ground-truth-runs", headers=auth(exec_token))
    assert r.status_code == 200
    summary = next(s for s in r.json() if s["id"] == run_id)
    assert "items" not in summary
    assert summary["n_items"] == 2
    assert summary["avg_match_score"] == round((0.97 + 0.12) / 2, 4)

    # Full detail, including items, via the id.
    r = client.get(f"/api/admin/ground-truth-runs/{run_id}", headers=auth(exec_token))
    assert r.status_code == 200
    assert len(r.json()["items"]) == 2


def test_a_run_with_no_items_has_zero_avg_score(client, exec_token):
    r = client.post("/api/admin/ground-truth-runs",
                    json={"domain": None, "provider": "algo-akanade-slm", "items": []},
                    headers=auth(exec_token))
    assert r.status_code == 201
    assert r.json()["avg_match_score"] == 0.0


def test_get_a_nonexistent_run_is_404(client, exec_token):
    r = client.get("/api/admin/ground-truth-runs/does-not-exist", headers=auth(exec_token))
    assert r.status_code == 404
