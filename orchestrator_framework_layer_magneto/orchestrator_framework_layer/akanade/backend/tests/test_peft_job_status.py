"""app/framework/peft_job_status.py + GET/POST /api/admin/peft-jobs —
the "background job execution with a live status bar" surface.
"""
from __future__ import annotations

from app.framework import peft_job_status
from tests.conftest import auth


def test_start_and_finish_a_successful_job():
    job_id = peft_job_status.start_job(domain="jobtest-trading", mode="cpu", slot=0)
    assert peft_job_status.is_domain_running("jobtest-trading") is True

    peft_job_status.finish_job(job_id, ok=True, summary="Trained fine.")
    assert peft_job_status.is_domain_running("jobtest-trading") is False

    jobs = {j["job_id"]: j for j in peft_job_status.list_jobs()}
    assert jobs[job_id]["status"] == "done"
    assert jobs[job_id]["summary"] == "Trained fine."
    assert jobs[job_id]["mode"] == "cpu"


def test_finish_a_failed_job():
    job_id = peft_job_status.start_job(domain="jobtest-coding", mode="gpu", slot=1)
    peft_job_status.finish_job(job_id, ok=False, summary="OOM.")
    jobs = {j["job_id"]: j for j in peft_job_status.list_jobs()}
    assert jobs[job_id]["status"] == "failed"
    assert jobs[job_id]["mode"] == "gpu"


def test_finish_job_is_a_noop_for_an_unknown_job_id():
    peft_job_status.finish_job("never-started", ok=True, summary="x")  # must not raise


def test_empty_domain_defaults_to_general():
    job_id = peft_job_status.start_job(domain="", mode="cpu", slot=0)
    jobs = {j["job_id"]: j for j in peft_job_status.list_jobs()}
    assert jobs[job_id]["domain"] == "general"
    peft_job_status.finish_job(job_id, ok=True, summary="done")


def test_list_jobs_endpoint_requires_admin(client, member_token):
    r = client.get("/api/admin/peft-jobs", headers=auth(member_token))
    assert r.status_code == 403


def test_list_jobs_endpoint_reflects_real_state(client, exec_token):
    job_id = peft_job_status.start_job(domain="jobtest-endpoint", mode="cpu", slot=0)
    r = client.get("/api/admin/peft-jobs", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    running_ids = {j["job_id"] for j in body["running"]}
    assert job_id in running_ids
    peft_job_status.finish_job(job_id, ok=True, summary="done")

    r2 = client.get("/api/admin/peft-jobs", headers=auth(exec_token))
    running_ids_after = {j["job_id"] for j in r2.json()["running"]}
    assert job_id not in running_ids_after


def test_start_job_endpoint_and_finish_job_endpoint(client, exec_token):
    r = client.post("/api/admin/peft-jobs/start",
                    json={"domain": "jobtest-via-api", "mode": "cpu", "slot": 0},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert peft_job_status.is_domain_running("jobtest-via-api") is True

    r2 = client.post(f"/api/admin/peft-jobs/{job_id}/finish",
                     json={"ok": True, "summary": "done via API"},
                     headers=auth(exec_token))
    assert r2.status_code == 200, r2.text
    assert peft_job_status.is_domain_running("jobtest-via-api") is False
