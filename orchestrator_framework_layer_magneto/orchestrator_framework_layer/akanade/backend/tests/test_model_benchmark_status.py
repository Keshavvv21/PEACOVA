"""app/framework/model_benchmark_status.py + GET /api/admin/model-
benchmark-jobs and /api/admin/model-benchmarks — same "background job
execution with a live status bar" pattern as peft_job_status.py, applied
to scripts/eval/harness.py runs instead of nightly fine-tune jobs.

Deliberately does NOT call POST /api/admin/model-benchmarks/run in any
test here — that endpoint's background task shells out to the real
eval harness (its own separate venv, see admin_model_benchmarks.py's
own docstring), which this test environment doesn't have set up; RBAC
on that endpoint is still verified (a 403 for Member never reaches the
subprocess at all).
"""
from __future__ import annotations

from app.framework import model_benchmark_status
from tests.conftest import auth


def test_start_and_finish_a_successful_job():
    job_id = model_benchmark_status.start_job(model_source="ours", triggered_by="test")
    assert any(j["job_id"] == job_id for j in model_benchmark_status.running_jobs())

    model_benchmark_status.finish_job(job_id, ok=True, summary="Sweep completed.")
    jobs = {j["job_id"]: j for j in model_benchmark_status.list_jobs()}
    assert jobs[job_id]["status"] == "done"
    assert jobs[job_id]["summary"] == "Sweep completed."
    assert jobs[job_id]["model_source"] == "ours"
    assert not any(j["job_id"] == job_id for j in model_benchmark_status.running_jobs())


def test_finish_a_failed_job():
    job_id = model_benchmark_status.start_job(model_source="candidate", triggered_by="test")
    model_benchmark_status.finish_job(job_id, ok=False, summary="harness exited non-zero.")
    jobs = {j["job_id"]: j for j in model_benchmark_status.list_jobs()}
    assert jobs[job_id]["status"] == "failed"
    assert jobs[job_id]["model_source"] == "candidate"


def test_finish_job_is_a_noop_for_an_unknown_job_id():
    model_benchmark_status.finish_job("never-started", ok=True, summary="x")  # must not raise


def test_jobs_endpoint_requires_admin(client, member_token):
    r = client.get("/api/admin/model-benchmarks/jobs", headers=auth(member_token))
    assert r.status_code == 403


def test_jobs_endpoint_reflects_real_state(client, exec_token):
    job_id = model_benchmark_status.start_job(model_source="ours", triggered_by="test-endpoint")
    r = client.get("/api/admin/model-benchmarks/jobs", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert job_id in {j["job_id"] for j in r.json()["running"]}
    model_benchmark_status.finish_job(job_id, ok=True, summary="done")

    r2 = client.get("/api/admin/model-benchmarks/jobs", headers=auth(exec_token))
    assert job_id not in {j["job_id"] for j in r2.json()["running"]}


def test_run_endpoint_requires_executive(client, admin_token):
    # Admin can VIEW (jobs/list above) but not TRIGGER a real sweep —
    # same Executive-write/Admin-read split as every other admin action.
    r = client.post("/api/admin/model-benchmarks/run", json={"model_source": "ours"},
                    headers=auth(admin_token))
    assert r.status_code == 403


def test_run_endpoint_rejects_an_unknown_model_source(client, exec_token):
    r = client.post("/api/admin/model-benchmarks/run", json={"model_source": "not-a-real-source"},
                    headers=auth(exec_token))
    assert r.status_code == 422


def test_list_benchmarks_endpoint_requires_admin(client, member_token):
    r = client.get("/api/admin/model-benchmarks", headers=auth(member_token))
    assert r.status_code == 403


def test_list_benchmarks_endpoint_reads_real_rows(client, exec_token):
    from app.db import store
    store("ModelBenchmarks").put("BENCHMARK#test123", "META", {
        "timestamp": "2026-01-01T00:00:00+00:00", "model": "test-model",
        "modelSource": "ours", "taskFamily": "reasoning", "qualityScore": 0.8,
    })
    r = client.get("/api/admin/model-benchmarks", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    models = {b["model"] for b in r.json()["benchmarks"]}
    assert "test-model" in models

    r2 = client.get("/api/admin/model-benchmarks?model_source=candidate", headers=auth(exec_token))
    assert "test-model" not in {b["model"] for b in r2.json()["benchmarks"]}
