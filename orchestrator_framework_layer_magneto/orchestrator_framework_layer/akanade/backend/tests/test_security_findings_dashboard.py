"""app/framework/agents/security_findings.py's log_scan()/
list_recent_scans() (persisting every scan run, not just the one-off
AgentResult) and GET /api/admin/security/findings.
"""
from __future__ import annotations

from app.framework.agents.security_findings import Finding, ScanResult, list_recent_scans, log_scan
from tests.conftest import auth


def test_log_scan_persists_a_clean_scan_as_highest_severity_none():
    scan = ScanResult(scanned=["clean_file.py"])
    log_scan("test-code-security-agent", scan)
    scans = list_recent_scans(days=1)
    row = next(s for s in scans if s["agent_id"] == "test-code-security-agent"
              and s["scanned"] == ["clean_file.py"])
    assert row["highest_severity"] == "none"
    assert row["findings"] == []


def test_log_scan_persists_findings_sorted_by_severity():
    scan = ScanResult(
        scanned=["risky_file.py"],
        findings=[
            Finding(rule_id="LOW-1", severity="low", message="minor"),
            Finding(rule_id="CRIT-1", severity="critical", message="serious"),
        ],
    )
    log_scan("test-code-security-agent-2", scan)
    scans = list_recent_scans(days=1)
    row = next(s for s in scans if s["agent_id"] == "test-code-security-agent-2")
    assert row["highest_severity"] == "critical"
    assert row["findings"][0]["rule_id"] == "CRIT-1"  # sorted, most severe first


def test_findings_endpoint_summarizes_by_severity(client, exec_token):
    scan = ScanResult(scanned=["dashboard_test.py"],
                      findings=[Finding(rule_id="DASH-TEST", severity="high", message="test finding")])
    log_scan("test-dashboard-agent", scan)

    r = client.get("/api/admin/security/findings", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["scans"], list)
    assert body["summary"]["total_scans"] >= 1
    assert body["summary"]["by_severity"].get("high", 0) >= 1


def test_findings_endpoint_requires_admin(client, member_token):
    r = client.get("/api/admin/security/findings", headers=auth(member_token))
    assert r.status_code == 403
