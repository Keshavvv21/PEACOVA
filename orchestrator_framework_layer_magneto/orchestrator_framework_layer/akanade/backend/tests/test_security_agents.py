"""The three security agents (code_security_agent.py, supply_chain_agent.py,
cloud_config_agent.py) — real deterministic detection first, LLM summary
only over real findings. Mirrors test_document_pipeline.py's TierClient
mocking convention for the summary step.
"""
from __future__ import annotations

from app.framework.agents.cloud_config_agent import CloudConfigAgent, _scan_text as scan_cloud_text
from app.framework.agents.code_security_agent import CodeSecurityAgent, _scan_text as scan_code_text
from app.framework.agents.supply_chain_agent import (SupplyChainAgent, parse_requirements,
                                                  query_osv, scan_dependencies)
from app.framework.document_pipeline import UnsafePathError, resolve_within
from app.framework.task import Task, TaskContext
from app.llm.providers import LLMResult
from app.llm.router import TierClient


def _fake_chat_returning(text: str):
    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text=text, provider="fake-llm")
    return fake_chat


# ── code_security_agent ──────────────────────────────────────────────────

def test_scan_text_detects_eval_shell_true_and_pickle():
    code = (
        "user_input = get_request_param()\n"
        "eval(user_input)\n"
        "subprocess.run(cmd, shell=True)\n"
        "data = pickle.loads(raw)\n"
    )
    findings = scan_code_text(code)
    rule_ids = {f.rule_id for f in findings}
    assert {"PY-EVAL-EXEC", "PY-SHELL-TRUE", "PY-PICKLE-LOAD"} <= rule_ids


def test_scan_text_detects_hardcoded_secret_but_not_env_lookup():
    code = 'password = "hunter2xyz"\ntoken = os.getenv("TOKEN")\n'
    findings = scan_code_text(code)
    secret_findings = [f for f in findings if f.rule_id == "PY-HARDCODED-SECRET"]
    assert len(secret_findings) == 1
    assert "password" in secret_findings[0].message.lower()


def test_scan_text_clean_code_has_no_findings():
    code = "def add(a, b):\n    return a + b\n"
    assert scan_code_text(code) == []


def test_code_security_agent_end_to_end(monkeypatch):
    monkeypatch.setattr(TierClient, "chat", _fake_chat_returning("Summary of real findings."))
    agent = CodeSecurityAgent()
    result = agent.run(Task(text="eval(x)\n", context=TaskContext()))
    assert result.ok is True
    assert result.raw["findings"]
    assert result.raw["findings"][0]["rule_id"] == "PY-EVAL-EXEC"


def test_code_security_agent_falls_back_to_raw_findings_on_llm_error(monkeypatch):
    monkeypatch.setattr(TierClient, "chat",
                        lambda self, node, messages, tools=None: LLMResult(error="boom", provider="none"))
    agent = CodeSecurityAgent()
    result = agent.run(Task(text="eval(x)\n", context=TaskContext()))
    assert result.ok is False
    assert "PY-EVAL-EXEC" in result.text  # raw findings, never silently lost


# ── supply_chain_agent ───────────────────────────────────────────────────

def test_parse_requirements_separates_pinned_from_unpinned():
    text = "requests==2.31.0\nflask>=2.0\n# a comment\nnumpy\n"
    pinned, skipped = parse_requirements(text)
    assert pinned == [("requests", "2.31.0")]
    assert any("flask" in s for s in skipped)
    assert any("numpy" in s for s in skipped)


def test_scan_dependencies_reports_a_real_osv_finding(monkeypatch):
    monkeypatch.setattr("app.framework.agents.supply_chain_agent.query_osv",
                        lambda name, version, ecosystem="PyPI": (
                            [{"id": "GHSA-test-1234", "summary": "A known issue."}]
                            if name == "vulnerable-pkg" else []))
    result = scan_dependencies("vulnerable-pkg==1.0.0\nsafe-pkg==2.0.0\n")
    assert result.scanned == ["vulnerable-pkg==1.0.0", "safe-pkg==2.0.0"]
    assert len(result.findings) == 1
    assert result.findings[0].rule_id == "GHSA-test-1234"


def test_query_osv_never_raises_on_network_failure(monkeypatch):
    import httpx as httpx_module

    def broken_post(*a, **k):
        raise httpx_module.ConnectError("no network")
    monkeypatch.setattr("httpx.post", broken_post)
    assert query_osv("anything", "1.0.0") == []


def test_supply_chain_agent_end_to_end(monkeypatch):
    monkeypatch.setattr("app.framework.agents.supply_chain_agent.query_osv",
                        lambda name, version, ecosystem="PyPI": [])
    monkeypatch.setattr(TierClient, "chat", _fake_chat_returning("Nothing found."))
    agent = SupplyChainAgent()
    result = agent.run(Task(text="safe-pkg==1.0.0\n", context=TaskContext()))
    assert result.ok is True
    assert result.raw["scanned"] == ["safe-pkg==1.0.0"]


# ── cloud_config_agent ───────────────────────────────────────────────────

def test_scan_cloud_text_detects_privileged_and_open_cidr():
    config = 'privileged: true\ncidr_blocks = ["0.0.0.0/0"]\n'
    findings = scan_cloud_text(config)
    rule_ids = {f.rule_id for f in findings}
    assert {"CFG-PRIVILEGED", "CFG-OPEN-CIDR"} <= rule_ids


def test_scan_cloud_text_flags_dockerfile_with_no_user_directive():
    dockerfile = "FROM python:3.12-slim\nCOPY . .\nCMD [\"python\", \"app.py\"]\n"
    findings = scan_cloud_text(dockerfile, location_prefix="Dockerfile")
    assert any(f.rule_id == "CFG-NO-USER-DIRECTIVE" for f in findings)


def test_scan_cloud_text_does_not_flag_dockerfile_with_a_real_user_directive():
    dockerfile = "FROM python:3.12-slim\nUSER appuser\nCMD [\"python\", \"app.py\"]\n"
    findings = scan_cloud_text(dockerfile, location_prefix="Dockerfile")
    assert not any(f.rule_id == "CFG-NO-USER-DIRECTIVE" for f in findings)


def test_cloud_config_agent_end_to_end(monkeypatch):
    monkeypatch.setattr(TierClient, "chat", _fake_chat_returning("Summary."))
    agent = CloudConfigAgent()
    result = agent.run(Task(text="privileged: true\n", context=TaskContext()))
    assert result.ok is True
    assert any(f["rule_id"] == "CFG-PRIVILEGED" for f in result.raw["findings"])


# ── shared path-traversal guard (all three agents) ───────────────────────

def test_resolve_within_rejects_a_path_outside_the_base_dir(tmp_path):
    base = tmp_path / "ingest"
    base.mkdir()
    try:
        resolve_within("../../etc/passwd", base)
        assert False, "expected UnsafePathError"
    except UnsafePathError:
        pass


def test_security_agents_skip_rather_than_read_an_unsafe_path(monkeypatch, tmp_path):
    monkeypatch.setattr(TierClient, "chat", _fake_chat_returning("n/a"))
    for AgentCls in (CodeSecurityAgent, SupplyChainAgent, CloudConfigAgent):
        agent = AgentCls()
        result = agent.run(Task(text="", context=TaskContext(),
                                metadata={"path": "../../../etc/passwd"}))
        assert result.raw["scanned"] == []
        assert result.raw["skipped"]
