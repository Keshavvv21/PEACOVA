"""app/agent/peft_training_tool.py — "train the local model on this
file/link/text" from chat, and native_agent.py's domain-confirmation
clarify() for it.
"""
from __future__ import annotations

import httpx
import pytest

from app.agent import native_agent, peft_training_tool, tools
from app.agent.finetune_log import _log_path


class _FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=self)


def test_request_training_with_pasted_content_logs_to_the_domain_file():
    result = peft_training_tool.request_training(
        domain="trading-test-domain", topic="RSI indicator", content="RSI measures momentum.")
    assert result["domain"] == "trading-test-domain"
    assert result["source"] == "pasted text"
    assert result["chars_logged"] > 0

    path = _log_path("trading-test-domain")
    assert path.exists()
    assert "RSI measures momentum" in path.read_text()


def test_request_training_fetches_and_strips_html_from_a_url(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResponse(
        "<html><body><p>Backtesting means testing a strategy on historical data.</p></body></html>"))
    result = peft_training_tool.request_training(
        domain="trading-test-domain-2", topic="Backtesting", url="https://example.com/backtesting")
    assert result["source"] == "https://example.com/backtesting"
    path = _log_path("trading-test-domain-2")
    text = path.read_text()
    assert "Backtesting means testing a strategy" in text
    assert "<p>" not in text  # HTML tags stripped, not logged as training noise


def test_request_training_requires_content_or_url():
    with pytest.raises(peft_training_tool.TrainingRequestError):
        peft_training_tool.request_training(domain="general", topic="nothing given")


def test_fetch_url_text_raises_a_clean_error_on_network_failure(monkeypatch):
    def _raise(*a, **k):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(httpx, "get", _raise)
    with pytest.raises(peft_training_tool.TrainingRequestError):
        peft_training_tool.fetch_url_text("https://unreachable.example")


# ── domain confirmation happens via chat (resolve_arguments' clarify) ───

def test_resolve_arguments_asks_for_domain_when_missing():
    user = {"sub": "test-user"}
    resolved, clarify = native_agent.resolve_arguments(
        user, "request_peft_training", {"topic": "moving averages", "content": "some text"})
    assert resolved is None
    assert clarify["field"] == "domain"
    assert "domain" in clarify["question"].lower()


def test_resolve_arguments_asks_for_content_when_domain_given_but_nothing_to_train_on():
    user = {"sub": "test-user"}
    resolved, clarify = native_agent.resolve_arguments(
        user, "request_peft_training", {"domain": "trading", "topic": "moving averages"})
    assert resolved is None
    assert clarify["field"] == "content"


def test_resolve_arguments_passes_through_once_domain_and_content_are_both_given():
    user = {"sub": "test-user"}
    resolved, clarify = native_agent.resolve_arguments(
        user, "request_peft_training",
        {"domain": "trading", "topic": "moving averages", "content": "some real text"})
    assert clarify is None
    assert resolved["domain"] == "trading"


def test_request_peft_training_never_requires_confirmation():
    user = {"sub": "test-user"}
    assert native_agent.requires_confirmation(user, "request_peft_training") is False


def test_execute_request_peft_training_returns_ok(monkeypatch):
    user = {"sub": "test-user", "role": "Member"}
    result = tools.execute(user, "request_peft_training",
                           {"domain": "trading-test-exec", "topic": "test", "content": "training text"})
    assert result["ok"] is True
    assert result["kind"] == "peft_training_request"
    assert result["domain"] == "trading-test-exec"
