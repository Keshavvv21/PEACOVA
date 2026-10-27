"""app/llm/router.py's hard wall-clock timeout for local (Ollama-family)
rungs — real bug found live: a single call to a local rung reached via
the NORMAL chain (not an explicit preferred= pick) took 145 seconds
despite LangChainOllama's own `timeout=30.0` field, which is passed to
ChatOllama but doesn't reliably bound the actual wait under real load.
This is a genuine, deterministic ThreadPoolExecutor-based cap, applied
to every local-prefixed provider key regardless of how it was chosen.
"""
from __future__ import annotations

import time

from app.llm import router as router_module
from app.llm.providers import LLMResult
from app.llm.router import TierClient


def test_a_hanging_local_provider_is_cut_off_and_falls_through(monkeypatch):
    monkeypatch.setattr(router_module, "_LOCAL_HARD_TIMEOUT_S", 1.0)

    calls = []

    class _FakeLocalProvider:
        name = "ollama-qwen2.5:1.5b"
        timeout = 30.0

        def available(self) -> bool:
            return True

        def chat(self, messages, tools=None) -> LLMResult:
            calls.append("ollama")
            time.sleep(10)  # would hang the whole test without the hard cap
            return LLMResult(text="too slow to matter", provider=self.name)

    class _FakeHostedProvider:
        name = "fake-hosted"

        def available(self) -> bool:
            return True

        def chat(self, messages, tools=None) -> LLMResult:
            calls.append("hosted")
            return LLMResult(text="fast real answer", provider=self.name)

    fake_providers = {"ollama": _FakeLocalProvider(), "hosted": _FakeHostedProvider(),
                      "rule": _FakeHostedProvider()}
    monkeypatch.setattr(router_module, "_providers", lambda: fake_providers)
    monkeypatch.setattr(router_module, "_effective_chains",
                        lambda: {"realtime": ["ollama", "hosted", "rule"]})

    # Forces "ollama" to be tried first regardless of provider_optimizer/
    # cost_optimizer's own real ordering (which now correctly disfavors
    # a free/local rung by default — see cost_optimizer.py's
    # _FREE_PROVIDER_COLD_START_SEED) — the same explicit-pick shape
    # Hybrid/Domain mode already uses, and an equally real scenario this
    # hard timeout must protect: an explicitly-preferred local pick that
    # hangs must still be cut off, not just an organically-chosen one.
    client = TierClient(tier="realtime", preferred="ollama")
    t0 = time.time()
    result = client.chat("perceive_intent", [{"role": "user", "content": "hi"}])
    elapsed_s = time.time() - t0

    assert result.text == "fast real answer"
    assert calls == ["ollama", "hosted"]  # tried local first, cut off, fell through
    assert elapsed_s < 5.0  # nowhere near the 10s the hung local call would have taken


def test_a_fast_local_provider_still_answers_normally(monkeypatch):
    monkeypatch.setattr(router_module, "_LOCAL_HARD_TIMEOUT_S", 5.0)

    class _FakeFastLocalProvider:
        name = "ollama-qwen2.5:1.5b"
        timeout = 30.0

        def available(self) -> bool:
            return True

        def chat(self, messages, tools=None) -> LLMResult:
            return LLMResult(text="a real fast local answer", provider=self.name)

    fake_providers = {"ollama": _FakeFastLocalProvider(), "rule": _FakeFastLocalProvider()}
    monkeypatch.setattr(router_module, "_providers", lambda: fake_providers)
    monkeypatch.setattr(router_module, "_effective_chains",
                        lambda: {"realtime": ["ollama", "rule"]})

    client = TierClient(tier="realtime")
    result = client.chat("perceive_intent", [{"role": "user", "content": "hi"}])
    assert result.text == "a real fast local answer"
