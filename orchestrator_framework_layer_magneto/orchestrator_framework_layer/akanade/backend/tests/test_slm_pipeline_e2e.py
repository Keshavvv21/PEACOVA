"""End-to-end test for the SLM Generation Layer's operational surface
(README_SLM_ALL.md) — NOT the actual training/quantization (that needs
a real llama.cpp checkout + GPU/CPU time and is covered by this
session's honesty sections instead), but everything AROUND it that IS
real, wireable Python: configure what should train tonight, simulate a
successful nightly run registering a domain-specific model, confirm
the version-history dashboard reflects it, confirm domain_router.py
discovers the newly-registered provider for real (no monkeypatched
available_providers() here — a genuine admin_providers.py row), and
confirm a real chat turn routed through Domain mode actually reaches
that specific provider.

Only the provider's own outbound HTTP call (OpenAICompatible.chat) is
faked — every other layer (config API, provider registry,
domain_router's live discovery, TierClient's chain-building,
native_agent's handle_message) runs for real, exactly as it would
against a real registered Algo-Akanade-SLM-legal.
"""
from __future__ import annotations

import time

from app.llm.domain_router import DomainSelection, domain_router, registered_domains
from app.llm.providers import LLMResult, OpenAICompatible

from tests.conftest import auth

DOMAIN = "legal"
PROVIDER_NAME = f"algo-akanade-slm-{DOMAIN}"


def test_full_nightly_cycle_and_domain_routed_chat_turn(client, exec_token, member_token, monkeypatch):
    # ── Step 1: an admin configures what should train tonight ──────────
    r = client.put(f"/api/admin/finetune-config/{DOMAIN}",
                   json={"domain": DOMAIN, "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                         "ollama_base_tag": "qwen2.5:1.5b", "max_minutes": 90, "enabled": True},
                   headers=auth(exec_token))
    assert r.status_code == 200

    # ── Step 2: this is exactly the shape deploy/nightly_finetune.sh's
    # fetch_configs() parses (r['domain'], r['base_model'],
    # r['ollama_base_tag'], r['max_minutes']) — confirm the contract. ──
    r = client.get("/api/admin/finetune-config/enabled", headers=auth(exec_token))
    assert r.status_code == 200
    configs = r.json()
    legal_config = next(c for c in configs if c["domain"] == DOMAIN)
    assert legal_config["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert legal_config["ollama_base_tag"] == "qwen2.5:1.5b"
    assert legal_config["max_minutes"] == 90

    # ── Step 3: simulate convert_and_register_ollama_model.sh's own
    # successful ending for tonight — the candidate passed the sanity
    # check and the regression gate, so it registers itself. ──────────
    r = client.post("/api/admin/providers",
                    json={"name": PROVIDER_NAME, "base_url": "http://ak-ollama:11434/v1",
                          "model": f"{PROVIDER_NAME}:20260828",
                          "api_key": "not-needed-ollama-ignores-this",
                          "tiers": ["realtime", "bulk"]},
                    headers=auth(exec_token))
    assert r.status_code == 201

    # ── Step 4: the script also reports the outcome to the dashboard ──
    r = client.post("/api/admin/slm-versions",
                    json={"tag": f"{PROVIDER_NAME}:20260828", "base_tag": "qwen2.5:1.5b",
                          "quant_type": "Q4_K_M", "sanity_check": "passed",
                          "regression_gate": "passed", "registered": True, "domain": DOMAIN},
                    headers=auth(exec_token))
    assert r.status_code == 201

    # ── Step 5: the Admin dashboard shows it as the live version for
    # the "legal" domain. ──────────────────────────────────────────────
    r = client.get("/api/admin/slm-versions", headers=auth(exec_token))
    versions = [v for v in r.json() if v["domain"] == DOMAIN]
    assert versions and versions[0]["registered"] is True
    assert versions[0]["tag"] == f"{PROVIDER_NAME}:20260828"

    # ── Step 6: domain_router.py discovers it live — a REAL admin_
    # providers.py row, not a monkeypatched available_providers(). ────
    assert registered_domains().get(DOMAIN) == PROVIDER_NAME

    # ── Step 7: a real chat turn in Domain mode actually reaches the
    # newly-registered provider. The domain PICK is forced deterministic
    # (Thompson Sampling is randomized pre-training and already has its
    # own dedicated unit tests in test_domain_router.py) — what this
    # verifies is everything AROUND that pick: does the selection
    # actually thread through native_agent.py -> TierClient.preferred
    # -> the chain -> the correct provider instance answering. ────────
    monkeypatch.setattr(domain_router, "choose",
                        lambda message, tier: DomainSelection(arm=DOMAIN, context=[0.1, 0.0, 0.0],
                                                             task_text=message))

    captured = {}

    def fake_chat(self, messages, tools=None):
        captured["provider_name"] = self.name
        return LLMResult(text="Termination requires 30 days' written notice.", provider=self.name)

    monkeypatch.setattr(OpenAICompatible, "chat", fake_chat)

    r = client.post("/api/chat/message",
                    json={"message": "What's the termination clause in a standard vendor contract?",
                          "session_id": "e2e-domain-test", "provider": "domain"},
                    headers=auth(member_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == PROVIDER_NAME
    assert captured["provider_name"] == PROVIDER_NAME
    assert "30 days" in body["reply"]

    # ── Step 8: the outcome feeds back into the SAME bandit + memory
    # domain_router.py uses for its NEXT decision (§6 of README_SLM_ALL.md
    # — "record_outcome(success) updates BOTH the bandit's posterior for
    # that arm AND the embedding memory"). A free-text answer with no
    # PM tool call is classified `out_of_scope=True` by native_agent.py's
    # own step 1066-1068 heuristic (it's built for "did this produce a
    # grounded PM action," which a legal-domain prose answer never will)
    # — so this turn's reward is genuinely NEGATIVE, not a test bug. The
    # memory/bandit update happening at all, with the correct sign, is
    # exactly what this step confirms. The actual record_outcome() call
    # now runs on a background thread (app/background.py) — the chat
    # response no longer waits on it (see native_agent.py's finish()) —
    # so this test polls briefly rather than asserting the instant the
    # HTTP response comes back, same as any real caller of this bandit's
    # NEXT decision would only ever see it eventually-consistent too.
    for _ in range(50):
        if domain_router._core.posterior_mean(DOMAIN) is not None:
            break
        time.sleep(0.02)
    assert domain_router._core.posterior_mean(DOMAIN) is not None
    bonus = domain_router._memory.bonus_for(
        "What's the termination clause in a standard vendor contract?", DOMAIN)
    assert bonus < 0
