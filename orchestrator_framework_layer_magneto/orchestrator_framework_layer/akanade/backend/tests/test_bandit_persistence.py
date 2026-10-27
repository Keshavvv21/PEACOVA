"""app/framework/bandit_persistence.py + LinearBanditCore.to_state()/
load_state() — the durable-across-restarts mechanism security_optimizer.py
opts into via `name=`. Without this, every RL optimizer's learned state
lives only in process memory and silently resets on every restart.
"""
from __future__ import annotations

from app.framework import bandit_persistence
from app.framework.bandit import LinearBanditCore
from app.security_optimizer import SecurityOptimizer


def test_to_state_and_load_state_round_trip():
    core = LinearBanditCore(feature_dim=1)
    core.update("arm-a", [1.0], 1.0)
    core.update("arm-a", [1.0], 1.0)
    core.update("arm-b", [1.0], -1.0)
    snapshot = core.to_state()

    restored = LinearBanditCore(feature_dim=1)
    restored.load_state(snapshot)
    assert (restored.posterior_mean("arm-a") == core.posterior_mean("arm-a")).all()
    assert (restored.posterior_mean("arm-b") == core.posterior_mean("arm-b")).all()


def test_save_then_load_persists_across_separate_core_instances():
    core = LinearBanditCore(feature_dim=1)
    core.update("SEC-TEST::user_prompt", [1.0], 1.0)
    bandit_persistence.save("test-optimizer", core)

    fresh_core = LinearBanditCore(feature_dim=1)
    assert fresh_core.posterior_mean("SEC-TEST::user_prompt") is None  # blank before load
    bandit_persistence.load("test-optimizer", fresh_core)
    assert fresh_core.posterior_mean("SEC-TEST::user_prompt") is not None


def test_load_is_a_noop_when_nothing_was_ever_saved():
    core = LinearBanditCore(feature_dim=1)
    bandit_persistence.load("never-saved-optimizer", core)  # must not raise
    assert core.posterior_mean("anything") is None


def test_security_optimizer_with_a_name_persists_learned_state_across_instances():
    opt1 = SecurityOptimizer(name="test-security-optimizer-persistence")
    for _ in range(8):
        opt1.record_feedback(["SEC-UNAUTHORIZED-ACTION: msg"], source="user_prompt",
                             approved=False)
    note1 = opt1.confidence_note("SEC-UNAUTHORIZED-ACTION", "user_prompt")
    assert note1 is not None

    # A FRESH instance with the SAME name loads what opt1 already learned.
    opt2 = SecurityOptimizer(name="test-security-optimizer-persistence")
    assert opt2.confidence_note("SEC-UNAUTHORIZED-ACTION", "user_prompt") == note1


def test_security_optimizer_without_a_name_never_persists():
    opt1 = SecurityOptimizer()
    for _ in range(8):
        opt1.record_feedback(["SEC-EXFIL-FORWARD: msg"], source="user_prompt", approved=False)
    # A different unnamed instance starts blank — no shared/durable state.
    opt2 = SecurityOptimizer()
    assert opt2.confidence_note("SEC-EXFIL-FORWARD", "user_prompt") is None
