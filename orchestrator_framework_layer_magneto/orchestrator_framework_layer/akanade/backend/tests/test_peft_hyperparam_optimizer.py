"""Tests for scripts/peft_hyperparam_optimizer.py — the 10th optimizer:
which nightly PEFT hyperparameter preset to train with, learned from
real regression-gate outcomes. Lives in the main (light) test suite
deliberately: this module only needs numpy (already a main-app
dependency, transitively via chromadb), not the heavy torch/transformers/
peft venv finetune_local_slm.py itself needs.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import peft_hyperparam_optimizer as peft_bandit  # noqa: E402


class TestArms:
    def test_three_named_presets_exist(self):
        assert set(peft_bandit.ARMS) == {"conservative", "balanced", "aggressive"}

    def test_every_arm_declares_the_three_interacting_hyperparams(self):
        for name, hp in peft_bandit.ARMS.items():
            assert set(hp) == {"lora_r", "lora_alpha", "min_reward"}, name

    def test_balanced_matches_finetune_local_slm_pys_long_standing_default(self):
        # r=8/alpha=16/min_reward=0.7 is that script's own hand-picked
        # default — "balanced" has to stay the cold-start-safe middle
        # option, not silently drift from it.
        assert peft_bandit.ARMS["balanced"] == {"lora_r": 8, "lora_alpha": 16, "min_reward": 0.7}


class TestPeftHyperparamOptimizer:
    def test_select_arm_returns_a_real_arm_and_its_hyperparams(self, tmp_path):
        opt = peft_bandit.PeftHyperparamOptimizer(tmp_path / "state.json")
        arm, hp = opt.select_arm(curated_count=200, avg_reward=0.8)
        assert arm in peft_bandit.ARMS
        assert hp == peft_bandit.ARMS[arm]

    def test_select_arm_full_also_returns_the_context_used(self, tmp_path):
        opt = peft_bandit.PeftHyperparamOptimizer(tmp_path / "state.json")
        arm, hp, context = opt.select_arm_full(curated_count=250, avg_reward=0.9)
        assert context == pytest.approx([0.5, 0.9])  # 250/500, avg_reward verbatim

    def test_context_clamps_curated_count_at_500(self, tmp_path):
        opt = peft_bandit.PeftHyperparamOptimizer(tmp_path / "state.json")
        _, _, context = opt.select_arm_full(curated_count=5000, avg_reward=1.0)
        assert context[0] == 1.0  # never exceeds 500/500, however large the real corpus is

    def test_record_outcome_persists_state_to_disk(self, tmp_path):
        path = tmp_path / "state.json"
        opt = peft_bandit.PeftHyperparamOptimizer(path)
        arm, _, context = opt.select_arm_full(curated_count=100, avg_reward=0.75)
        opt.record_outcome(arm, context, score_margin=0.05)
        assert path.exists()

    def test_a_repeatedly_winning_arm_gets_picked_more_often_after_reloading(self, tmp_path):
        path = tmp_path / "state.json"
        opt = peft_bandit.PeftHyperparamOptimizer(path)
        context = opt._context(curated_count=200, avg_reward=0.8)

        # Strongly reward "conservative" every time, punish the others,
        # many rounds — enough for the posterior to clearly separate.
        for _ in range(40):
            opt.record_outcome("conservative", context, score_margin=0.08)
            opt.record_outcome("balanced", context, score_margin=-0.05)
            opt.record_outcome("aggressive", context, score_margin=-0.08)

        reloaded = peft_bandit.PeftHyperparamOptimizer(path)  # fresh process, state read from disk
        wins = {"conservative": 0, "balanced": 0, "aggressive": 0}
        for _ in range(60):
            arm, _, _ = reloaded.select_arm_full(curated_count=200, avg_reward=0.8)
            wins[arm] += 1
        assert wins["conservative"] > wins["balanced"]
        assert wins["conservative"] > wins["aggressive"]

    def test_a_missing_state_file_starts_fresh_rather_than_crashing(self, tmp_path):
        opt = peft_bandit.PeftHyperparamOptimizer(tmp_path / "does_not_exist_yet.json")
        arm, hp = opt.select_arm(curated_count=0, avg_reward=0.0)
        assert arm in peft_bandit.ARMS

    def test_a_corrupt_state_file_starts_fresh_rather_than_crashing(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not valid json", encoding="utf-8")
        opt = peft_bandit.PeftHyperparamOptimizer(path)
        arm, hp = opt.select_arm(curated_count=10, avg_reward=0.5)
        assert arm in peft_bandit.ARMS

    def test_positive_and_negative_margins_push_the_posterior_in_opposite_directions(self, tmp_path):
        # tanh(margin * 10) — a real regression-gate pass (positive
        # margin) must push the arm's posterior weighted-reward sum (f)
        # up, a real fail (negative margin) must push it down, for the
        # exact same starting state and context.
        context = peft_bandit.PeftHyperparamOptimizer._context(curated_count=100, avg_reward=0.7)

        won = peft_bandit.PeftHyperparamOptimizer(tmp_path / "won.json")
        won._core.ensure("balanced")
        won.record_outcome("balanced", context, score_margin=0.05)

        lost = peft_bandit.PeftHyperparamOptimizer(tmp_path / "lost.json")
        lost._core.ensure("balanced")
        lost.record_outcome("balanced", context, score_margin=-0.05)

        f_after_win = won._core._f["balanced"]
        f_after_loss = lost._core._f["balanced"]
        assert (f_after_win > f_after_loss).all()
