"""app/llm/memory_estimator.py — real transformer memory-sizing math
(see README_MEMORY_MANAGEMENT.md for the derivation), sanity-checked
against known real-world orders of magnitude, not just "does it run."
"""
from __future__ import annotations

from app.llm import memory_estimator as me


def test_param_memory_scales_with_dtype():
    fp32 = me.param_memory_bytes(1_000_000_000, "fp32")
    fp16 = me.param_memory_bytes(1_000_000_000, "fp16")
    int8 = me.param_memory_bytes(1_000_000_000, "int8")
    int4 = me.param_memory_bytes(1_000_000_000, "int4")
    assert fp32 == 4e9
    assert fp16 == 2e9
    assert int8 == 1e9
    assert int4 == 0.5e9


def test_inference_estimate_for_a_70b_model_is_in_a_realistic_range():
    # Llama-3.1-70B-shaped config — the article this was built against
    # estimates ~141GB for parameters alone at fp16; our formula-driven
    # estimate should land in the same order of magnitude (not exact,
    # different rounding/assumptions, but not wildly off either).
    r = me.inference_memory_estimate(num_params=70_600_000_000, num_layers=80,
                                     hidden_size=8192, num_heads=64,
                                     seq_len=4096, batch_size=1, dtype="fp16")
    assert 100 < r.breakdown["params_gb"] < 160
    assert r.total_gb > r.breakdown["params_gb"]  # activations add on top


def test_inference_activation_memory_grows_with_batch_size():
    small = me.inference_memory_estimate(num_params=1_000_000_000, num_layers=24,
                                         hidden_size=2048, num_heads=16,
                                         seq_len=1024, batch_size=1)
    big = me.inference_memory_estimate(num_params=1_000_000_000, num_layers=24,
                                       hidden_size=2048, num_heads=16,
                                       seq_len=1024, batch_size=8)
    assert big.breakdown["activation_gb"] > small.breakdown["activation_gb"]
    assert big.breakdown["params_gb"] == small.breakdown["params_gb"]  # unaffected by batch


def test_inference_activation_memory_grows_quadratically_with_sequence_length():
    short = me.inference_memory_estimate(num_params=1_000_000_000, num_layers=24,
                                         hidden_size=2048, num_heads=16,
                                         seq_len=1024, batch_size=1)
    long = me.inference_memory_estimate(num_params=1_000_000_000, num_layers=24,
                                        hidden_size=2048, num_heads=16,
                                        seq_len=4096, batch_size=1)
    # 4x the sequence length should cost noticeably more than 4x the
    # activation memory (the S² attention term), not just linearly more.
    ratio = long.breakdown["activation_gb"] / short.breakdown["activation_gb"]
    assert ratio > 4.0


def test_lora_finetune_estimate_matches_the_real_nightly_job_shape():
    # scripts/finetune_local_slm.py's own real numbers (README's "PEFT,
    # mechanically" section): ~500M total params, ~4.4M trainable
    # (0.88%) for the smoke-tested checkpoint.
    r = me.lora_finetune_memory_estimate(num_params=500_000_000, num_layers=24,
                                         hidden_size=1536, num_heads=16,
                                         trainable_fraction=0.0088)
    assert r.breakdown["trainable_params"] == int(500_000_000 * 0.0088)
    # LoRA's whole point: optimizer memory is tiny relative to full
    # fine-tuning of the same model (which would need ~14 bytes per ALL
    # 500M params, not just the trainable ~4.4M).
    full_finetune_optimizer_gb = (500_000_000 * 14) / (1024 ** 3)
    assert r.breakdown["optimizer_gb"] < full_finetune_optimizer_gb / 50


def test_quantization_savings_int4_beats_int8_beats_fp16():
    fp16_to_int8 = me.quantization_savings(1_000_000_000, from_dtype="fp16", to_dtype="int8")
    fp16_to_int4 = me.quantization_savings(1_000_000_000, from_dtype="fp16", to_dtype="int4")
    assert fp16_to_int8["saved_pct"] == 50.0
    assert fp16_to_int4["saved_pct"] == 75.0
    assert fp16_to_int4["saved_gb"] > fp16_to_int8["saved_gb"]
