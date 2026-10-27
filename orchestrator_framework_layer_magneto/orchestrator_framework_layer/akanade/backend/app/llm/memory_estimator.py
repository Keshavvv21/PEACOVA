"""Memory estimation — "will this LLM/fine-tune job fit on my hardware,"
made concrete with the standard transformer memory-sizing formulas (see
README_MEMORY_MANAGEMENT.md for the full derivation and worked
examples). Grounded against THIS codebase's two real local-compute
paths — Ollama SLM inference (config.py's OLLAMA_BASE_URL rung,
llm/providers.py) and the nightly LoRA fine-tune job
(scripts/finetune_local_slm.py, README.md's "PEFT, mechanically"
section) — not a generic ML-infra library. Every function here is real
math an operator can actually use before running the nightly cron job
on a given machine, not a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_GIB = 1024 ** 3

# Bytes per parameter/activation element at each precision — the same
# quantization levels README_MEMORY_MANAGEMENT.md documents (fp32/fp16/
# bf16 dense, int8/int4 quantized).
BYTES_PER_ELEMENT: dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5,
}


@dataclass
class MemoryEstimate:
    total_gb: float
    breakdown: dict[str, float] = field(default_factory=dict)


def _activation_elements(*, seq_len: int, batch_size: int, hidden_size: int,
                         num_heads: int) -> int:
    """The self-attention + MLP + layer-norm breakdown for ONE
    transformer layer — 32·S·B·H (linear projections, Q/K matrices,
    softmax/attention output, MLP's two linear layers, GeLU, layer
    norms) + 4·A·S²·B (the attention-score matrix itself, which scales
    with sequence length SQUARED — the real reason long-context
    inference gets memory-expensive fast). See
    README_MEMORY_MANAGEMENT.md for the term-by-term derivation."""
    return 32 * seq_len * batch_size * hidden_size + 4 * num_heads * seq_len ** 2 * batch_size


def param_memory_bytes(num_params: int, dtype: str = "fp16") -> float:
    return num_params * BYTES_PER_ELEMENT[dtype]


def inference_memory_estimate(*, num_params: int, num_layers: int, hidden_size: int,
                              num_heads: int, seq_len: int = 4096, batch_size: int = 1,
                              dtype: str = "fp16") -> MemoryEstimate:
    """Real inference footprint: model weights, plus ONE layer's worth
    of activations live at a time (inference streams layer by layer —
    unlike fine-tuning below, it never needs every layer's activations
    retained simultaneously for a backward pass)."""
    params = param_memory_bytes(num_params, dtype)
    activation = _activation_elements(seq_len=seq_len, batch_size=batch_size,
                                      hidden_size=hidden_size,
                                      num_heads=num_heads) * BYTES_PER_ELEMENT[dtype]
    total = params + activation
    return MemoryEstimate(
        total_gb=round(total / _GIB, 3),
        breakdown={"params_gb": round(params / _GIB, 3),
                  "activation_gb": round(activation / _GIB, 3)})


def lora_finetune_memory_estimate(*, num_params: int, num_layers: int, hidden_size: int,
                                  num_heads: int, trainable_fraction: float = 0.0088,
                                  seq_len: int = 2048, batch_size: int = 1,
                                  base_dtype: str = "bf16") -> MemoryEstimate:
    """Approximates THIS codebase's real nightly LoRA job
    (scripts/finetune_local_slm.py — r=8/alpha=16, ~0.88% trainable
    parameters for the smoke-tested checkpoint size, the
    `trainable_fraction` default below):

    - Base model frozen in `base_dtype` — no fp32 master copy needed for
      the FROZEN weights (only the trainable LoRA adapter needs one).
    - AdamW optimizer state for ONLY the trainable LoRA parameters: two
      fp32 moment buffers + one fp32 master copy + one fp16 gradient per
      trainable parameter (4+4+4+2 = 14 bytes/trainable-param) — the
      same real AdamW memory shape README_MEMORY_MANAGEMENT.md derives,
      just scoped to ~1% of the model instead of all of it (the whole
      point of PEFT).
    - Activations for ALL layers, not one — every layer's activations
      must be retained for the backward pass during training, unlike
      inference.
    """
    base_params = param_memory_bytes(num_params, base_dtype)
    trainable_params = int(num_params * trainable_fraction)
    optimizer = trainable_params * 14  # 4+4+4+2 bytes, see docstring above
    activation = (_activation_elements(seq_len=seq_len, batch_size=batch_size,
                                       hidden_size=hidden_size, num_heads=num_heads)
                 * num_layers * BYTES_PER_ELEMENT[base_dtype])
    total = base_params + optimizer + activation
    return MemoryEstimate(
        total_gb=round(total / _GIB, 3),
        breakdown={"base_params_gb": round(base_params / _GIB, 3),
                  "trainable_params": trainable_params,
                  "optimizer_gb": round(optimizer / _GIB, 3),
                  "activation_gb": round(activation / _GIB, 3)})


def quantization_savings(num_params: int, *, from_dtype: str = "fp16",
                         to_dtype: str = "int4") -> dict[str, float]:
    """How much a quantization pass (e.g. what Ollama already applies to
    every local GGUF model this codebase's OLLAMA_BASE_URL rung talks
    to) actually saves — real arithmetic, not a rule of thumb."""
    before = param_memory_bytes(num_params, from_dtype)
    after = param_memory_bytes(num_params, to_dtype)
    return {"before_gb": round(before / _GIB, 3), "after_gb": round(after / _GIB, 3),
           "saved_gb": round((before - after) / _GIB, 3),
           "saved_pct": round((1 - after / before) * 100, 1)}
