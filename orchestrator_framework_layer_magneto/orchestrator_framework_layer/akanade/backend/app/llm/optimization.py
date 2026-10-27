"""Real code, not just documentation, for ARCHITECTURE.md §7's model
optimization strategies — as an `OptimizationProfile` any model provider
can DECLARE (metadata describing how its backend is actually deployed,
since ALGO AKANADE's own process never runs the optimization itself — see the
module docstring below) plus genuine `extra_body` pass-through
(providers.py's OpenAICompatible/CustomProvider/HuggingFaceProvider all
now accept it) so an operator can actually forward server-specific
sampling/optimization flags into a real request, not just describe them.

Every one of these — KV cache, continuous batching, speculative
decoding, optimized attention (FlashAttention/PagedAttention), and
tensor/pipeline/data parallelism — is a property of the INFERENCE
SERVER (vLLM, TGI, ...) behind a provider's `base_url`, configured once
at that server's own startup (`vllm serve --tensor-parallel-size 4
--speculative-model ...`), not something a per-request HTTP call from
this process can add. `OptimizationProfile` is therefore declarative —
"this endpoint IS configured with X" — not a switch that turns X on;
setting `tensor_parallel_size=4` here does not itself shard anything.
What genuinely happens code-side is `extra_body`: any provider that
supports extra sampling/inference parameters on its chat-completions
endpoint (most vLLM/TGI deployments do — e.g. `top_k`,
`repetition_penalty`, `use_beam_search`, a server-specific speculative-
decoding knob) gets them from `extra_body`, merged verbatim into the
real request payload every single call.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizationProfile:
    """What a model provider's backend is DECLARED to run — set from
    config (see llm/providers.py's CustomProvider/HuggingFaceProvider,
    llm/router.py's construction of them) to match however that backend
    was actually deployed. Purely descriptive metadata: surfaced via
    GET /api/v1/models so a caller/operator can see it, never consulted
    to decide HOW to make a request (that would require this process to
    actually understand vLLM/TGI internals, which it deliberately does
    not)."""

    kv_cache: bool = False
    continuous_batching: bool = False
    speculative_decoding: bool = False
    speculative_draft_model: str | None = None
    # "default" | "flash_attention" | "paged_attention"
    attention_backend: str = "default"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_replicas: int = 1

    def enabled_strategies(self) -> list[str]:
        """Which named strategies this profile declares active — the
        "combining approaches" list ARCHITECTURE.md §7 describes,
        computed from whichever fields are actually set, not hard-coded
        per profile."""
        strategies: list[str] = []
        if self.kv_cache:
            strategies.append("kv_cache")
        if self.continuous_batching:
            strategies.append("continuous_batching")
        if self.speculative_decoding:
            strategies.append(f"speculative_decoding(draft={self.speculative_draft_model or '?'})")
        if self.attention_backend != "default":
            strategies.append(f"attention={self.attention_backend}")
        if self.tensor_parallel_size > 1:
            strategies.append(f"tensor_parallel(x{self.tensor_parallel_size})")
        if self.pipeline_parallel_size > 1:
            strategies.append(f"pipeline_parallel(x{self.pipeline_parallel_size})")
        if self.data_parallel_replicas > 1:
            strategies.append(f"data_parallel(x{self.data_parallel_replicas})")
        return strategies

    def to_dict(self) -> dict:
        return {"kv_cache": self.kv_cache, "continuous_batching": self.continuous_batching,
               "speculative_decoding": self.speculative_decoding,
               "speculative_draft_model": self.speculative_draft_model,
               "attention_backend": self.attention_backend,
               "tensor_parallel_size": self.tensor_parallel_size,
               "pipeline_parallel_size": self.pipeline_parallel_size,
               "data_parallel_replicas": self.data_parallel_replicas,
               "enabled_strategies": self.enabled_strategies()}

    @classmethod
    def from_dict(cls, raw: dict) -> "OptimizationProfile":
        known = {"kv_cache", "continuous_batching", "speculative_decoding",
                "speculative_draft_model", "attention_backend", "tensor_parallel_size",
                "pipeline_parallel_size", "data_parallel_replicas"}
        return cls(**{k: v for k, v in raw.items() if k in known})


NONE_DECLARED = OptimizationProfile()
