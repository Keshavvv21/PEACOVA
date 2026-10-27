# Memory management — conversation memory, model memory, and sizing math

Two genuinely different kinds of "memory" show up in this codebase, and
this doc covers both real, grounded facts — not a generic ML-infra
essay:

1. **Conversation memory** — what a chat session remembers across many
   turns, and how it stays bounded instead of growing forever.
2. **Model/GPU memory** — what it actually costs, in bytes, to run or
   fine-tune the transformer models this stack talks to (local SLMs via
   Ollama, the nightly LoRA fine-tune job), and the real techniques
   (quantization, LoRA/PEFT, mixed precision) already in use here.

## 1. Conversation memory — `app/agent/memory.py` + `sessions.py`

Every track (native/langgraph/adk — Claude too, via native's helpers)
shares the identical mechanism, not three separate implementations:

- `sessions.save()` trims `state["messages"]` to the last
  `SESSION_WINDOW_TURNS` (config.py, default 12) before persisting to
  the `ChatSessions` table (see `README_DATA_MODEL.md`).
- Whatever's about to be trimmed off the front doesn't just vanish —
  `memory.fold_into_summary()` compresses it into a running summary
  string via one more `get_llm_client()` call (tier `"bulk"`: background
  housekeeping, not something the user is waiting on), merged with
  whatever was already summarized.
- Every track reads `state["summary"]` back out and prepends it as one
  system message alongside the normal rolling window — see
  `native_agent.handle_message`, `langgraph_agent.perceive_intent`, and
  `adk_agent._perceive`'s `summary` param.
- **Fail-soft, always**: if every LLM provider is down and the fold call
  bottoms out at the deterministic `RuleBased` provider, that output is
  explicitly rejected (it isn't even trying to summarize) — the function
  returns the previous summary unchanged rather than saving a bogus one.

This is genuinely how ADK, LangGraph, and the native track ALL get
"long conversation" memory in this stack — not three different memory
systems, one shared compression layer every framework adapter reads
from identically (§6.5 equivalence, same principle applied to memory as
to everything else in this codebase).

## 2. Model memory — the real math

`app/llm/memory_estimator.py` (new) implements the standard transformer
memory-sizing formulas — real math an operator can use before running
the nightly LoRA job or pointing `OLLAMA_BASE_URL` at a bigger local
model, not a rule of thumb.

### Parameter memory

`bytes = num_params × bytes_per_element`, where the element size depends
on precision:

| dtype | bytes/param | Used where in this codebase |
|---|---|---|
| fp32 | 4 | AdamW's master-weight copy during fine-tuning |
| fp16 / bf16 | 2 | The base model during inference AND fine-tuning (frozen weights) |
| int8 | 1 | A more aggressively quantized local model |
| int4 | 0.5 | Most Ollama GGUF models by default (e.g. `Q4_K_M`) — see below |

### Activation memory (per transformer layer)

```
elements = 32 · S · B · H + 4 · A · S² · B
```
`S` = sequence length, `B` = batch size, `H` = hidden size, `A` =
attention heads. The `32·S·B·H` term covers the self-attention block's
linear projections/Q·K matrices/softmax output, the MLP block's two
linear layers + GeLU, and the layer norms; the `4·A·S²·B` term is the
attention-score matrix itself — the reason activation memory blows up
non-linearly with context length, not just linearly (see
`test_memory_estimator.py`'s `test_inference_activation_memory_grows_quadratically_with_sequence_length`
for this verified as real behavior, not just claimed).

**Inference** only needs ONE layer's activations live at a time
(`inference_memory_estimate()`); **fine-tuning** needs EVERY layer's,
since the backward pass needs them all (`lora_finetune_memory_estimate()`
multiplies by `num_layers`).

### Optimizer memory — why LoRA/PEFT matters here for real

Full fine-tuning with AdamW costs, PER TRAINABLE PARAMETER: two fp32
moment buffers (8 bytes) + one fp32 master-weight copy (4 bytes) + one
fp16 gradient (2 bytes) = **14 bytes/trainable-param** — on top of the
frozen weights themselves. For a 500M-parameter model, that's 7GB just
for optimizer state if EVERY parameter were trainable.

This codebase's real nightly fine-tune job
(`scripts/finetune_local_slm.py`, README's "PEFT, mechanically" section)
never does that: LoRA freezes the base model and trains only small
adapter matrices — **~0.88% of parameters** for the smoke-tested
checkpoint size (`r=8, alpha=16` on the attention/MLP projections). Run
the same 14-bytes/param AdamW cost against ~4.4M trainable params
instead of ~500M, and optimizer memory drops from ~7GB to ~60MB — a
>100× reduction, verified in `test_memory_estimator.py`'s
`test_lora_finetune_estimate_matches_the_real_nightly_job_shape` (asserts
LoRA's optimizer footprint is under 1/50th of what full fine-tuning of
the same model would cost). This is WHY the nightly job is CPU-tractable
at all on ordinary hardware — not a claim, a measured consequence of the
real technique already in production here.

### Quantization — what Ollama already does for you

Every local SLM this stack talks to via `OLLAMA_BASE_URL` is typically
already a quantized GGUF file (commonly 4-bit, e.g. `Q4_K_M`) — Ollama
handles this transparently; `memory_estimator.quantization_savings()`
makes the real number concrete: fp16 → int4 is a **75% memory
reduction** (verified in tests), the difference between a model that
fits on a laptop's RAM and one that doesn't.

### Mixed precision

The base model already runs in bf16/fp16 throughout this stack (both
inference and the frozen weights during fine-tuning) rather than fp32 —
half the memory of full precision with negligible accuracy impact for
these workloads, the same "mixed precision" technique the wider
industry uses, applied here by default rather than as an opt-in flag.

### Gradient checkpointing

Not currently used by `scripts/finetune_local_slm.py` — worth naming
honestly as a real technique NOT yet applied here, not just omitting
it: trading recomputation time for memory by not retaining every
layer's activations, recomputing some during the backward pass instead.
Given the nightly job already fits comfortably in CPU RAM at LoRA's
~0.88% trainable-parameter scale (see the worked estimate above), this
hasn't been necessary — a real, honest reason to skip a technique,
rather than an oversight.

## Worked example — sizing the nightly job

```python
from app.llm.memory_estimator import lora_finetune_memory_estimate

estimate = lora_finetune_memory_estimate(
    num_params=500_000_000, num_layers=24, hidden_size=1536,
    num_heads=16, trainable_fraction=0.0088, seq_len=2048, batch_size=1,
)
# MemoryEstimate(total_gb=..., breakdown={'base_params_gb': ..., 'trainable_params': 4_400_000,
#                                          'optimizer_gb': ..., 'activation_gb': ...})
```

Use this BEFORE changing `--max-minutes`/model size in
`scripts/finetune_local_slm.py`, to know whether the target machine's
RAM can actually hold the job — the same question `README.md`'s
"bounded by wall-clock" design note is already implicitly answering by
trial and error; this makes it a calculation instead.
