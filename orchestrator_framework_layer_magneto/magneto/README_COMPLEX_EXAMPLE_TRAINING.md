# Training harder on complex examples

## The problem this addresses

Live-tested during this session: the local SLM (`ollama-coder`,
Qwen2.5-Coder 1.5B) was asked a genuinely complex prompt ("design a
distributed job queue with retries, leases, heartbeats, and crash
recovery"). It took 60 seconds (CPU-only) and produced a structurally
correct but shallow answer — it named the right concepts but missed the
actual hard parts (idempotency on retry, races between lease expiry and
a slow-but-alive worker, dead-lettering after max retries).

That ceiling is set by the model's parameter count, not by how it's
trained. Fine-tuning (LoRA/PEFT) on a handful of examples teaches
**style and formatting consistency** — it does not add reasoning depth
beyond what a 1.5B model can support. Nothing in this document changes
that ceiling. What it does do: make the model measurably more
consistent on the *specific shapes* of hard prompt it's actually seen
worked examples of, by making sure those examples pull harder during
training than an average example does.

If you need genuinely strong answers to complex prompts today, route
them to a hosted model (Auto Hybrid / an explicit pick) rather than
force them onto the SLM — that's exactly what Auto Hybrid and
`llm/domain_router.py` already do.

## The mechanism: `training_weight`

`scripts/finetune_local_slm.py` already does **reward-weighted**
supervised fine-tuning: each example's loss is scaled by its own
`reward` (0.0-1.0) before backprop, so a fully-correct example already
teaches more than a partially-correct one.

This session added a second, orthogonal knob: `training_weight`
(default `1.0`, so every example logged before this existed is
unaffected). The two get multiplied together into the one loss scale
the trainer already applies —
`scripts/finetune_local_slm.py`'s `build_dataset()`:

```python
weight = reward * training_weight
```

`reward` still means "was this answer correct." `training_weight` means
"how much should a correct example of this kind count relative to
another correct example" — the actual lever for "train harder on the
complex ones specifically," without hand-duplicating rows in the JSONL
log to get the same effect.

## How to log a complex example with elevated weight

`app/agent/finetune_log.py`'s `log_ground_truth_turn()` takes the new
`training_weight` kwarg directly:

```python
from app.agent.finetune_log import log_ground_truth_turn

log_ground_truth_turn(
    messages=[...],       # the prompt, as a real chat messages list
    reply="...",          # a REAL, reviewed-quality answer — from a
                           # stronger hosted model, or a human
    provider="gemini-flash",
    domain="coding",      # writes to <domain>_turns.jsonl, same file
                           # the nightly per-domain fine-tune reads
    training_weight=2.0,  # this example counts as two ordinary ones
)
```

## The ready-made script

`scripts/log_complex_ground_truth.py` does the above for a curated set
of L3/L4-shaped prompts (the same shapes the live test found the SLM
struggling with — distributed systems design, multi-symptom diagnosis,
cost/quality tradeoffs, concurrency correctness), asking a real,
already-configured hosted provider for each answer and logging it with
`training_weight` between 1.5 and 2.0 depending on how hard the prompt
is.

Run it inside a container that already has real provider API keys
(`ak-chatbot`) — it imports this repo's own `llm.router`/
`agent.finetune_log` directly, so it works whether or not the API
server itself is up:

```bash
docker exec ak-chatbot python -m scripts.log_complex_ground_truth \
    --provider gemini-flash --domain coding
```

- `--provider` — any configured, working provider key from
  `llm/router.py` (default `gemini-flash`, already live in this
  deployment).
- `--domain` — which `<domain>_turns.jsonl` to write into. Omit to log
  to the generic corpus instead.

Logged examples are picked up by the very next nightly fine-tune run
for that domain the same way any other ground-truth example is —
no separate wiring needed. To add your OWN curated complex prompts,
edit `COMPLEX_PROMPTS` at the top of the script (each entry is
`(prompt, training_weight)`).

## What this does *not* do

- It does not make the SLM as capable as a hosted model on complex
  reasoning. See "The problem this addresses" above.
- It does not replace the regression gate
  (`scripts/eval/regression_gate.py`) — a candidate trained with these
  examples still has to score at least as well as the currently-serving
  model to be promoted. A worse candidate still gets correctly refused,
  same as any other night.
- `training_weight` is not currently surfaced anywhere in the
  PEFT/Train-Algo-AI dashboards — it's a training-input control, not a
  reported metric.
