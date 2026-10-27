"""One-shot / occasionally-rerun script: generates real (prompt, answer)
ground-truth pairs for GENUINELY HARD prompts by asking a strong hosted
model, then logs each with an elevated training_weight via
agent/finetune_log.py's log_ground_truth_turn() — the concrete "train
harder on the complex examples specifically" lever (see that function's
own docstring for the reward-vs-training_weight distinction, and
scripts/finetune_local_slm.py's build_dataset() for where the two get
multiplied into one loss scale).

Why this exists: a 1.5B local model's ceiling on multi-step reasoning is
set by its parameter count, not by how it's trained — PEFT on a handful
of examples teaches STYLE and formatting consistency, not new reasoning
depth. What training CAN realistically do is make the model more
consistent on the specific shapes of hard prompt it's actually seen
worked examples of. This script is how you put worked examples of those
shapes in front of it, weighted so the nightly trainer favors them.

Run inside a container that already has the real provider API keys
(ak-chatbot) — imports this repo's own router/finetune_log directly
rather than calling the HTTP API, so it works the same whether or not
the API server is even up:

    docker exec ak-chatbot python -m scripts.log_complex_ground_truth \\
        --provider gemini-flash --domain coding

--provider defaults to gemini-flash (already configured and live in
this deployment — see llm/router.py); pick any configured, working
provider instead if you'd rather distill from a different one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The prompts below are the L3/L4-shaped items from README_LIVE_PROMPT_
# TEST_REPORT's own demo runbook (architecture design, multi-symptom
# diagnosis, cost/quality tradeoffs) — the exact shapes the live test
# this session found the 1.5B model producing shallow answers for.
# training_weight=2.0: each counts as two ordinary (reward=1.0,
# weight=1.0) examples in the reward-weighted loss — enough to actually
# move the needle without letting a handful of examples dominate an
# otherwise-diverse batch outright.
COMPLEX_PROMPTS: list[tuple[str, float]] = [
    ("Design a distributed job queue with retries, leases, heartbeats, and "
     "crash recovery. Cover the data model, the failure-handling rules, and "
     "explicitly call out the race conditions a naive design would miss "
     "(a lease expiring while its worker is still alive but slow, "
     "at-least-once vs exactly-once delivery, dead-lettering after max "
     "retries).", 2.0),
    ("Diagnose a multi-symptom incident: Kafka consumer lag is climbing, a "
     "TRT inference engine has stalled, and false-positive detections are "
     "rising at the same time. Determine the most likely root cause and the "
     "order to investigate and fix things in, and explain why that order "
     "matters (fixing the wrong one first could mask or worsen the others).", 2.0),
    ("A model has AUROC 0.91 offline but only 0.18 precision once deployed. "
     "Explain exactly how both numbers can be true at once, and what you'd "
     "check first to confirm which explanation actually applies here.", 2.0),
    ("Given a fixed monthly budget and per-request cost/quality data across "
     "several model tiers, work out the maximum sustainable request volume "
     "per model choice, and explain the tradeoff between routing more "
     "traffic to a cheaper model versus a better one.", 2.0),
    ("Implement a thread-safe LRU cache in Python supporting concurrent "
     "reads and writes. Explain the specific race condition a naive "
     "(lock-per-method) implementation has under concurrent access, and how "
     "your implementation avoids it.", 1.5),
    ("Design a rate limiter using Redis with a token-bucket algorithm, "
     "usable from multiple application servers at once. Explain what goes "
     "wrong if the increment-and-check isn't atomic across servers, and how "
     "your design makes it atomic.", 1.5),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gemini-flash",
                        help="A configured, working provider key from llm/router.py "
                             "to distill answers from (default: gemini-flash).")
    parser.add_argument("--domain", default=None,
                        help="Fine-tune-config domain to log these under (e.g. 'coding'). "
                             "Omit to log to the generic/no-domain log.")
    args = parser.parse_args()

    import time

    from app.agent.finetune_log import log_ground_truth_turn
    from app.framework.agents.coding_agent import SYSTEM_PROMPT
    from app.llm.router import call_provider_directly

    # Real, observed problem this fixes: firing all 6 long, complex
    # requests back-to-back tripped Gemini's free-tier per-minute rate
    # limit (429) on every request after the first — this is a genuine
    # quota constraint, not a bug, and this is a background/occasional
    # script with no reason to race a shared free tier. 20s between
    # calls stays comfortably under a 3 RPM-class free-tier limit for
    # this kind of long-generation request.
    INTER_CALL_DELAY_SECONDS = 20.0

    logged = 0
    for i, (prompt, weight) in enumerate(COMPLEX_PROMPTS):
        if i > 0:
            time.sleep(INTER_CALL_DELAY_SECONDS)
        messages = [SYSTEM_PROMPT, {"role": "user", "content": prompt}]
        # Calls --provider DIRECTLY, bypassing every tier/chain/bandit —
        # confirmed live this needed fixing TWICE: TierClient's own
        # `preferred` is only a hint that still degrades through the
        # tier's normal fallback chain the moment the named provider
        # errors, times out, or the bandit's live-learned order tries a
        # different rung first — silently landing on groq, the
        # rule-based floor, or even the local SLM itself with no error
        # ever surfacing. None of those are acceptable substitutes here:
        # this script's whole point is a real answer from the SPECIFIC
        # named strong model to distill from.
        # A longer timeout than the provider's own interactive-latency
        # default (30s) — confirmed live: these prompts genuinely ask
        # for long, detailed answers, and most calls hit "the read
        # operation timed out" at 30s before this was added.
        result = call_provider_directly(args.provider, messages, timeout=120.0)
        if result.error or not result.text:
            print(f"SKIP ({args.provider!r}, error={result.error}): {prompt[:70]}...",
                 file=sys.stderr)
            continue
        log_ground_truth_turn(messages=messages, reply=result.text, provider=result.provider,
                              domain=args.domain, training_weight=weight)
        logged += 1
        print(f"Logged (weight={weight}, via {result.provider}): {prompt[:70]}...")

    print(f"\n{logged}/{len(COMPLEX_PROMPTS)} complex examples logged"
         f"{f' under domain={args.domain}' if args.domain else ''} — "
         f"picked up by tonight's nightly fine-tune the same as any other "
         f"ground-truth example.")


if __name__ == "__main__":
    main()
