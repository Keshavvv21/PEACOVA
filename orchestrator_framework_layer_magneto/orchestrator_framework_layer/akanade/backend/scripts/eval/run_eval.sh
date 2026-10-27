#!/usr/bin/env bash
# Full candidate-selection sweep (§C-F): every candidate in
# eval/candidates.yaml (local Ollama baselines + OpenRouter candidates)
# against every item in eval/test_assets/*.yaml. This is the "same
# harness" that regression_gate.py also reuses nightly for the cheap
# two-candidate PEFT promotion check (§G) — this script is the
# heavier, on-demand, full-roster version a human runs when actually
# deciding what to self-host next, not something to cron nightly as-is.
#
# Needs OPENROUTER_API_KEY (see deploy/eval.env.example) to query the
# OpenRouter candidates; local-only Ollama candidates still get scored
# without one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # .../scripts/eval
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"                    # .../ (holds scripts/ and eval/)
# Deploy layout: .../finetune/scripts/eval/<this file>, .../finetune/eval.env
# (sibling of scripts/ and eval/) — same convention as finetune.env.
ENV_FILE="${EVAL_ENV_FILE:-$BACKEND_DIR/eval.env}"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

cd "$BACKEND_DIR"  # so `python -m scripts.eval.*` resolves
# Same interpreter convention as convert_and_register_ollama_model.sh's
# EVAL_PYTHON — any Python with eval_requirements.txt installed; the
# scripts/finetune_local_slm.py venv works fine, its deps are tiny.
EVAL_PYTHON="${EVAL_PYTHON:-python3}"

echo "== Running the harness against every candidate =="
"$EVAL_PYTHON" -m scripts.eval.harness --openrouter-api-key "${OPENROUTER_API_KEY:-}"

echo "== Sampling for human spot-check (§D third lane — fill in results/human_review.csv by hand) =="
"$EVAL_PYTHON" -m scripts.eval.human_sample

echo "== Aggregating (§E) =="
"$EVAL_PYTHON" -m scripts.eval.aggregate

echo "== Deciding (§F) =="
"$EVAL_PYTHON" -m scripts.eval.decide

echo
echo "Review results/human_review.csv, fill in human_score for a real run, then re-run"
echo "'scripts.eval.aggregate' (it picks up human overrides automatically) before trusting"
echo "results/decision.md for an actual self-hosting decision."
