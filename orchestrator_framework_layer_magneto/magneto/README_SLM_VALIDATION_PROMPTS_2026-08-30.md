# SLM validation prompts — Trading, Coding, PM (2026-08-30)

A single, consistent prompt list to run against all three domain SLMs
before and after a nightly fine-tune, so a score change is attributable
to the training run and not to prompt drift. Pairs with
[`README_UNIFIED_AGENT_TESTING.md`](README_UNIFIED_AGENT_TESTING.md)
(cross-platform coverage) — this document is about cross-*version*
coverage: same prompts, base model vs. the newly fine-tuned candidate,
via the regression gate (`scripts/eval/regression_gate.py`) or by hand.

## How to run these

Each prompt is a single `POST /api/chat/message` call
(`{"message": "<prompt>", "session_id": "slm-validation-1"}`), same as
any other chat turn — see `README_EXTERNAL_API.md`. To compare a
specific model version rather than whatever auto-routing picks, pin it
with `"provider": "<tag>"` using a tag from `GET /api/chat/providers`.

## Trading SLM

Sourced from the real, scored NL+BT chatbot ground-truth set ingested
into the `trading` domain's fine-tune log on 2026-08-30 (25 examples,
each reward=1.0) — these are exactly what tonight's run trained on, so
they're the most direct "did it learn this" check, plus a few prompts
from `README_UNIFIED_AGENT_TESTING.md` for the live backtest tool path
specifically.

**Query refinement / structuring** (expect a refined query or
structured JSON, not an answer — these agents must never execute the
query):
1. `Show me stocks with good growth and momentum` — expect only
   casual phrasing removed, "good" preserved (not upgraded to "strong").
2. `top growth stocks` — expect ambiguity preserved and clarifying
   questions about metric/timeframe/threshold, not an assumed answer.
3. `Filter top 10 stocks where Net Profit Margin > 15% every year from
   2016 to 2021` — expect zero required clarifications (query is
   already unambiguous).

**Query splitting (NLQ vs. backtest)**:
4. `Screen stocks with ROCE > 25% and invest equally` — expect both
   `flag_nlq` and `flag_bt` true (hybrid).
5. `Backtest buy and hold on INFY.NS from 2020 to 2024` — expect
   `flag_nlq: false`, `flag_bt: true` (pure backtest, no screening).

**Trading-strategy interpretation**:
6. `Buy TCS when RSI below 30, sell when RSI above 70, from 2023 to
   2025` — expect `strategy: "RSI"` with `crosses_above`/`crosses_below`
   encoding, not plain `<`/`>` inequalities.
7. `Backtest a buy and hold strategy on RELIANCE.NS from 2020 to 2024`
   — expect `strategy: "BuyAndHold"`, empty buy/sell conditions.

**Live backtest tool (the actual deployed Trading Agent, not the
query-refiner prompts above)**:
8. `Backtest AAPL for the last 10 years` — expect CAGR, buy-and-hold
   CAGR, annualized volatility, Sharpe, max drawdown, and an inline
   chart image.
9. `Run a 10/30 day moving average crossover on MSFT`

## Coding SLM

Sourced from real, self-contained coding prompts ingested into the
`coding` domain's fine-tune log on 2026-08-30 (8 examples, each
reward=1.0) — chosen specifically because they're answerable as a
direct code snippet with no repo/file context needed, unlike most of
the Aksha benchmark catalog's file-dependent prompts.

1. `Write a Python function that resolves GPU vs. CPU device with
   fallback.`
2. `Write a bash script to rotate and gzip logs older than 30 days.`
3. `Write a Dockerfile for a Python service needing OpenCV and
   CUDA-optional Torch.`
4. `pandas: load a CSV and print rows, columns and percent missing per
   column.`
5. `sklearn: return precision, recall and F1 without zero-division
   errors.`
6. `sklearn: 80/20 stratified train/test split with a fixed seed.`
7. `FastAPI: POST /predict with validation rejecting negative
   amounts.`
8. `Python: return TN, FP, FN and TP from y_true / y_pred as a
   dictionary.`

**What "pass" looks like**: a real code block using the right library
call (not a hallucinated function signature or a bare tool-call-shaped
JSON blob — this is a real failure mode observed live on 2026-08-30
from the pre-fine-tune 1.5B coding SLM on prompt 1 above, which is
exactly why this domain has ground truth to train on now).

## PM SLM

See `README_UNIFIED_AGENT_TESTING.md`'s "PM Tool" section — unchanged,
reused here rather than duplicated. No PM-domain ground truth was
added in this round (no PM-specific source documents were provided);
its regression gate still runs against whatever real chat traffic has
accumulated.

## Honesty note on this round's ground truth

- **Trading** (25 examples): the *exact* text from a real,
  human-scored ground-truth spreadsheet (query-refiner/structuring/
  interpreter/translator prompts for an NL+BT financial chatbot,
  correctness-scored 4.5–5/5) — transcribed in full via a script, not
  hand-typed, specifically to avoid introducing transcription drift
  into training data.
- **Coding** (8 examples): the deployed coding agent was tried live
  against these prompts first; it produced usable output for some but
  a malformed non-answer for others (see above). Rather than log that
  mixed-quality output as "ground truth," these 8 replies were written
  fresh, by hand, to the same bar as the trading set's reference
  answers — real, correct, verifiable code, not fabricated to look
  plausible.
- Both are intentionally small (25 and 8 examples) — enough to clear
  `finetune_local_slm.py`'s `--min-examples 8` floor for tonight's run,
  not a claim of comprehensive domain coverage. Treat tonight's result
  as "does the pipeline learn from real ground truth at all," not "is
  the domain SLM now broadly competent."
