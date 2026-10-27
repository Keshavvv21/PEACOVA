"""Candidate-selection pipeline — Orchestration + Evaluation layers (§C/§D).

Runs every test-asset item (eval/test_assets/*.yaml) against every
candidate model (eval/candidates.yaml), scores deterministic items
directly (exact/regex/schema), and scores llm_judge items via a
pluggable judge (judge.py — stubbed by default, see that file's
docstring for wiring in a real Claude referee). Writes one JSON line per
(candidate, item) to --out, which aggregate.py and decide.py read next.

Bias guard (§D footnote): judge.judge_score() is only ever passed the
answer text, the gold answer, and the rubric — never the candidate's
model name or id, so the judge can't be swayed by knowing which model
produced an answer. There's no pairwise A/B comparison here (each item
is scored independently against its own gold+rubric, not against other
candidates' answers), so "randomized A/B labels" isn't literally
applicable yet — if a future pairwise-comparison mode is added, apply
the same blind-identity principle plus randomizing which slot (A vs B)
a given candidate's answer lands in per call.

Deliberately standalone (own venv — see scripts/eval_requirements.txt),
same reasoning as scripts/finetune_local_slm.py: this is an occasional
batch tool, not something the always-running chatbot image needs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
import yaml

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional, schema items degrade gracefully
    jsonschema = None

from .judge import judge_score

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_assets(test_assets_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(test_assets_dir.glob("*.yaml")):
        items.extend(load_yaml(path))
    return items


def fetch_openrouter_pricing(api_key: str) -> dict[str, dict]:
    """Live per-token pricing from OpenRouter's own model catalog — never
    hardcoded, since prices change without this repo being touched."""
    try:
        resp = requests.get(OPENROUTER_MODELS_URL,
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
        resp.raise_for_status()
        pricing = {}
        for m in resp.json().get("data", []):
            p = m.get("pricing", {})
            pricing[m["id"]] = {
                "prompt": float(p.get("prompt", 0) or 0),
                "completion": float(p.get("completion", 0) or 0),
            }
        return pricing
    except (requests.RequestException, ValueError, KeyError):
        return {}


def call_candidate(candidate: dict, messages: list[dict], *, api_key: str,
                    pricing: dict[str, dict]) -> dict:
    """Returns {"answer": str, "latency_s": float, "cost_usd": float,
    "error": str|None}. Never raises — a single dead candidate must not
    kill the whole run; its items just score 0 with the error recorded."""
    backend = candidate["backend"]
    if backend == "openrouter":
        url = OPENROUTER_CHAT_URL
        headers = {"Authorization": f"Bearer {api_key}"}
    elif backend == "ollama":
        url = candidate["base_url"].rstrip("/") + "/chat/completions"
        headers = {}
    else:
        return {"answer": "", "latency_s": 0.0, "cost_usd": 0.0,
                "error": f"unknown backend {backend!r}"}

    start = time.monotonic()
    try:
        resp = requests.post(url, headers=headers, timeout=180, json={
            "model": candidate["model"], "messages": messages, "temperature": 0,
        })
        resp.raise_for_status()
        body = resp.json()
        answer = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        rates = pricing.get(candidate["model"], {"prompt": 0.0, "completion": 0.0})
        cost = (usage.get("prompt_tokens", 0) * rates["prompt"]
                + usage.get("completion_tokens", 0) * rates["completion"])
        return {"answer": answer, "latency_s": time.monotonic() - start,
               "cost_usd": cost, "error": None}
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        return {"answer": "", "latency_s": time.monotonic() - start,
               "cost_usd": 0.0, "error": str(exc)}


def score_deterministic(item: dict, answer: str) -> float:
    match = item["match"]
    if match == "exact":
        return 1.0 if answer.strip().lower() == item["gold_answer"].strip().lower() else 0.0
    if match == "regex":
        return 1.0 if re.search(item["gold_pattern"], answer) else 0.0
    if match == "schema":
        text = answer.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1] if "\n" in text else text.strip("`")
        try:
            parsed = json.loads(text)
        except ValueError:
            return 0.0
        if jsonschema is None:
            return 1.0 if isinstance(parsed, dict) else 0.0
        try:
            jsonschema.validate(parsed, item["gold_schema"])
            return 1.0
        except jsonschema.ValidationError:
            return 0.0
    raise ValueError(f"unknown match type {match!r}")


def score_item(candidate: dict, item: dict, *, api_key: str, pricing: dict[str, dict],
               use_real_judge: bool) -> dict:
    """One (candidate, item) trial — the reusable core both run() and
    regression_gate.py's on-demand two-candidate comparison call into, so
    the scoring dispatch logic (deterministic vs judge, error handling)
    lives in exactly one place."""
    call = call_candidate(candidate, item["messages"], api_key=api_key, pricing=pricing)
    record = {
        "candidate_id": candidate["id"], "item_id": item["id"],
        "category": item["category"], "scoring_type": item["scoring_type"],
        "answer": call["answer"], "latency_s": call["latency_s"],
        "cost_usd": call["cost_usd"], "error": call["error"],
    }
    if call["error"]:
        record["score"] = 0.0
    elif item["scoring_type"] == "deterministic":
        record["score"] = score_deterministic(item, call["answer"])
    else:
        judged = judge_score(answer=call["answer"], gold_answer=item["gold_answer"],
                             rubric=item["rubric"], use_real_judge=use_real_judge)
        record["score"] = judged["score"]
        record["judge_is_mock"] = judged["is_mock"]
        record["needs_human_review"] = True
    return record


def run(*, test_assets_dir: Path, candidates_path: Path, out_path: Path,
        api_key: str, use_real_judge: bool) -> None:
    items = load_test_assets(test_assets_dir)
    candidates = load_yaml(candidates_path)
    pricing = fetch_openrouter_pricing(api_key) if api_key else {}
    if not pricing and any(c["backend"] == "openrouter" for c in candidates):
        print("Warning: couldn't fetch OpenRouter pricing (no/invalid key?) — "
             "cost will be recorded as 0 for those candidates.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for candidate in candidates:
            print(f"== {candidate['id']} ({len(items)} items) ==")
            for item in items:
                record = score_item(candidate, item, api_key=api_key, pricing=pricing,
                                    use_real_judge=use_real_judge)
                out.write(json.dumps(record) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-assets-dir", type=Path,
                        default=here.parent.parent / "eval" / "test_assets")
    parser.add_argument("--candidates", type=Path,
                        default=here.parent.parent / "eval" / "candidates.yaml")
    parser.add_argument("--out", type=Path, default=here.parent.parent / "eval" / "results" / "raw.jsonl")
    parser.add_argument("--openrouter-api-key", default="")
    parser.add_argument("--real-judge", action="store_true",
                        help="Call the real Claude API as referee instead of the mock heuristic — "
                             "not implemented yet, see judge.py.")
    args = parser.parse_args()
    run(test_assets_dir=args.test_assets_dir, candidates_path=args.candidates,
        out_path=args.out, api_key=args.openrouter_api_key, use_real_judge=args.real_judge)


if __name__ == "__main__":
    main()
