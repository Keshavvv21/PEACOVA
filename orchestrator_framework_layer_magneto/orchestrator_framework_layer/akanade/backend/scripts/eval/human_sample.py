"""Human spot-check (§D, third scoring lane) — samples ~10-15% of the
llm_judge-scored items, weighted toward reasoning/planning/complex_reasoning
(the categories where a human's judgment matters most), and writes a CSV a
person fills in by hand. merge_human_scores() reads it back and overrides
the mock/real judge's score wherever a human filled one in, so
aggregate.py always prefers the human number when one exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

WEIGHTED_CATEGORIES = {"reasoning", "planning", "complex_reasoning"}


def sample(records: list[dict], *, frac: float, weight_multiplier: float,
           rng) -> list[dict]:
    """Weighted sampling without replacement — items in WEIGHTED_CATEGORIES
    are `weight_multiplier`x more likely to be picked than everything else.
    rng is an injected random.Random for deterministic tests."""
    judged = [r for r in records if r.get("needs_human_review")]
    n = max(1, round(len(judged) * frac)) if judged else 0
    weights = [weight_multiplier if r["category"] in WEIGHTED_CATEGORIES else 1.0
              for r in judged]
    chosen: list[dict] = []
    pool = list(zip(judged, weights))
    for _ in range(min(n, len(pool))):
        total = sum(w for _, w in pool)
        pick = rng.uniform(0, total)
        acc = 0.0
        for i, (_, w) in enumerate(pool):
            acc += w
            if acc >= pick:
                chosen.append(pool.pop(i)[0])
                break
    return chosen


def write_review_csv(sampled: list[dict], items_by_id: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "item_id", "category", "prompt", "gold_answer",
                         "rubric", "model_answer", "mock_score", "human_score"])
        for r in sampled:
            item = items_by_id[r["item_id"]]
            prompt = " | ".join(m["content"] for m in item["messages"] if m["role"] == "user")
            writer.writerow([r["candidate_id"], r["item_id"], r["category"], prompt,
                             item["gold_answer"], item["rubric"], r["answer"],
                             r["score"], ""])


def merge_human_scores(records: list[dict], review_csv: Path) -> list[dict]:
    if not review_csv.exists():
        return records
    overrides: dict[tuple[str, str], float] = {}
    with review_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = (row.get("human_score") or "").strip()
            if human:
                overrides[(row["candidate_id"], row["item_id"])] = float(human)
    for r in records:
        key = (r["candidate_id"], r["item_id"])
        if key in overrides:
            r["score"] = overrides[key]
            r["scored_by_human"] = True
    return records


def main() -> None:
    import random

    here = Path(__file__).resolve().parent
    eval_dir = here.parent.parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=eval_dir / "results" / "raw.jsonl")
    parser.add_argument("--test-assets-dir", type=Path, default=eval_dir / "test_assets")
    parser.add_argument("--out", type=Path, default=eval_dir / "results" / "human_review.csv")
    parser.add_argument("--frac", type=float, default=0.125, help="~10-15%% target, default 12.5%%")
    parser.add_argument("--weight-multiplier", type=float, default=3.0)
    args = parser.parse_args()

    import yaml
    items_by_id = {}
    for path in sorted(args.test_assets_dir.glob("*.yaml")):
        for item in yaml.safe_load(path.open("r", encoding="utf-8")):
            items_by_id[item["id"]] = item

    records = [json.loads(line) for line in args.results.open("r", encoding="utf-8")]
    sampled = sample(records, frac=args.frac, weight_multiplier=args.weight_multiplier,
                     rng=random.Random())
    write_review_csv(sampled, items_by_id, args.out)
    print(f"Sampled {len(sampled)}/{sum(1 for r in records if r.get('needs_human_review'))} "
         f"llm_judge-scored items -> {args.out}. Fill in the human_score column, then re-run "
         f"aggregate.py to have it override the mock/real judge score.")


if __name__ == "__main__":
    main()
