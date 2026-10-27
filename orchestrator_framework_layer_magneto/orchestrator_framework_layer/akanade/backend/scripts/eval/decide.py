"""Decision report (§F) — ranks candidates by weighted score adjusted for
cost and latency, and names the top 1-2. Deliberately stops there: actual
hosting (vLLM/TGI deployment) and OSS-license verification for commercial
self-hosting are real infra/legal decisions this script can't make for
you — it prints them as a manual checklist instead of pretending to
automate them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# Small penalties, not knockouts — a slightly slower/costlier candidate
# with a meaningfully better score should still win. Tune per how much
# your deployment actually cares about latency/cost vs quality.
COST_PENALTY_PER_DOLLAR = 0.02   # per $/1k-requests
LATENCY_PENALTY_PER_SEC = 0.01   # per second of p95 latency


def rank(report: dict) -> list[tuple[str, float, dict]]:
    scored = []
    for candidate_id, r in report.items():
        adjusted = (r["weighted_score"]
                   - r["cost_per_1k_requests_usd"] * COST_PENALTY_PER_DOLLAR
                   - r["latency_p95_s"] * LATENCY_PENALTY_PER_SEC)
        scored.append((candidate_id, round(adjusted, 3), r))
    return sorted(scored, key=lambda t: -t[1])


def render(report: dict, candidates_meta: dict[str, dict], top_n: int) -> str:
    ranked = rank(report)
    lines = ["# Candidate decision report\n",
            "| Rank | Candidate | Hosting | Adjusted score | Raw weighted | $/1k req | p95 latency |",
            "|---|---|---|---|---|---|---|"]
    for i, (candidate_id, adjusted, r) in enumerate(ranked, 1):
        hosting = candidates_meta.get(candidate_id, {}).get("hosting", "?")
        lines.append(f"| {i} | {candidate_id} | {hosting} | {adjusted} | {r['weighted_score']} | "
                     f"${r['cost_per_1k_requests_usd']} | {r['latency_p95_s']}s |")

    lines.append(f"\n## Top {top_n}\n")
    for candidate_id, adjusted, r in ranked[:top_n]:
        meta = candidates_meta.get(candidate_id, {})
        hosting = meta.get("hosting", "unknown")
        hosting_note = ("runs on your own hardware, free, already under your control" if hosting == "local"
                        else "runs on OpenRouter/upstream cloud GPUs today — still needs real "
                             "self-hosting (vLLM/TGI) before it's actually yours, see checklist below")
        lines.append(f"- **{candidate_id}** (adjusted score {adjusted}) — hosting: **{hosting}** "
                     f"({hosting_note}), license: {meta.get('license', 'unknown')}, "
                     f"~{meta.get('param_b', '?')}B params, "
                     f"~{meta.get('vram_gb', '?')}GB VRAM (fp16 estimate)")

    lines.append("\n## Before self-hosting either of these — manual checklist (§F, not automated)\n")
    lines.append("- [ ] Re-verify the OSS license actually permits YOUR commercial use case "
                "(param count/VRAM here are informational, not a license clearance — see the "
                "notes field in eval/candidates.yaml for any known restrictions, e.g. Llama's "
                "MAU threshold)")
    lines.append("- [ ] Confirm target GPU/VRAM budget matches the vram_gb estimate above "
                "(fp16 estimate; a quantized AWQ/GGUF/FP8 build will need meaningfully less)")
    lines.append("- [ ] Decide vLLM vs TGI for actual serving (Ollama is fine for this "
                "evaluation and for dev, not recommended for production throughput)")
    lines.append("- [ ] If any llm_judge scores above are mock-judged (see report.md), "
                "re-run with a real judge wired in (judge.py) before trusting this ranking "
                "for a real decision")
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    eval_dir = here.parent.parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=eval_dir / "results" / "report.json")
    parser.add_argument("--candidates", type=Path, default=eval_dir / "candidates.yaml")
    parser.add_argument("--out", type=Path, default=eval_dir / "results" / "decision.md")
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    candidates_meta = {c["id"]: c for c in yaml.safe_load(args.candidates.open("r", encoding="utf-8"))}
    text = render(report, candidates_meta, args.top_n)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
