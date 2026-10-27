"""Regression Loop (§G) as a promotion gate — the concrete tie between
this eval pipeline and the nightly PEFT fine-tune (scripts/finetune_local_slm.py
+ scripts/convert_and_register_ollama_model.sh). Reruns the SAME harness
and test-asset set against last night's PEFT-tuned Ollama tag vs the
current base tag; only a genuine improvement (within --tolerance) gets
promoted (registered with the live chatbot). A worse or unchanged
fine-tune still gets created in Ollama for manual inspection, but this
script's non-zero exit tells convert_and_register_ollama_model.sh to
skip registering it — so a bad night's training run can never
regress the model actually serving.

Exit 0 = promote, exit 1 = ran fine but the new tag genuinely scored
worse (real regression, don't promote), exit 2 = couldn't run at all
(missing deps/config — the caller should skip the gate and warn, not
treat this the same as a real regression). Both candidates are local
Ollama tags (no OpenRouter call, no cost) — this gate runs every night,
it needs to be cheap and fast, not another full candidate-selection sweep.

--update-peft-bandit closes the loop finetune_local_slm.py's
--auto-hyperparams opens: this gate's own new_score - old_score margin
becomes the REWARD for whichever hyperparameter arm tonight's fine-tune
used, so tomorrow's --auto-hyperparams run is a genuinely better-
informed pick, not the same fixed guess every night. Reads the sidecar
--auto-hyperparams wrote (--peft-arm-info) — silently skipped, never a
hard error, if that file doesn't exist (a run without --auto-hyperparams
last night, or the very first night ever).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _update_peft_bandit(arm_info_path: Path, *, margin: float) -> None:
    """Best-effort, never fatal — a missing/corrupt sidecar (no
    --auto-hyperparams run last night, or the very first night ever)
    just means there's nothing to update yet, not a reason to fail
    tonight's actual promotion decision."""
    if not arm_info_path.exists():
        print(f"--update-peft-bandit: no sidecar at {arm_info_path} — nothing to update "
             f"(expected on the first night, or a night without --auto-hyperparams).")
        return
    try:
        info = json.loads(arm_info_path.read_text(encoding="utf-8"))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import peft_hyperparam_optimizer as peft_bandit
        bandit = peft_bandit.PeftHyperparamOptimizer(info["state_path"])
        bandit.record_outcome(info["arm"], info["context"], score_margin=margin)
        print(f"--update-peft-bandit: recorded margin={margin:+.4f} for arm={info['arm']!r} "
             f"— tomorrow's --auto-hyperparams pick reflects tonight's real outcome.")
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        print(f"--update-peft-bandit: couldn't read/apply {arm_info_path} ({exc}) — skipping, "
             f"tonight's promotion decision is unaffected.", file=sys.stderr)


def main() -> None:
    try:
        from .aggregate import aggregate, load_weights
        from .harness import load_test_assets, score_item
    except ImportError as exc:
        print(f"Regression gate couldn't import its own dependencies ({exc}) — "
             f"is eval_requirements.txt installed in this interpreter?", file=sys.stderr)
        sys.exit(2)
    here = Path(__file__).resolve().parent
    eval_dir = here.parent.parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tag", required=True, help="Currently-serving Ollama tag, or a "
                        "descriptive label for the currently-live candidate in --new-base-url "
                        "setups (e.g. SERVE_BACKEND=llama_server, where there's no Ollama tag).")
    parser.add_argument("--new-tag", required=True, help="Tonight's candidate tag to gate")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434/v1",
                        help="Base URL for the OLD/base candidate — and the new one too, unless "
                             "--new-base-url is given (the normal same-server-two-tags Ollama case, "
                             "where one Ollama instance already serves both tags).")
    parser.add_argument("--new-base-url", default=None,
                        help="Base URL for the NEW candidate specifically, when it's served from a "
                             "DIFFERENT endpoint than the base tag — e.g. two separate llama-server "
                             "containers (SERVE_BACKEND=llama_server), which don't share one server "
                             "the way two Ollama tags do. Defaults to --ollama-base-url when omitted.")
    parser.add_argument("--test-assets-dir", type=Path, default=eval_dir / "test_assets")
    parser.add_argument("--weights", type=Path, default=eval_dir / "weights.yaml")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="New tag may score up to this much lower than base and still pass "
                             "(noise tolerance) — default 0.02.")
    parser.add_argument("--update-peft-bandit", action="store_true",
                        help="After scoring, feed this gate's new_score - old_score margin back "
                             "into the PEFT hyperparameter bandit as a reward for whichever arm "
                             "--auto-hyperparams picked last night (see --peft-arm-info). Off by "
                             "default — only meaningful paired with finetune_local_slm.py's "
                             "--auto-hyperparams.")
    parser.add_argument("--peft-arm-info", default="data/finetune/merged/peft_arm_used.json",
                        help="The sidecar finetune_local_slm.py's --auto-hyperparams wrote — "
                             "which arm/context tonight's run actually used.")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="Write the full per-item scoring report (this gate's own "
                             "SlmEvalReportIn shape, app/models.py) as JSON here — "
                             "convert_and_register_ollama_model.sh reads it and POSTs it to "
                             "/api/admin/slm-eval-reports so the Admin dashboard can show "
                             "exactly which prompts tonight's candidate was tested against and "
                             "how it scored on each. Not written at all if scoring never ran "
                             "(missing deps/test-assets, exit 2) — there's nothing to report yet.")
    args = parser.parse_args()

    try:
        items = load_test_assets(args.test_assets_dir)
        if not items:
            raise ValueError(f"no test-asset items found under {args.test_assets_dir}")
        weights = load_weights(args.weights)
    except (OSError, ValueError) as exc:
        print(f"Couldn't load eval config ({exc}) — skipping the gate.", file=sys.stderr)
        sys.exit(2)
    old = {"id": "old", "backend": "ollama", "model": args.base_tag, "base_url": args.ollama_base_url}
    new = {"id": "new", "backend": "ollama", "model": args.new_tag,
           "base_url": args.new_base_url or args.ollama_base_url}

    records = []
    for candidate in (old, new):
        print(f"Scoring {candidate['model']} against {len(items)} regression items...")
        for item in items:
            records.append(score_item(candidate, item, api_key="", pricing={},
                                      use_real_judge=False))

    report = aggregate(records, weights)
    old_score, new_score = report["old"]["weighted_score"], report["new"]["weighted_score"]
    print(f"base ({args.base_tag}): {old_score}   new ({args.new_tag}): {new_score}   "
         f"tolerance: -{args.tolerance}")

    if report["old"]["n_mock_judged"] or report["new"]["n_mock_judged"]:
        print("Note: reasoning/planning/complex_reasoning items were scored by the mock judge "
             "heuristic (judge.py), not a real referee — this gate is directionally useful but "
             "not a substitute for a real judge once one is wired in.")

    if args.update_peft_bandit:
        _update_peft_bandit(Path(args.peft_arm_info), margin=new_score - old_score)

    passed = new_score >= old_score - args.tolerance

    if args.report_out:
        # Only the "new" candidate's own per-item rows — the base tag's
        # scores already exist from whichever earlier night produced
        # ITS report; this is specifically "how did TONIGHT'S candidate
        # do," item by item.
        new_items = [
            {"item_id": r["item_id"], "category": r["category"], "scoring_type": r["scoring_type"],
             "score": r["score"], "error": r.get("error"), "answer": r["answer"]}
            for r in records if r["candidate_id"] == "new"
        ]
        report_doc = {
            "tag": args.new_tag, "base_tag": args.base_tag, "tolerance": args.tolerance,
            "old_score": old_score, "new_score": new_score, "passed": passed,
            "old_by_category": report["old"]["by_category"],
            "new_by_category": report["new"]["by_category"],
            "items": new_items,
        }
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report_doc, indent=2), encoding="utf-8")
        print(f"Wrote the full per-item verification report to {args.report_out}")

    if passed:
        print(f"PASS — {args.new_tag} may be promoted.")
        sys.exit(0)
    print(f"FAIL — {args.new_tag} scores meaningfully worse than {args.base_tag}, not promoting.")
    sys.exit(1)


if __name__ == "__main__":
    main()
