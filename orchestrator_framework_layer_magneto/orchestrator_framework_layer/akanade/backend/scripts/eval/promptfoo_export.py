"""Optional: exports eval/test_assets + eval/candidates.yaml as a real
promptfoo config (§C names promptfoo explicitly) — for teams who already
have the `promptfoo` CLI (Node) and want to run this same test set through
it directly, alongside/instead of harness.py.

harness.py is the validated, primary path (pure Python, no Node
dependency, smoke-tested for real). This exporter follows promptfoo's
documented config schema but was NOT run against a real `promptfoo` CLI
in this environment (no Node.js available here) — validate the generated
file with `promptfoo validate` once you have the CLI, before trusting it
for a real run.

    pip install pyyaml   # already in eval_requirements.txt
    python -m scripts.eval.promptfoo_export
    npm install -g promptfoo
    OPENROUTER_API_KEY=... promptfoo eval -c eval/promptfooconfig.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def build_config(items: list[dict], candidates: list[dict]) -> dict:
    providers = []
    for c in candidates:
        if c["backend"] == "openrouter":
            providers.append(f"openrouter:{c['model']}")
        else:  # ollama — promptfoo's openai-compatible provider works against any /v1 endpoint
            providers.append({
                "id": f"openai:chat:{c['model']}",
                "config": {"apiBaseUrl": c["base_url"]},
            })

    tests = []
    for item in items:
        asserts = []
        if item["scoring_type"] == "deterministic":
            if item["match"] == "exact":
                asserts.append({"type": "equals", "value": item["gold_answer"]})
            elif item["match"] == "regex":
                asserts.append({"type": "regex", "value": item["gold_pattern"]})
            elif item["match"] == "schema":
                asserts.append({"type": "is-json", "value": item["gold_schema"]})
        else:
            asserts.append({
                "type": "llm-rubric",
                "value": f"{item['rubric']}\n\nGold answer for reference: {item['gold_answer']}",
            })
        tests.append({
            "description": item["id"],
            "vars": {"system": item["messages"][0]["content"],
                    "user": item["messages"][1]["content"]},
            "assert": asserts,
        })

    return {
        "description": "ALGO AKANADE candidate-selection pipeline (auto-exported from "
                       "eval/test_assets + eval/candidates.yaml — see promptfoo_export.py)",
        "providers": providers,
        "prompts": ["[{{\"role\": \"system\", \"content\": {{system | dump}} }}, "
                   "{{\"role\": \"user\", \"content\": {{user | dump}} }}]"],
        "tests": tests,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    eval_dir = here.parent.parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-assets-dir", type=Path, default=eval_dir / "test_assets")
    parser.add_argument("--candidates", type=Path, default=eval_dir / "candidates.yaml")
    parser.add_argument("--out", type=Path, default=eval_dir / "promptfooconfig.yaml")
    args = parser.parse_args()

    items = []
    for path in sorted(args.test_assets_dir.glob("*.yaml")):
        items.extend(yaml.safe_load(path.open("r", encoding="utf-8")))
    candidates = yaml.safe_load(args.candidates.open("r", encoding="utf-8"))

    config = build_config(items, candidates)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.out} ({len(items)} tests, {len(candidates)} providers) — "
         f"unvalidated against the real promptfoo CLI, see this module's docstring.")


if __name__ == "__main__":
    main()
