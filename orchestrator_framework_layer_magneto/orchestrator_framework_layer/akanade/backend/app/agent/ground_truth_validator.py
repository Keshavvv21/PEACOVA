"""Ground Truth Validation — a daily check, run right after a nightly
PEFT candidate is registered (deploy/nightly_finetune.sh, right after
convert_and_register_ollama_model.sh returns), asking: for prompts drawn
from TONIGHT's own new training data, does the freshly fine-tuned
candidate's answer still roughly agree with what a strong hosted model
(Groq) says about the exact same prompt?

This is a DIFFERENT signal from scripts/eval/regression_gate.py's
old-vs-new comparison against static, hand-written gold answers
(eval/test_assets/*.yaml) — that catches "did tonight's fine-tune make
things worse on our fixed benchmark." This instead catches "does the
model still broadly agree with an independent, much larger reference
model on the very data it was just trained on" — a cheap sanity signal
a static benchmark can't give, since it's generated fresh every night
from whatever domain-specific content that night's fine-tune actually
used.

Runs inside the always-on chatbot image (needs app.llm.router's
call_provider_directly and app.agent.finetune_log's own JSONL log —
same reasoning app.agent.peft_analyzer already documents for living
here rather than in scripts/'s separate, heavy-dependency venv):

    docker exec ak-chatbot python -m app.agent.ground_truth_validator --domain <domain>

`match_score` is a cheap embedding cosine similarity
(app.retrieval.vectorstore.HashEmbedder — the SAME hashed-bag-of-words
technique scripts/data_engine.py's dedup and framework/task_memory.py's
routing-relevance already use), never a second LLM-judge call — see
app.models.GroundTruthRunItem's docstring for the same reasoning.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ..llm.domain_router import DOMAIN_PROVIDER_PREFIX
from ..llm.router import call_provider_directly
from ..models import GroundTruthRunIn, GroundTruthRunItem
from ..retrieval.vectorstore import HashEmbedder
from ..routers.admin_ground_truth import create_run
from .finetune_log import _log_path

_GROQ_PROVIDER = "groq"
_embedder = HashEmbedder()


def _candidate_provider_name(domain: str | None) -> str:
    return f"{DOMAIN_PROVIDER_PREFIX}{domain}" if domain else "algo-akanade-slm"


def _todays_prompts(domain: str | None, *, sample_size: int) -> list[str]:
    """The most recent distinct user prompts logged TODAY (UTC) for this
    domain — real content this exact night's fine-tune actually trained
    on, not a static/hardcoded validation set. Oldest-write-order file,
    read newest-first (reversed) so a capped sample favors the most
    recently logged prompts over whatever was first written at midnight."""
    path = _log_path(domain)
    if not path.exists():
        return []
    today = datetime.now(timezone.utc).date().isoformat()
    seen: set[str] = set()
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not record.get("timestamp", "").startswith(today):
            continue
        user_turns = [m.get("content", "") for m in record.get("messages", []) if m.get("role") == "user"]
        prompt = (user_turns[-1] if user_turns else "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) >= sample_size:
            break
    return prompts


def _similarity(a: str, b: str) -> float:
    """Cosine similarity between two answers' hashed-bag-of-words
    embeddings — HashEmbedder's output is already unit-normalized (see
    its own docstring), so this is just the dot product. 0.0 for an
    empty answer on either side rather than a division/domain error."""
    if not a.strip() or not b.strip():
        return 0.0
    va, vb = _embedder.embed(a), _embedder.embed(b)
    return round(max(0.0, sum(x * y for x, y in zip(va, vb))), 4)


def run(*, domain: str | None = None, sample_size: int = 8) -> dict:
    """Never raises — a bad night's validation run degrades to a
    "skipped" summary, never a crashed nightly job (the fine-tune +
    registration this runs after must already have succeeded for this
    to even be reached; a validation-script bug shouldn't retroactively
    un-register a candidate that already passed its real regression
    gate)."""
    provider_name = _candidate_provider_name(domain)
    prompts = _todays_prompts(domain, sample_size=sample_size)
    if not prompts:
        return {"skipped": "no new prompts logged today", "domain": domain, "provider": provider_name}

    items: list[GroundTruthRunItem] = []
    for prompt in prompts:
        groq_result = call_provider_directly(_GROQ_PROVIDER, [{"role": "user", "content": prompt}])
        model_result = call_provider_directly(provider_name, [{"role": "user", "content": prompt}])
        if groq_result.error or model_result.error:
            continue  # a failed call for this one prompt — skip it, never fabricate a score
        items.append(GroundTruthRunItem(
            prompt=prompt, groq_answer=groq_result.text, model_answer=model_result.text,
            match_score=_similarity(groq_result.text, model_result.text)))

    if not items:
        return {"skipped": "no successful Groq/candidate comparisons", "domain": domain,
               "provider": provider_name}

    result = create_run(GroundTruthRunIn(domain=domain, provider=provider_name, items=items))
    return {"run_id": result.id, "avg_match_score": result.avg_match_score,
           "n_items": len(items), "domain": domain, "provider": provider_name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=None, help="Empty/omitted = the general (non-domain) model.")
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(domain=args.domain, sample_size=args.sample_size), indent=2))
