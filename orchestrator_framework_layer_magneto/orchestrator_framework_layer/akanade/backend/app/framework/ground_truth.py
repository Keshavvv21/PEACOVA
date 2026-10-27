""""Create ground truth" — the enterprise-pipeline stage between
collecting real prompts your organization cares about and actually
fine-tuning on them. Distinct from every other source that feeds
scripts/finetune_local_slm.py's training corpus (organic chat turns,
Hybrid escalations, document-derived Q&A): those are all INCIDENTAL —
whatever happened to get asked, or whatever an ingested document
happened to produce. This module is DELIBERATE — you hand it the
prompts you specifically want the model to get right, and it generates
a real, grounded, reviewable answer for each one before that pair ever
becomes training data.

Grounding is optional but recommended: pass a `store_name` (any
already-registered VectorStoreAdapter — an ingested CSV/PDF/TXT/MD
corpus, or the PM domain's own "pm-index") and each prompt's answer is
generated from real retrieved excerpts via the SAME
rag_optimizer.retrieve_with_optimizer() pipeline document_pipeline.py's
summarize_document()/suggest_prompts() already use, not the model
answering from parametric memory alone. Every result is returned to
the caller for review BEFORE it's trusted, exactly like a summary or
suggested prompts — logging a bad "ground truth" answer as training
data would be worse than not logging anything at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..agent import finetune_log
from ..llm.router import get_llm_client
from .rag_optimizer import rag_optimizer, retrieve_with_optimizer

_GROUND_TRUTH_SYSTEM = {"role": "system", "content": (
    "You produce a single, precise, verified reference answer to a real "
    "enterprise question — this answer will be used as TRAINING DATA for "
    "a smaller model, so accuracy matters more than style. If excerpts are "
    "provided, answer ONLY from them and say so plainly if they don't "
    "actually cover the question; never invent a fact not present in the "
    "excerpts. If no excerpts are provided, answer from general knowledge "
    "but be direct and unhedged — no filler, no 'as an AI'.")}


@dataclass
class GroundTruthResult:
    prompt: str
    answer: str
    provider: str | None
    grounded: bool
    sources_used: list[dict] = field(default_factory=list)
    ok: bool = True


def create_ground_truth(prompts: list[str], *, store_name: str | None = None,
                        k: int = 6, max_context_chars: int = 4000,
                        log_for_training: bool = True) -> list[GroundTruthResult]:
    """For each prompt: optionally retrieve grounding context from
    `store_name`, ask the LLM router's "bulk" tier for one precise
    answer, and — if `log_for_training` (default) and the call actually
    succeeded — log it via finetune_log.log_ground_truth_turn() so it's
    eligible for tonight's curated fine-tune corpus (data_engine.py
    still scrubs/dedups it exactly like every other source). Returns
    every result regardless of success, so a caller can review what
    actually got logged vs. what failed."""
    results: list[GroundTruthResult] = []
    for prompt in prompts:
        context, sources = "", []
        if store_name:
            retrieval = retrieve_with_optimizer(
                prompt, store_names=[store_name], k=k,
                max_context_chars=max_context_chars, refine_query=False)
            context, sources = retrieval.context, retrieval.sources_used
            if retrieval.selection is not None:
                rag_optimizer.record_outcome(retrieval.selection, success=bool(context),
                                             n_hits=retrieval.n_hits)

        user_content = (f"Excerpts:\n\n{context}\n\nQuestion: {prompt}" if context
                        else f"Question: {prompt}")
        messages = [_GROUND_TRUTH_SYSTEM, {"role": "user", "content": user_content}]
        client = get_llm_client("bulk")
        res = client.chat("create_ground_truth", messages)
        ok = bool(res.text) and not res.error

        if ok and log_for_training:
            # The LOGGED messages deliberately do NOT include this
            # module's own "produce verified training data" system
            # prompt above — that instruction is specific to generating
            # ground truth, never present at real inference time, and
            # training on it would teach the model to expect an
            # instruction it will never actually see in production. If
            # grounded, the context IS included as a system message,
            # matching real usage (rag_optimizer.py injects retrieved
            # context as an extra system message before perceive_intent
            # sees the question) — so the training format matches the
            # inference format it's meant to improve.
            logged_messages = ([{"role": "system", "content": context}] if context else []) + \
                [{"role": "user", "content": prompt}]
            finetune_log.log_ground_truth_turn(
                messages=logged_messages, reply=res.text, provider=res.provider)

        results.append(GroundTruthResult(
            prompt=prompt, answer=res.text or "Couldn't generate an answer.",
            provider=res.provider, grounded=bool(context), sources_used=sources, ok=ok))
    return results
