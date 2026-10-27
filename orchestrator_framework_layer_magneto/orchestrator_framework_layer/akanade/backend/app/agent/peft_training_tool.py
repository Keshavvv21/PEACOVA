""""Train the local model on this" — a chat-reachable trigger into the
same nightly PEFT pipeline the PEFT Analyzer's scraper already feeds
(see peft_analyzer.py, app/agent/finetune_log.py). A user pastes a URL
or some text and says "PEFT train this for <domain>" (or just "train
this" — native_agent.py's resolve_arguments asks which domain if it
isn't named, the same clarify()/resume_clarification round trip every
other ambiguous PM request already uses); this tool fetches the URL (if
given), logs the content as a real ground-truth training example for
THAT domain's own log file, and reports back what will happen — not
"trained," since actual training only runs on the nightly schedule
(deploy/nightly_finetune.sh), never synchronously inside a chat reply.

Same "translation layer only" shape as trading_tools.py — no PM-entity
resolution needed for its own args, only the domain-confirmation step
lives in native_agent.py's resolve_arguments.
"""
from __future__ import annotations

import re

import httpx

from .finetune_log import log_ground_truth_turn

_REQUEST_TIMEOUT_S = 10.0
_MAX_CONTENT_CHARS = 8000  # a training example, not a whole book — same bound style as agent/tools.py
_TAG_RE = re.compile(r"<[^>]+>")


class TrainingRequestError(Exception):
    """Human-readable — a chat reply can show this as-is."""


def _strip_html(text: str) -> str:
    """A deliberately crude tag-stripper, not a real HTML parser — good
    enough to turn a fetched page into plain-ish text for a training
    example; this is NOT a rendering/security-sensitive path (the
    result only ever becomes model training data, never rendered as
    HTML anywhere)."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def fetch_url_text(url: str) -> str:
    try:
        r = httpx.get(url, timeout=_REQUEST_TIMEOUT_S, follow_redirects=True,
                      headers={"User-Agent": "Algo-Akanade-PEFT-Training-Request/1.0"})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise TrainingRequestError(f"Could not fetch {url!r}: {e}") from e
    text = _strip_html(r.text)
    if not text:
        raise TrainingRequestError(f"Fetched {url!r} but found no readable text in it.")
    return text[:_MAX_CONTENT_CHARS]


def request_training(*, domain: str, topic: str, content: str | None = None,
                     url: str | None = None) -> dict:
    """`content` and `url` are mutually exclusive-ish — url is fetched
    if content isn't already provided. Logs ONE ground-truth example
    (`domain`-specific log file — see finetune_log.py's log_ground_
    truth_turn) and returns a plain status dict; the actual LoRA
    fine-tune for this domain still only runs on tonight's schedule."""
    if content:
        text = content[:_MAX_CONTENT_CHARS]
        source = "pasted text"
    elif url:
        text = fetch_url_text(url)
        source = url
    else:
        raise TrainingRequestError("Need either some text or a URL to train on.")

    log_ground_truth_turn(
        messages=[{"role": "user", "content": f"What is {topic}?"}],
        reply=text,
        provider="peft-training-request",
        domain=domain,
    )
    return {
        "domain": domain,
        "topic": topic,
        "source": source,
        "chars_logged": len(text),
    }
