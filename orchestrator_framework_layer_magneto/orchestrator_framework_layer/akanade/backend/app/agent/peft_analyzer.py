"""PEFT Analyzer — scrapes real content into (prompt, verified-answer)
ground-truth pairs so tonight's fine-tune (scripts/finetune_local_slm.py,
via log_ground_truth_turn() below) has something real to learn from even
before organic chat traffic accumulates enough reward-scored turns on
its own. Run once, before the per-domain fine-tune loop, by
deploy/nightly_finetune.sh (`docker exec ak-chatbot python -m
app.agent.peft_analyzer`) — lives inside the always-running chatbot
image so it can reuse log_ground_truth_turn()'s exact write path (same
file deploy/nightly_finetune.sh already `docker cp`s out), never a
second, divergent data format.

Source: Wikipedia's free, keyless REST summary endpoint — the same
domain the chatbot's own "fetch" MCP tool already reaches in production
(confirmed during live testing), so this introduces no new external
dependency risk. Topics come from every enabled finetune-config's own
`scrape_topics` list (admin-configurable, see models.FinetuneConfigIn) —
no topic is ever hardcoded here.

Local storage is capped, not unbounded — same "capped, not unbounded"
convention as CHAT_LOG_STATE_KEY/MetricsSnapshots elsewhere in this
codebase: raw scrape blobs are kept under PEFT_ANALYZER_BLOB_DIR purely
for inspection/audit, evicted oldest-first once PEFT_ANALYZER_MAX_BYTES
is exceeded. On the deploy target this MUST point under /home
(confirmed /dev/sda5, 667G/42G used/591G avail on the worker-node) —
the env default here only matters for local/dev runs.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..config import settings
from .finetune_log import log_ground_truth_turn

_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_REQUEST_TIMEOUT_S = 10.0
_PROVIDER_TAG = "peft-analyzer"

DEFAULT_MAX_BYTES = 40 * 1024**3  # 40GB


def _blob_dir() -> Path:
    default = os.path.join(settings().DATA_DIR, "peft_analyzer_blobs")
    path = Path(os.getenv("PEFT_ANALYZER_BLOB_DIR", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _max_bytes() -> int:
    return int(os.getenv("PEFT_ANALYZER_MAX_BYTES", DEFAULT_MAX_BYTES))


def _slug(topic: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", topic.strip()).strip("_") or "topic"


def _blob_path(blob_dir: Path, topic: str, *, day: str) -> Path:
    return blob_dir / f"{day}__{_slug(topic)}.json"


def _dir_size_bytes(blob_dir: Path) -> int:
    return sum(f.stat().st_size for f in blob_dir.glob("*.json") if f.is_file())


def _evict_oldest_until_under_cap(blob_dir: Path, max_bytes: int, incoming_bytes: int) -> None:
    """Oldest-file-first eviction, run BEFORE writing a new blob — never
    lets a single write push the directory over cap even momentarily."""
    files = sorted(blob_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
    total = _dir_size_bytes(blob_dir)
    i = 0
    while total + incoming_bytes > max_bytes and i < len(files):
        size = files[i].stat().st_size
        try:
            files[i].unlink()
            total -= size
        except OSError:
            pass
        i += 1


def fetch_topic_summary(topic: str) -> dict | None:
    """None on any failure (unknown title, network hiccup, disambiguation
    page with no extract) — a scrape failure for one topic must never
    abort the rest of the run."""
    try:
        r = httpx.get(_WIKI_SUMMARY_URL.format(title=topic.replace(" ", "_")),
                      timeout=_REQUEST_TIMEOUT_S,
                      headers={"User-Agent": "Algo-Akanade-PEFT-Analyzer/1.0"})
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    return data


def enabled_scrape_topics() -> list[str]:
    """Every enabled finetune-config's own scrape_topics, deduplicated —
    reads the same FinetuneConfigs table admin_finetune_config.py owns,
    directly (this runs as a one-off script, not an HTTP client of its
    own API)."""
    from ..db import store  # deferred: keeps this module importable standalone for tests

    rows = store("FinetuneConfigs").scan("FINETUNECFG#")
    topics: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.get("enabled", True):
            continue
        for topic in row.get("scrapeTopics", []) or []:
            if topic not in seen:
                seen.add(topic)
                topics.append(topic)
    return topics


def scrape_topic(topic: str, *, blob_dir: Path, max_bytes: int, day: str) -> str:
    """Returns "scraped", "skipped_already_done", or "skipped_no_data" —
    a plain string rather than a bool/enum so the CLI summary below can
    report exactly what happened per topic without a second lookup."""
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = _blob_path(blob_dir, topic, day=day)
    if blob_path.exists():
        return "skipped_already_done"  # idempotent per topic per day

    data = fetch_topic_summary(topic)
    if data is None:
        return "skipped_no_data"

    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _evict_oldest_until_under_cap(blob_dir, max_bytes, len(raw))
    blob_path.write_bytes(raw)

    extract = data["extract"].strip()
    log_ground_truth_turn(
        messages=[{"role": "user", "content": f"What is {topic}?"}],
        reply=extract,
        provider=_PROVIDER_TAG,
    )
    return "scraped"


def run(*, blob_dir: Path | None = None, max_bytes: int | None = None) -> dict:
    """The CLI entry point's actual work, split out so tests can call it
    directly without shelling out. Never raises — a bad topic/network
    blip degrades to a smaller `topics_scraped` count, never a crashed
    nightly run (deploy/nightly_finetune.sh's fine-tune step must still
    get a chance to run after this)."""
    blob_dir = blob_dir or _blob_dir()
    max_bytes = max_bytes if max_bytes is not None else _max_bytes()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    topics = enabled_scrape_topics()
    results = {topic: scrape_topic(topic, blob_dir=blob_dir, max_bytes=max_bytes, day=day)
              for topic in topics}
    return {
        "topics_scraped": sum(1 for v in results.values() if v == "scraped"),
        "topics_skipped_already_done": sum(1 for v in results.values() if v == "skipped_already_done"),
        "topics_skipped_no_data": sum(1 for v in results.values() if v == "skipped_no_data"),
        "new_examples_added": sum(1 for v in results.values() if v == "scraped"),
        "blob_bytes_used": _dir_size_bytes(blob_dir),
        "blob_dir": str(blob_dir),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
