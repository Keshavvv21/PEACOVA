"""The final SLM coding-prompt test suite — at least 100 prompts across
coding/generation, debugging, refactoring, testing, and repository-
level tasks, run for real against a named provider (default:
ollama-coder), with a generated pass/fail + latency + token + quality
report. See README_SLM_TEST_SUITE.md for the full methodology and how
to read the report.

Run inside a container that already has the real provider reachable
(ak-chatbot can reach ak-ollama over ak-net) — imports this repo's own
llm.router/llm.intent_reasoner directly, same pattern as
scripts/log_complex_ground_truth.py:

    docker exec ak-chatbot python -m scripts.run_slm_test_suite \\
        --provider ollama-coder --out /srv/data/slm_test_report.json

Deliberately calls call_provider_directly() (bypasses every tier/chain/
bandit) — this suite needs the NAMED provider's own real answer to
every prompt, never a silent substitute from TierClient's own fallback
chain (see log_complex_ground_truth.py's own hard-won lesson on this,
same session).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── the prompt catalog ──────────────────────────────────────────────
# Each entry: (id, category, prompt). Categories match
# llm/intent_reasoner.py's own six intents exactly (generate/debug/
# refactor/test/repo_analysis, plus a handful of "reasoning"-shaped
# prompts left uncategorized on purpose - general_coding is a REAL
# bucket the reasoner falls back to, and the suite should exercise it
# too, not just the five specific ones).

CodingPrompt = tuple[str, str, str]

PROMPTS: list[CodingPrompt] = [
    # ── generate (30) ──────────────────────────────────────────────
    ("GEN-01", "generate", "Write a Python function that reverses a string."),
    ("GEN-02", "generate", "Write a Python function that checks whether a number is prime."),
    ("GEN-03", "generate", "Write a function that returns the nth Fibonacci number iteratively."),
    ("GEN-04", "generate", "Write a Python function to flatten a nested list of arbitrary depth."),
    ("GEN-05", "generate", "Write a function that merges two sorted lists into one sorted list."),
    ("GEN-06", "generate", "Write a Python decorator that times how long a function takes to run."),
    ("GEN-07", "generate", "Write a context manager that temporarily changes the working directory."),
    ("GEN-08", "generate", "Write a function that validates whether a string is a valid email address."),
    ("GEN-09", "generate", "Write a Python class representing a simple stack with push/pop/peek."),
    ("GEN-10", "generate", "Write a binary search function over a sorted list."),
    ("GEN-11", "generate", "Write a function that counts word frequency in a block of text."),
    ("GEN-12", "generate", "Write a FastAPI endpoint that accepts a POST with a JSON body and echoes it back."),
    ("GEN-13", "generate", "Write a Dockerfile for a Python 3.12 service served by uvicorn on port 8000."),
    ("GEN-14", "generate", "Write a bash script that rotates and gzips log files older than 30 days."),
    ("GEN-15", "generate", "Write a Python script that reads a CSV and prints the row and column counts."),
    ("GEN-16", "generate", "Implement a thread-safe LRU cache in Python supporting concurrent reads and writes."),
    ("GEN-17", "generate", "Write a Python function that computes the SHA-256 hash of a file's contents."),
    ("GEN-18", "generate", "Write a generator function that yields prime numbers indefinitely."),
    ("GEN-19", "generate", "Write a function that deep-merges two Python dictionaries."),
    ("GEN-20", "generate", "Write a simple token-bucket rate limiter class in Python."),
    ("GEN-21", "generate", "Write a Python function that converts a snake_case string to camelCase."),
    ("GEN-22", "generate", "Write a retry decorator with exponential backoff for a flaky function."),
    ("GEN-23", "generate", "Write a Python function that parses a simple INI-style config string into a dict."),
    ("GEN-24", "generate", "Write a function that finds the longest common prefix among a list of strings."),
    ("GEN-25", "generate", "Write a Python class for a min-heap supporting push and pop-min."),
    ("GEN-26", "generate", "Write a function that validates balanced parentheses/brackets/braces in a string."),
    ("GEN-27", "generate", "Write a Python async function that fetches multiple URLs concurrently and returns their status codes."),
    ("GEN-28", "generate", "Write a function that computes the edit distance (Levenshtein) between two strings."),
    ("GEN-29", "generate", "Write a simple event emitter class supporting on/off/emit."),
    ("GEN-30", "generate", "Write a Python function that paginates a list into chunks of a given size."),

    # ── debug (25) ─────────────────────────────────────────────────
    ("DBG-01", "debug", "This function should return the max of a list but returns the wrong value for negative numbers: def find_max(nums):\n    m = 0\n    for n in nums:\n        if n > m: m = n\n    return m\nFix it."),
    ("DBG-02", "debug", "Debug why this Kafka consumer keeps rebalancing every few seconds."),
    ("DBG-03", "debug", "Find and fix a race condition in a Python asyncio producer-consumer system."),
    ("DBG-04", "debug", "This recursive factorial function causes a RecursionError for n=0. Why, and how do you fix it? def factorial(n): return n * factorial(n-1)"),
    ("DBG-05", "debug", "Debug 'MongoServerError: $changeStream only supported on replica sets.'"),
    ("DBG-06", "debug", "Why does this loop never terminate? i = 0\nwhile i < 10:\n    if i == 5:\n        continue\n    i += 1"),
    ("DBG-07", "debug", "This Python dict comprehension raises 'unhashable type: list' — why, and what's the fix? {[k]: v for k, v in items}"),
    ("DBG-08", "debug", "Debug why two batch pods land on the same node despite anti-affinity."),
    ("DBG-09", "debug", "This function leaks file handles under load — find the bug: def read_all(paths):\n    results = []\n    for p in paths:\n        f = open(p)\n        results.append(f.read())\n    return results"),
    ("DBG-10", "debug", "Fix this Kubernetes Deployment YAML — the readiness probe never passes."),
    ("DBG-11", "debug", "This binary search has an off-by-one bug causing an infinite loop. Find and fix it: def bsearch(a, target):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo+hi)//2\n        if a[mid] < target: lo = mid\n        else: hi = mid\n    return lo"),
    ("DBG-12", "debug", "Diagnose a multi-symptom incident: Kafka consumer lag is climbing, a TRT inference engine has stalled, and false-positive detections are rising at the same time. What's the likely root cause and fix order?"),
    ("DBG-13", "debug", "This Python service's memory grows continuously over several hours — what are the likely causes and how would you confirm which one?"),
    ("DBG-14", "debug", "This mutable-default-argument bug causes state to leak between calls — find it: def add_item(item, items=[]):\n    items.append(item)\n    return items"),
    ("DBG-15", "debug", "Fix a PostgreSQL transaction bug causing duplicate financial transactions under concurrent requests."),
    ("DBG-16", "debug", "This async function never actually awaits its coroutine — find the bug: async def fetch(url):\n    result = fetch_data(url)\n    return result"),
    ("DBG-17", "debug", "Why would AUROC of 0.91 offline and deployed precision of only 0.18 both be true at once?"),
    ("DBG-18", "debug", "This code silently swallows every exception, hiding real bugs — find and fix it: try:\n    process()\nexcept:\n    pass"),
    ("DBG-19", "debug", "Analyze and fix a database deadlock involving multiple concurrent workers."),
    ("DBG-20", "debug", "This regex-based email validator rejects valid addresses with a '+' — why, and fix it: re.match(r'^[\\w.]+@[\\w.]+$', email)"),
    ("DBG-21", "debug", "35% of served responses still use the old model after a deployment meant to fully switch over — what are the likely causes?"),
    ("DBG-22", "debug", "This function compares floats with == and fails intermittently — find the bug: if result == 0.3: ..."),
    ("DBG-23", "debug", "Given failing logs showing a connection pool exhausted error under load, identify the likely root cause and the minimal production-safe fix."),
    ("DBG-24", "debug", "This code mutates a list while iterating over it, skipping elements — find the bug: for item in items:\n    if should_remove(item): items.remove(item)"),
    ("DBG-25", "debug", "Profile a slow API and describe how you'd determine whether the bottleneck is CPU, database, network, or serialization."),

    # ── refactor (15) ──────────────────────────────────────────────
    ("REF-01", "refactor", "Refactor this 400-line function into smaller modules without changing behavior."),
    ("REF-02", "refactor", "Refactor this nested-if chain into a cleaner form, preserving exact behavior: if a:\n    if b:\n        do_x()\n    else:\n        do_y()\nelse:\n    do_z()"),
    ("REF-03", "refactor", "Refactor this repeated try/except block into a reusable decorator without changing what it catches."),
    ("REF-04", "refactor", "Extract the duplicated validation logic in these three functions into one shared helper."),
    ("REF-05", "refactor", "Refactor this class to use dataclasses instead of a hand-written __init__ and __repr__."),
    ("REF-06", "refactor", "Rewrite this callback-based function using async/await, preserving the exact same external behavior."),
    ("REF-07", "refactor", "Refactor this global-mutable-state counter into a proper class, without changing any caller's code."),
    ("REF-08", "refactor", "Simplify this deeply nested list comprehension into clearer, equivalent code."),
    ("REF-09", "refactor", "Refactor this function that returns different types depending on a flag into two clearly-named functions."),
    ("REF-10", "refactor", "Rename and reorganize this poorly-named utils.py's functions into cohesive groups without breaking any imports."),
    ("REF-11", "refactor", "Refactor this code to remove the magic numbers, replacing them with named constants."),
    ("REF-12", "refactor", "Refactor this synchronous file-processing loop to process files concurrently, preserving the same output order."),
    ("REF-13", "refactor", "Convert this class-based singleton pattern into a simpler module-level singleton, without changing its public API."),
    ("REF-14", "refactor", "Refactor this SQL query built via string concatenation into a parameterized query, without changing its results."),
    ("REF-15", "refactor", "De-duplicate this pair of near-identical functions into one, parameterized by their one real difference."),

    # ── test (15) ──────────────────────────────────────────────────
    ("TST-01", "test", "Write unit tests for this alert-rule cooldown logic."),
    ("TST-02", "test", "Write pytest tests for a function that returns precision, recall, and F1 without zero-division errors."),
    ("TST-03", "test", "Write a test that reproduces this bug in the parser before fixing it: it crashes on empty input."),
    ("TST-04", "test", "Write unit tests covering the edge cases (empty list, single element, duplicates) for a binary search function."),
    ("TST-05", "test", "Write comprehensive pytest tests for a FastAPI service including failures, concurrency, and edge cases."),
    ("TST-06", "test", "Write a test that mocks an external HTTP call and asserts the function retries on a 500 response."),
    ("TST-07", "test", "Write tests for a rate limiter verifying it allows exactly N requests per window and rejects the N+1th."),
    ("TST-08", "test", "Write a parametrized pytest test covering positive, negative, and zero inputs for an is_prime function."),
    ("TST-09", "test", "Write a test asserting a thread-safe counter produces the correct total under concurrent increments."),
    ("TST-10", "test", "Write tests for a CSV parser verifying it handles quoted commas and missing trailing newlines correctly."),
    ("TST-11", "test", "Write a test that verifies a cache eviction policy actually evicts the least-recently-used entry."),
    ("TST-12", "test", "Write tests asserting a JSON schema validator rejects a payload missing a required field."),
    ("TST-13", "test", "Write a test verifying a decorator preserves the wrapped function's name and docstring."),
    ("TST-14", "test", "Write tests for a function that should raise ValueError on negative input, not silently clamp it."),
    ("TST-15", "test", "Write a test suite for a stack class covering push, pop on empty, and peek on empty."),

    # ── repo_analysis (10) ─────────────────────────────────────────
    ("REPO-01", "repo_analysis", "Find every TODO/FIXME in this codebase with file and line number."),
    ("REPO-02", "repo_analysis", "Extract all API endpoints and their HTTP methods from this FastAPI project."),
    ("REPO-03", "repo_analysis", "List every environment variable referenced anywhere in this repository."),
    ("REPO-04", "repo_analysis", "Find all Kafka topic names used across this repository."),
    ("REPO-05", "repo_analysis", "Review this entire codebase for obvious security issues (hardcoded secrets, SQL string concatenation, eval usage)."),
    ("REPO-06", "repo_analysis", "Identify every place in this project that reads from an environment variable without a default value."),
    ("REPO-07", "repo_analysis", "Find all functions in this repository longer than 100 lines and list them by file."),
    ("REPO-08", "repo_analysis", "Reconcile this repo's requirements.txt against what's actually imported anywhere in the codebase."),
    ("REPO-09", "repo_analysis", "Find every place this project catches a bare `except:` with no specific exception type."),
    ("REPO-10", "repo_analysis", "Identify which modules in this repository have no corresponding test file."),

    # ── general_coding / reasoning (10 — intentionally NOT keyword-matched) ──
    ("GEN-REASON-01", "general_coding", "What's the difference between a Kubernetes Deployment and a StatefulSet?"),
    ("GEN-REASON-02", "general_coding", "Compare vLLM vs. TGI vs. Ollama for self-hosting a 13B model on one GPU box."),
    ("GEN-REASON-03", "general_coding", "What is the purpose of a nodeSelector vs. a taint/toleration in Kubernetes?"),
    ("GEN-REASON-04", "general_coding", "Explain the tradeoff between polling a database vs. using change streams for a live dashboard."),
    ("GEN-REASON-05", "general_coding", "When would you choose AWQ over GGUF quantization for a self-hosted model?"),
    ("GEN-REASON-06", "general_coding", "Explain what a Kafka consumer group is and why a service would use one."),
    ("GEN-REASON-07", "general_coding", "What's the difference between at-least-once and exactly-once delivery, and why does it matter for a payment system?"),
    ("GEN-REASON-08", "general_coding", "Compare a rules engine vs. a small language model for structured decision-making on latency and determinism."),
    ("GEN-REASON-09", "general_coding", "Explain why adding more retrieved chunks to a RAG pipeline can make answers worse even when the right chunk is included."),
    ("GEN-REASON-10", "general_coding", "Give a decision framework for choosing between one large model and a router with several small specialist models."),
]

assert len(PROMPTS) >= 100, f"only {len(PROMPTS)} prompts — need at least 100"


@dataclass
class PromptResult:
    id: str
    category: str
    prompt: str
    passed: bool
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None
    response_preview: str
    intent_classified: str | None
    intent_matches_category: bool | None
    quality_notes: list[str] = field(default_factory=list)


_FENCE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _quality_check(category: str, prompt: str, text: str) -> list[str]:
    """Cheap, deterministic quality signals — not a substitute for human
    review, but a real, reproducible first pass. Every check here is
    something you could verify by eye in the response, not a vibe."""
    notes: list[str] = []
    blocks = _FENCE.findall(text)
    wants_code = category in ("generate", "debug", "refactor", "test")
    if wants_code and not blocks:
        notes.append("No fenced code block in a response that should contain code.")
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            # Not fatal — plenty of real snippets are intentionally
            # partial (a function body meant to be pasted into existing
            # code) — but always worth flagging for a human to check.
            notes.append(f"A code block did not parse as valid Python: {e.msg} (line {e.lineno}).")
    if category == "repo_analysis" and "list_files" not in text and "grep" not in text and len(blocks) == 0:
        notes.append("repo_analysis prompt but the response shows no sign of having looked at real files "
                    "(expected in client_tools mode; a plain single-call test like this one can't use "
                    "list_files/grep at all, so this is EXPECTED here, not a real defect).")
    if len(text.strip()) < 20:
        notes.append("Suspiciously short response.")
    return notes


def run_suite(provider: str, timeout: float, domain: str | None,
             out_path: Path | None = None) -> list[PromptResult]:
    from app.framework.agents.coding_agent import SYSTEM_PROMPT
    from app.llm.intent_reasoner import classify_intent, guidance_for
    from app.llm.router import call_provider_directly

    results: list[PromptResult] = []
    for pid, category, prompt in PROMPTS:
        intent = classify_intent(prompt)
        system_content = SYSTEM_PROMPT["content"] + guidance_for(intent)
        messages = [{**SYSTEM_PROMPT, "content": system_content}, {"role": "user", "content": prompt}]
        t0 = time.time()
        result = call_provider_directly(provider, messages, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
        usage = result.usage or {}
        error = result.error
        text = result.text or ""
        quality_notes = [] if error else _quality_check(category, prompt, text)
        passed = bool(not error and text.strip() and not any(
            n.startswith("A code block did not parse") for n in quality_notes))
        intent_matches = None if category == "general_coding" else (intent == category)
        results.append(PromptResult(
            id=pid, category=category, prompt=prompt, passed=passed,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            error=error, response_preview=text[:400],
            intent_classified=intent, intent_matches_category=intent_matches,
            quality_notes=quality_notes,
        ))
        status = "PASS" if passed else "FAIL"
        preview_line = text[:120].replace("\n", " ") if text else ""
        print(f"[{status}] {pid} ({category}) — {latency_ms:.0f}ms" + (f" — ERROR: {error}" if error else ""))
        if preview_line:
            print(f"       └─ {preview_line}{'…' if len(text) > 120 else ''}")
        # Written after EVERY prompt, not just at the end — a run this
        # long (real, observed: ~90s/prompt average on a CPU-only 1.5B
        # model, 105 prompts) is worth being able to inspect mid-flight
        # (e.g. a published report page polling this same file) rather
        # than only ever producing a result if it runs to full
        # completion uninterrupted.
        if out_path is not None:
            write_report(results, out_path, provider, domain)
    return results


def write_report(results: list[PromptResult], out_path: Path, provider: str, domain: str | None) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    latencies = [r.latency_ms for r in results if r.error is None]
    intent_checks = [r for r in results if r.intent_matches_category is not None]
    intent_correct = sum(1 for r in intent_checks if r.intent_matches_category)
    by_category: dict[str, dict] = {}
    for r in results:
        c = by_category.setdefault(r.category, {"total": 0, "passed": 0, "latencies": []})
        c["total"] += 1
        c["passed"] += int(r.passed)
        if r.error is None:
            c["latencies"].append(r.latency_ms)
    for c in by_category.values():
        c["avg_latency_ms"] = round(sum(c["latencies"]) / len(c["latencies"]), 1) if c["latencies"] else None
        del c["latencies"]

    report = {
        "provider": provider,
        "domain": domain,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_prompts": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1) if latencies else None,
            "intent_reasoner_accuracy": round(intent_correct / len(intent_checks), 3) if intent_checks else None,
        },
        "by_category": by_category,
        "results": [r.__dict__ for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report to {out_path}")
    print(f"Pass rate: {passed}/{total} ({report['summary']['pass_rate']:.0%})" if total else "No prompts run.")
    if intent_checks:
        print(f"Intent-reasoner accuracy: {intent_correct}/{len(intent_checks)} "
             f"({report['summary']['intent_reasoner_accuracy']:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="ollama-coder",
                        help="Provider key from llm/router.py to test (default: ollama-coder).")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="Per-prompt timeout in seconds (default 90 — some of these prompts "
                             "genuinely ask for long, detailed answers).")
    parser.add_argument("--domain", default=None,
                        help="Recorded in the report only — which fine-tune domain this provider "
                             "corresponds to, if any.")
    parser.add_argument("--out", type=Path, default=Path("slm_test_report.json"))
    args = parser.parse_args()
    results = run_suite(args.provider, args.timeout, args.domain, out_path=args.out)
    write_report(results, args.out, args.provider, args.domain)


if __name__ == "__main__":
    main()
