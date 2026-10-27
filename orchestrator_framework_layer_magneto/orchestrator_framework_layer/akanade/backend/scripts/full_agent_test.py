"""Comprehensive 60-prompt test (20 each: Coding Agent, Trading Agent,
PM Agent) verifying the real end-to-end response path for all three —
latency, which provider actually answered (SLM/LLM/hybrid/domain),
tier_log fallback behavior, and how much of the total latency is
accounted for by the LLM call itself vs. everything else in the request
path (routing, memory, logging) — the concrete way to check whether
non-LLM work is adding meaningful latency on the hot path.

Deliberately read-only in effect even where a prompt is write-shaped
(e.g. "create a task X") — every write-shaped PM/Trading action stops
at a pending-confirmation reply (native_agent.py's destructive-confirm
step) within a single turn; this script never sends a follow-up "yes",
so nothing is ever actually mutated against a real backend.

Run inside the live container so it hits the real configured providers:
    docker exec ak-chatbot python -m scripts.full_agent_test
"""
from __future__ import annotations

import json
import sys
import time
import uuid

sys.path.insert(0, ".")

from app.framework.bootstrap import bootstrap_default_agents  # noqa: E402
bootstrap_default_agents()

from app.framework.orchestrator import orchestrator  # noqa: E402
from app.framework.task import Task, TaskContext, TaskType  # noqa: E402
from app.agent.native_agent import handle_message  # noqa: E402

CODING_PROMPTS = [
    "Write a Python function that computes the greatest common divisor of two numbers.",
    "Write a function that checks if two strings are anagrams of each other.",
    "Debug this: def get_first(lst): return lst[0] — it crashes on an empty list, fix it.",
    "Write a Python class for a simple bank account with deposit and withdraw methods.",
    "Refactor this to use enumerate: i = 0\nfor item in items:\n    print(i, item)\n    i += 1",
    "Write a function that returns the second-largest number in a list.",
    "Write a context manager that suppresses a specific exception type.",
    "Write unit tests for a function that checks if a number is even.",
    "Write a function that converts a list of tuples into a dictionary.",
    "Write a recursive function to compute factorial.",
    "Review this for bugs: def get_avg(nums): total = 0\nfor n in nums: total += n\nreturn total / len(nums)",
    "Write a Python generator that yields the Fibonacci sequence indefinitely.",
    "Write a function that title-cases every word in a sentence except small words like 'a', 'the'.",
    "Write a simple LRU cache class using an OrderedDict.",
    "Write a function that validates a US ZIP code using regex.",
    "Write a decorator that caches a function's return value (memoization).",
    "Write a function that merges overlapping intervals given as a list of (start, end) tuples.",
    "Write a Python script that counts the number of lines in a text file.",
    "Write a function that checks if a binary tree is balanced.",
    "Write a simple retry loop that retries a function up to 3 times on failure.",
]

TRADING_PROMPTS = [
    "Fetch MSFT price history for 2023.",
    "Backtest a 20/50-day moving average crossover on MSFT for 2023.",
    "Backtest a 10/30 crossover on GOOGL for 2024.",
    "What was AMZN's price history for Q1 2024?",
    "Run a 20/50 crossover backtest on ETH-USD for 2024 and give me the total return.",
    "Compare a 10/30 crossover backtest on NVDA vs buy-and-hold for 2023.",
    "Backtest a 5/20 crossover on SPY for 2021.",
    "Fetch TSLA's daily prices for 2023 and summarize the trend.",
    "Backtest a moving average crossover on META for 2022 with fast window 12 and slow window 26.",
    "What's the Sharpe ratio of a 20/50 crossover strategy on QQQ for 2024?",
    "Fetch the price history for AAPL for the first half of 2024.",
    "Backtest a 15/40 crossover on IBM for 2023.",
    "How many trades did a 20/50 crossover strategy make on AAPL in 2024?",
    "What's the buy-and-hold return for BTC-USD in 2024?",
    "Backtest a 10/50 crossover on AMD for 2022 and tell me the max drawdown.",
    "Fetch NFLX price history for 2024.",
    "Run a 20/50 crossover on DIS for 2023 and compare it to buy-and-hold.",
    "What was the max drawdown for a 5/20 crossover on COIN in 2024?",
    "Backtest a moving average crossover on INTC for 2021 with fast window 20 and slow window 50.",
    "Fetch price history for GOOG for the last quarter of 2023.",
]

PM_PROMPTS = [
    "What tasks are overdue right now?",
    "List all my open tasks.",
    "Show me the current sprint status.",
    "What's on my plate this week?",
    "Summarize the progress of all active projects.",
    "Who has the most tasks assigned right now?",
    "Create a task called 'Fix login bug' in the Aksha PM Web project.",
    "Mark the task 'Fix login bug' as done.",
    "What projects am I a member of?",
    "How many tasks are in the backlog?",
    "Show me tasks that haven't been updated in over a week.",
    "What's the team's completion rate this month?",
    "List all high priority tasks.",
    "Who is the project manager for Aksha PM Mobile?",
    "Reassign task 'Update docs' to Meera.",
    "What's due tomorrow?",
    "Give me a summary of yesterday's completed tasks.",
    "How many active sprints are there right now?",
    "Delete the task 'Old test task'.",
    "What's the average time to complete a task in this project?",
]


def _sum_tier_ms(tier_log: list[dict]) -> float:
    return sum(entry.get("ms", 0) or 0 for entry in tier_log)


def test_coding_agent() -> list[dict]:
    results = []
    for i, prompt in enumerate(CODING_PROMPTS, 1):
        context = TaskContext(task_type=TaskType.CODE_GENERATION,
                              required_capabilities=frozenset({"code_generation"}))
        task = Task(text=prompt, context=context, session_id=f"full-test-coding-{uuid.uuid4()}",
                   metadata={"user": {"sub": "full-test-bot", "name": "Full Test Bot", "role": "Member"},
                            "preferred_provider": None, "client_tools": False, "history": []})
        t0 = time.time()
        error = None
        text = ""
        provider = None
        tier_log: list[dict] = []
        try:
            result = orchestrator.execute(task)
            text = (result.result.text or "") if result.result else ""
            raw = result.result.raw if result.result else {}
            provider = raw.get("provider")
            tier_log = raw.get("tier_log") or []
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000
        overhead_ms = latency_ms - _sum_tier_ms(tier_log)
        print(f"[coding {i}/20] {latency_ms:.0f}ms (overhead {overhead_ms:.0f}ms) provider={provider} "
             f"{'ERROR: ' + error if error else 'ok'} — {prompt[:55]!r}")
        results.append({"id": f"CODING-{i:02d}", "prompt": prompt, "latency_ms": round(latency_ms, 1),
                        "overhead_ms": round(overhead_ms, 1), "provider": provider, "tier_log": tier_log,
                        "error": error, "response_preview": text[:300]})
    return results


def _run_native(prompts: list[dict], label: str) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts, 1):
        user = {"sub": "full-test-bot", "name": "Full Test Bot", "role": "Member"}
        session_id = f"full-test-{label}-{uuid.uuid4()}"
        t0 = time.time()
        error = None
        text = ""
        provider = None
        tier_log: list[dict] = []
        try:
            reply = handle_message(user, session_id, prompt)
            text = reply.reply
            provider = reply.provider
            tier_log = reply.tier_log or []
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000
        overhead_ms = latency_ms - _sum_tier_ms(tier_log)
        print(f"[{label} {i}/{len(prompts)}] {latency_ms:.0f}ms (overhead {overhead_ms:.0f}ms) "
             f"provider={provider} {'ERROR: ' + error if error else 'ok'} — {prompt[:55]!r}")
        results.append({"id": f"{label.upper()}-{i:02d}", "prompt": prompt, "latency_ms": round(latency_ms, 1),
                        "overhead_ms": round(overhead_ms, 1), "provider": provider, "tier_log": tier_log,
                        "error": error, "response_preview": text[:300]})
    return results


def main() -> None:
    out_path = "/tmp/full_agent_test.json"
    out = {"coding_agent": [], "trading_agent": [], "pm_agent": []}

    def flush():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print("=== Coding Agent (orchestration/execute) — 20 prompts ===")
    out["coding_agent"] = test_coding_agent()
    flush()

    print("\n=== Trading Agent (native chat pipeline) — 20 prompts ===")
    out["trading_agent"] = _run_native(TRADING_PROMPTS, "trading")
    flush()

    print("\n=== PM Agent (native chat pipeline) — 20 prompts ===")
    out["pm_agent"] = _run_native(PM_PROMPTS, "pm")
    flush()

    def summarize(label: str, rows: list[dict]) -> None:
        ok = [r for r in rows if not r["error"]]
        lat = [r["latency_ms"] for r in ok]
        overhead = [r["overhead_ms"] for r in ok]
        if lat:
            print(f"{label}: {len(ok)}/{len(rows)} ok, avg={sum(lat)/len(lat):.0f}ms, "
                 f"min={min(lat):.0f}ms, max={max(lat):.0f}ms, "
                 f"avg_overhead={sum(overhead)/len(overhead):.0f}ms")
        else:
            print(f"{label}: 0/{len(rows)} ok")

    print("\n=== Summary ===")
    summarize("Coding Agent", out["coding_agent"])
    summarize("Trading Agent", out["trading_agent"])
    summarize("PM Agent", out["pm_agent"])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
