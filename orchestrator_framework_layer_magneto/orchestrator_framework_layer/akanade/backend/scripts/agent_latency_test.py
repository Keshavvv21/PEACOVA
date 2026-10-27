"""One-off latency probe for the two chat surfaces users reported as
slow: the Coding Agent (POST /api/v1/orchestration/execute) and the
Trading Agent (POST /api/chat/message's native pipeline, with the
trading tools from app/agent/trading_tools.py). Calls the exact same
in-process entry points those endpoints call — app.framework.
orchestrator.orchestrator.execute() and app.agent.native_agent.
handle_message() — so the measured latency includes real routing,
tool-calling, and (for Trading) real external HTTP to Yahoo Finance,
not just raw LLM inference. Bypasses HTTP/auth entirely (same
reasoning as scripts/run_slm_test_suite.py's direct-provider calls):
this is a synthetic, isolated user_id/session_id per prompt, never a
real account's data.

Run inside the live container so it hits the real configured providers:
    docker exec ak-chatbot python -m scripts.agent_latency_test
"""
from __future__ import annotations

import json
import sys
import time
import uuid

sys.path.insert(0, ".")

from app.framework.bootstrap import bootstrap_default_agents  # noqa: E402
from app.framework.orchestrator import orchestrator  # noqa: E402
from app.framework.task import Task, TaskContext, TaskType  # noqa: E402
from app.agent.native_agent import handle_message  # noqa: E402

# This script runs as its own fresh process (docker exec), not attached to
# the live uvicorn server — the registry singleton starts empty here and
# needs the same one-time bootstrap app/main.py runs at real startup.
bootstrap_default_agents()

CODING_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome.",
    "Refactor this function to use a list comprehension: result = []\n"
    "for x in range(10):\n    if x % 2 == 0:\n        result.append(x*x)",
    "Debug this function: it should return the average of a list but crashes "
    "on empty input. def avg(nums): return sum(nums)/len(nums)",
    "Write unit tests for a function that adds two numbers.",
    "Write a Python class for a simple queue implemented using two stacks.",
    "Write a function that removes duplicates from a list while preserving order.",
    "Review this code for bugs: def divide(a, b): return a / b",
    "Write a decorator that logs the arguments and return value of a function.",
    "Generate a FastAPI GET endpoint that returns the current server time.",
    "Write a function that flattens a nested list of arbitrary depth.",
]

TRADING_PROMPTS = [
    "Fetch AAPL price history for 2024.",
    "Backtest a 20/50-day moving average crossover on AAPL for 2024.",
    "Backtest a 10/30 crossover on MSFT from 2023-01-01 to 2023-12-31.",
    "What was TSLA's price history for the last quarter of 2023?",
    "Run a 20/50 crossover backtest on BTC-USD for 2024 and tell me the Sharpe ratio.",
    "Compare a 20/50 crossover backtest on AAPL vs buy-and-hold for 2024.",
    "Backtest a 5/20 crossover on GOOGL for 2022.",
    "Fetch NVDA's daily prices for 2024 and summarize the trend.",
    "Backtest a moving average crossover on AMZN for 2021 with fast window 15 and slow window 40.",
    "What's the max drawdown of a 20/50 crossover strategy on SPY for 2024?",
]


def test_coding_agent() -> list[dict]:
    results = []
    for i, prompt in enumerate(CODING_PROMPTS, 1):
        context = TaskContext(task_type=TaskType.CODE_GENERATION,
                              required_capabilities=frozenset({"code_generation"}))
        task = Task(text=prompt, context=context, session_id=f"latency-test-coding-{uuid.uuid4()}",
                   metadata={"user": {"sub": "latency-test-bot", "name": "Latency Test Bot",
                                     "role": "Member"},
                            "preferred_provider": None, "client_tools": False, "history": []})
        t0 = time.time()
        error = None
        text = ""
        selected_agents: list[str] = []
        try:
            result = orchestrator.execute(task)
            text = (result.result.text or "") if result.result else ""
            selected_agents = result.selected_agents
        except Exception as exc:  # noqa: BLE001 — a real failure IS the data point here
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000
        print(f"[coding {i}/{len(CODING_PROMPTS)}] {latency_ms:.0f}ms "
             f"{'ERROR: ' + error if error else 'ok'} — {prompt[:60]!r}")
        results.append({"id": f"CODING-{i:02d}", "prompt": prompt, "latency_ms": round(latency_ms, 1),
                        "selected_agents": selected_agents, "error": error,
                        "response_preview": text[:300]})
    return results


def test_trading_agent() -> list[dict]:
    results = []
    for i, prompt in enumerate(TRADING_PROMPTS, 1):
        user = {"sub": "latency-test-bot", "name": "Latency Test Bot", "role": "Member"}
        session_id = f"latency-test-trading-{uuid.uuid4()}"
        t0 = time.time()
        error = None
        text = ""
        try:
            reply = handle_message(user, session_id, prompt)
            text = reply.reply
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.time() - t0) * 1000
        print(f"[trading {i}/{len(TRADING_PROMPTS)}] {latency_ms:.0f}ms "
             f"{'ERROR: ' + error if error else 'ok'} — {prompt[:60]!r}")
        results.append({"id": f"TRADING-{i:02d}", "prompt": prompt, "latency_ms": round(latency_ms, 1),
                        "error": error, "response_preview": text[:300]})
    return results


def main() -> None:
    print("=== Coding Agent (orchestration/execute) ===")
    coding = test_coding_agent()
    print("\n=== Trading Agent (native chat pipeline) ===")
    trading = test_trading_agent()

    out = {"coding_agent": coding, "trading_agent": trading}
    with open("/tmp/agent_latency_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    def summarize(label: str, rows: list[dict]) -> None:
        ok = [r for r in rows if not r["error"]]
        lat = [r["latency_ms"] for r in ok]
        if lat:
            avg = sum(lat) / len(lat)
            print(f"{label}: {len(ok)}/{len(rows)} ok, avg={avg:.0f}ms, "
                 f"min={min(lat):.0f}ms, max={max(lat):.0f}ms")
        else:
            print(f"{label}: 0/{len(rows)} ok")

    print("\n=== Summary ===")
    summarize("Coding Agent", coding)
    summarize("Trading Agent", trading)
    print("Wrote /tmp/agent_latency_test.json")


if __name__ == "__main__":
    main()
