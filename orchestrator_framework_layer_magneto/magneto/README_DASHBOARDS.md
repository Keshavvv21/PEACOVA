# Admin dashboards — what's where, and what's real

Every dashboard here reads from real, persisted data — no page shows a
mocked or point-in-time-only number without saying so. All are Admin
role (admits the real app's Manager2/"Senior Manager"); mutations stay
Executive-only where noted.

## RL Metrics Dashboard (`RlMetricsDashboard.tsx`)

The orchestrator's own agent-selection bandit (`app/framework/router.py`)
— current per-agent posterior weights, reward-weight trend over time
(`GET /api/v1/metrics`, `GET /api/v1/metrics/history`). Now also shows
**"Model routing — who's winning"**: per-domain fine-tuned-SLM win rate
and per-`(tier, provider)` win rate (`GET /api/v1/metrics/routing`) —
the actual answer to "is a local model beating the hosted one, and for
what."

## PEFT Metrics Dashboard (`PeftMetricsDashboard.tsx`)

Nightly fine-tune history: gate pass/fail trend, eval score by domain
(`GET /api/admin/slm-eval-reports`, `GET /api/admin/slm-versions`) —
the regression-gate side of `README_PEFT_ANALYZER.md`'s pipeline. Shows
which domains have live, registered models and whether their last
promotion passed.

## RAG Metrics (`GET /api/v1/rag/metrics`)

Per-(knowledge source, retrieval algorithm) hit rate/success rate
(`app/framework/rag_optimizer.py`'s `stats()`), plus overall query
coverage (`total_queries`, `queries_with_sources`). The template every
other per-arm bandit view above (`domain_routing`, `provider_routing`)
follows.

## Security Dashboard (`AdminSecurity.tsx`)

Two real, distinct things, not one:
- **Recent events** (`GET /api/admin/security/events`) — risky/needs-
  review USER PROMPTS the chat's `security_guard.py` flagged, plus an
  LLM-summarized daily digest (`GET /api/admin/security/daily-report`).
- **Threats & vulnerabilities identified** (`GET /api/admin/security/
  findings`, new) — every scan run by the three static-analysis
  agents, persisted (`app/framework/agents/security_findings.py`):
  - `code-security-agent` — hardcoded secrets, `eval`/`exec`, shell
    injection patterns in pasted/scanned code.
  - `supply-chain-agent` — pinned dependency versions checked against
    OSV.dev (a real, free, public vulnerability database — not an LLM
    guessing).
  - `cloud-config-agent` — Dockerfile/compose/Kubernetes/Terraform
    misconfigurations (privileged containers, `0.0.0.0/0`, root users,
    disabled encryption, hardcoded secrets).

  A clean scan is logged too (`highest_severity: "none"`) — the
  dashboard shows what was actually checked and came back safe, not
  only what tripped a rule.

## Optimization Dashboard (standalone app, `algo-kaizen-optimization-
dashboard-{backend,frontend}`)

Its own login (independent of the main app's users/JWT — see
`README_OPTIMIZATION_DASHBOARD.md`), reading `OptimizationEvents`/
`BanditState` off a shared read-only volume from `ak-chatbot`. Shows
real per-call provider/latency/success stats and the RL-tier's live
posteriors. **Known deploy gotcha, already hit and fixed once**: this
only works if the chatbot's `DATA_DIR` actually points at the SAME
mounted volume the dashboard reads — see `README_LIVE_PROMPT_TEST_
REPORT_2026-08-29.md`'s §5 for the real incident where a stray
`DATA_DIR=./data` in `chatbot.env` silently broke this for a while.

## AI Models (`AdminAIModels.tsx`)

Already the "all available models and API keys, add/manage new models"
surface — provider CRUD (`GET/POST/PUT/DELETE /api/admin/providers`),
budgets, finetune-config (now including PEFT Analyzer's `scrape_
topics`, see `README_PEFT_ANALYZER.md`), SLM versions/eval reports.
