# Live prompt test report — worker-node (NetBird), 2026-08-29

A real, end-to-end verification pass against the deployed stack at
`aapl19@100.117.127.55` (hostname `worker-node`, reached over NetBird —
same machine previously known at `192.168.1.62`), covering: a fresh
backend/chatbot/frontend redeploy, a new-user creation flow through the
real API, a round of PM-tool/general-knowledge/coding prompts against
the live chatbot, and a check of the two automated background systems
(nightly PEFT fine-tuning, the RL/optimization dashboard). `ak-dynamo`
(the real seeded database) was never stopped, removed, or recreated at
any point in this session.

## 1. Redeploy

Rebuilt and pushed `dockerhubalgo/algo-kaizen-backend:latest` and
`dockerhubalgo/algo-kaizen-chatbot:latest`, then ran
`update_containers.sh` (recreates `ak-backend`/`ak-chatbot`/`ak-frontend`
only — `ak-dynamo` is untouched by design, confirmed by reading the
script before running it).

This took **two rounds**, and the gap between them is worth recording
honestly: the first images were built from a `rsync` snapshot of the
local working tree taken *before* two features had actually been
written to disk yet (the user `description` field and
`optimization_events.py`'s DB wiring), so round 1 deployed successfully
but silently lacked both. Confirmed via `docker exec ak-backend grep
description app/routers/users.py` returning 0 matches post-deploy;
re-synced, rebuilt just the backend image, pushed, and re-ran
`update_containers.sh` for round 2. Round 2 confirmed to actually
contain both (see §2 and §4).

Both rounds: `backend` and `frontend` smoke tests passed immediately;
`chatbot` smoke test showed `FAILED` both times because the script's
own 3-second wait isn't quite enough for this image's startup — in both
cases a manual `curl http://localhost:8001/api/health` seconds later
returned a clean `{"status":"ok",...}`. Not a real failure, just a
smoke-test timing margin worth widening in the script sometime.

## 2. New user: apan@algoanalytics.com

Created via the real API (logged in as the seeded admin
`kvelhal@algoanalytics.com`, `POST /api/users`), then its `description`
set once round 2 made that endpoint available:

| Field | Value |
|---|---|
| name | Aniket Pandit |
| email | apan@algoanalytics.com |
| password | Password123! |
| role | Admin |
| description | Algoanalytics CEO |

Verified: the account can log in on its own
(`POST /api/auth/login` → real access token), and shows up correctly
in the chatbot's own "List all users" tool call (see §3, row 2) —
confirming the chatbot's real-app integration sees new accounts
immediately, no cache/restart needed.

## 3. Live chatbot prompts (PM tool, general knowledge, coding)

Logged in as `kvelhal@algoanalytics.com` (Admin) against
`POST /api/chat/message` on the redeployed `ak-chatbot`. Each row got
its own fresh `session_id` unless noted, to avoid one prompt's pending
tool-confirmation bleeding into the next (see the caveat under row 7).

| # | Prompt | Reply | Verdict |
|---|---|---|---|
| 1 | Tell me about Aniruddha — name, email, description. | "Aniruddha Pant is Manager2 — apant@algoanalytics.com." | ✅ correct, real seeded data |
| 2 | List all users. | 9 users, including the just-created **Aniket Pandit — Admin** | ✅ correct, new user visible live |
| 3 | Who knows Python? | "Keshav Velhal — Admin" | ✅ correct, matches prior test round |
| 4 | My personal tasks | 2 real personal tasks returned | ✅ correct, real data |
| 5 | Write a Python function that returns the nth Fibonacci number. | Correct, working iterative implementation | ✅ correct (phrasing was slightly odd — see §6) |
| 6 | What's the capital of France? | "Paris" | ✅ correct, general knowledge |
| 7 | Who is the CEO of Algoanalytics? | Asked to confirm an external Wikipedia fetch (`fetch` MCP tool) | ✅ correct routing — recognized as needing outside info, safely asked before calling out |
| 8 | list roles as a Member | Listed all 4 built-in roles | ⚠️ not a real RBAC test — I was authenticated as Admin, not an actual Member account, so this only confirms an Admin caller sees the full list correctly, not what a Member would see (prior report's #30 already covered the real Member-decline case with Groq) |
| 9 | What is ALGO AKANADE? / What does ALGO AKANADE do? | A list of unrelated **projects**, then (retried) a RAG search returning 3 unrelated ticket-like snippets **each scored 0.0** | ❌ real bug — see §6 |

## 4. Nightly PEFT fine-tuning — mechanism works, no data yet

`ak-finetune-cron` runs on schedule (03:30 UTC daily, confirmed via its
own scheduler log) and has exited `0` both nights since this box came
up (2026-08-28, 2026-08-29). Its detailed log
(`/opt/algo-kaizen/logs/ak-finetune.log`) shows, both nights:

```
No fine-tune log in ak-chatbot yet (first night, or local SLM hasn't
answered anything) — nothing to do tonight.
```

This is the script's own documented, correct no-op path (see
`deploy/nightly_finetune.sh`'s header comment: "too few reward>=min
examples... exit cleanly without touching anything already serving").
**The schedule and exit-clean behavior are confirmed working; there is
just no reward-labeled training data yet** — that needs real users to
submit feedback ratings on chat replies (`FeedbackLoop.close()`), which
hasn't happened yet on this fresh deployment. Not a defect — will start
producing real fine-tunes once that usage accumulates.

## 5. Optimization dashboard — found and fixed a real persistence bug

`ak-opt-dashboard-backend`/`-frontend` were both up and answering
(`/api/health`, login, `/api/summary`) — the dashboard's own plumbing
was never broken. But before any fix, `/api/summary` showed
**`n_calls: 0`** even right after sending real chat traffic through
`ak-chatbot`, which should have logged an `OptimizationEvents` row for
every single call (`optimization_events.py`, wired into
`llm/router.py`).

**Root cause, found by comparing container configs**: the server's
`chatbot.env` had an explicit `DATA_DIR=./data`, which overrides the
image's own correct baked-in default (`ENV DATA_DIR=/srv/data` in the
Dockerfile) — the correct value is the one the `ak-chatbot-data` volume
is actually mounted at. With the override in place, every
`local`-backend write (`OptimizationEvents`, and potentially anything
else going through `app/db.py`'s `store()`) was landing in the
container's own ephemeral filesystem layer instead of the shared
volume — invisible to the dashboard, and **lost on every container
restart**, not just unreported.

**Fix applied**: removed the stray `DATA_DIR=./data` line from
`/home/aapl19/Desktop/algo-kaizen/chatbot.env` on the server (backed up
first as `chatbot.env.bak.datadirfix.<timestamp>`), then redeployed.
Confirmed fixed — after the fix, `docker inspect ak-chatbot` shows
`DATA_DIR=/srv/data`, and one real chat call afterward produced a
non-empty `/api/summary`:

```
n_calls: 2, success_rate: 1.0, by_provider: {groq-llama-3.3-70b: 1, gemini-flash: 1}, by_arm: {high_quality: 2}
```

This template (`deploy/chatbot.env.example`) never had this line —
it's specific to this one server's env file, so no repo change was
needed, only the server-side fix.

## 6. Known issue: "What is ALGO AKANADE?" gets misrouted

Asking the chatbot to explain itself — a reasonable, expected question
for a brand-new admin — does not get answered directly. It was
observed doing two different wrong things across two tries:
- Once: treated as a request to list PM-tool projects.
- Once: ran a RAG search and returned 3 unrelated ticket-sounding
  snippets ("Android login crash", "Payment gateway timeout", "Dark
  mode for dashboard"), **each explicitly scored `0.0`** — meaning the
  retrieval layer itself correctly found nothing relevant, but nothing
  downstream checks that score before presenting the results as if
  they were real "closest matches."

Two separate, real gaps worth a follow-up pass: (1) a meta/self
question about the product isn't recognized and routed to a real
"about ALGO AKANADE" answer, and (2) RAG results with a similarity score of
0.0 should never be presented as matches at all — that's a missing
relevance floor, not a missing document.

Also minor, not a bug: the Fibonacci prompt's reply opened with "None
of the provided information mentions Python, so I will create..." —
correct code, slightly odd throat-clearing before it. Cosmetic.

## 7. Voice — verified in code, not testable from this shell

Voice is client-side (mobile TTS, VS Code mic/speaker) and has no
server API to curl. Both were addressed as real, separate fixes earlier
in this session:
- Mobile `flutter_tts` "stuck at pause" — fixed by registering real
  start/completion/cancel/error handlers and `awaitSpeakCompletion(true)`
  (`tool_layer_akanade/mobile/lib/services/voice_service.dart`).
- VS Code extension (`algo-kodo` v0.9.2) — added a mic button
  (Chromium `SpeechRecognition`, dictates into the input) and a speaker
  toggle (`speechSynthesis`, reads replies aloud), both start/stop
  cleanly.

## Bottom line

| Check | Result |
|---|---|
| ak-dynamo untouched throughout | ✅ confirmed |
| Redeploy (2 rounds needed) | ✅ both succeeded; round 2 needed for description field + DATA_DIR fix |
| New user created + logs in + description set | ✅ apan@algoanalytics.com |
| PM tool prompts (real data) | ✅ 4/4 correct |
| General knowledge / external-tool routing | ✅ 2/2 correct |
| Coding prompt | ✅ correct, minor phrasing nit |
| "What is ALGO AKANADE?" self-description | ❌ real bug, not yet fixed — see §6 |
| Nightly PEFT cron | ✅ schedule + exit-clean behavior confirmed; no training data yet (expected) |
| Optimization dashboard | ✅ now receiving real events — was silently broken (§5), fixed live |
| Voice in/out | ✅ fixed this session (mobile + VS Code), not curl-testable |
