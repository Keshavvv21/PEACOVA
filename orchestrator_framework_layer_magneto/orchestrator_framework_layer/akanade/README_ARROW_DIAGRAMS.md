# ALGO AKANADE — Arrow Diagrams, Layer by Layer

Every real data-flow in this system, as an arrow diagram, in the order a
request actually moves through them. No math here (see
`README_RL_ALGORITHMS.md`) and no prose explainer (see
`README_EXPLAINED.md`) — this file is the pictures.

## 1. The whole stack, top to bottom

```
 Web (React) / Mobile (Flutter)
        │  HTTPS/WS
        ▼
 FastAPI app (main.py) ──┬─▶ PM REST API (routers/pm.py)       — spaces/projects/tasks/…
                         └─▶ Chat (routers/chat.py)             — §10, this file's real subject
                                    │
                                    ▼
                         get_agent(framework) → AgentHandle      — §2.2
                            native | langgraph | adk
                                    │
                                    ▼
                         handle_message(user, session_id, msg)  — §6.1, 17 steps (§2 below)
```

## 2. One chat turn — the 17-step algorithm

```
 0. query_refinement (§14, routers/chat.py — runs ONCE, above every
    track, skipped on a pinned confirmation/clarification reply)
                    │
                    ▼
 1. receive ──▶ 2. load_session_state (sessions.py: messages window,
                  pendingConfirmation, pendingClarification, summary)
                                    │
                    ┌───────────────┼──────────────────────────────┐
                    ▼               ▼                                ▼
        pendingConfirmation?  pendingClarification?          neither pending
        (§9b/11: treat as          (§8: resume, unless          │
         bare yes/no)               looks_like_a_fresh_command)  ▼
                    │               │                    3. input_guard (§12.3)
                    ▼               ▼                            │ fail → finish(refusal)
              execute or        resume_clarification              ▼
              cancel, finish    or fall through            4. classify_task_complexity
                                                             (local, non-LLM: realtime|delay_ok)
                                                                    │
                                                             [Hybrid mode intercept — §7]
                                                                    ▼
                                                            5. select_model
                                                             get_llm_client(tier, preferred)
                                                                    │
                                                             [RAG optimizer — §6, if
                                                              is_rag_relevant(msg)]
                                                                    ▼
                                                            6. perceive_intent
                                                             [Token optimizer — §5, if
                                                              provider is local]
                                                             client.chat(messages, tool_specs)
                                                                    │
                                                                    ▼
                                                       output_guard valid?  ──no──▶ retry once
                                                                    │ yes                │
                                                                    ▼                     │(fail again)
                                                          tool_call present?              ▼
                                                          │no (free text)         finish("couldn't
                                                          ▼                        turn into action")
                                                   out_of_scope flag set,
                                                   finish(text)
                                                          │yes
                                                          ▼
                                              7. resolve_entities (fuzzy-match
                                                 names → real IDs, never LLM-invented)
                                                          │
                                              ambiguous/missing? ──▶ 8. ask_clarifying_question,
                                                          │no          finish(question, clarify=...)
                                                          ▼
                                              9. requires_confirmation(tool)?
                                          (DESTRUCTIVE whitelist OR any MCP tool)
                                                    │yes         │no
                                                    ▼             ▼
                                        10. confirm_with_user   12. tools.execute(user, name, args)
                                            finish(prompt,            (as the CALLING USER, §11)
                                                  pending=...)         │
                                                    │                  ▼
                                          11. next turn: yes/no    13. success/failure
                                                    │                  │
                                                    └──────────────────┤
                                                                       ▼
                                                          14. (on success) side effects already
                                                              broadcast from the service layer
                                                                       ▼
                                                          15. format_result → short NL confirmation
                                                                       ▼
                                        16. persist_session_state (append turn, clear resolved
                                            pending/clarify, trim window, TTL) + finetune_log
                                            (if a real LLM call happened) + finalize every RL
                                            optimizer's deferred reward (§5/§6/§7 below)
                                                                       ▼
                                        17. return AgentReply → WS broadcast → widget
```

## 3. Model tier routing — `llm/router.py`'s fallback chains

```
 TierClient.chat(node, messages, tools)
        │
        ▼
 chain = _effective_chains()[tier]   (+ EXTRA_LLM_PROVIDERS-registered rungs)
        │
        ▼
 preferred set (UI model picker / Hybrid mode)?  ──yes──▶ move that rung to the front
        │ no                                              (still followed by the rest —
        ▼                                                   a bad pick degrades gracefully)
 for each provider in chain, in order:
        │
        ├─▶ optimization_router.select(provider, tier)   — §5.1 generation params
        ├─▶ token_optimizer / RAG already applied upstream if relevant
        ├─▶ provider.chat(messages, tools)
        │        │
        │    success? ──yes──▶ log {node,tier,provider,fallbackUsed,ms,arm} → RETURN
        │        │ no (exception: network/5xx/timeout)
        │        ▼
        │   log the failure, optimization_router.update(ok=False), try NEXT rung
        ▼
 realtime : Luna → Groq → Deepseek(shadow) → Ollama → rule-based
 delay_ok : Max  → Luna → Groq             →  → rule-based
 bulk     : Deepseek → Groq → Ollama       →  → rule-based
 (rule-based floor: deterministic parsing, never errors — the stack
  always answers something even with every hosted rung absent/down)
```

## 4. Agent routing + Shapley credit — `gmaofn/router.py` + `gmaofn/shapley.py`

```
 New task T arrives (task_type, required_capabilities, budget, latency_req)
        │
        ▼
 For each candidate agent a: ContextualThompsonRouter.select()
        │
        ├─ no history yet? → seed posterior mean at utility_prior(a, T)
        │    U(a|T) = w1·Q + w2·R + w3·C + w4·F − w5·Cost − w6·Latency
        │
        ▼
 sample θ_a ~ N(μ_a, ν² B_a⁻¹) for every candidate, argmax_a θ_a·x(T)
        │
        ▼
 chosen agent(s) execute (possibly a MULTI-agent workflow — orchestrator.py)
        │
        ▼
 observed outcome per agent → v(S) = 1 − Π_{i∈S}(1 − quality_i)  (noisy-OR)
        │
        ▼
 Shapley value φ_i(v) — exact (≤ _EXACT_LIMIT agents) or Monte Carlo permutation
 estimate above that — splits the ONE observed task reward fairly across
 however many agents jointly worked it
        │
        ▼
 reward.compute_reward(outcome, weights) → R = α·Quality + β·Success +
   γ·UserFeedback − δ·Cost − ε·Latency − ζ·Failure   (per agent, using its φ_i share)
        │
        ├─▶ router.update(agent, context, R)          — bandit posterior update
        └─▶ AdaptiveWeights.update(outcome, R)         — α..ζ drift toward what
                                                          actually correlates with high R
```

## 5. Token optimization layer — `llm/token_optimizer.py`

```
 TierClient.chat() reaches a provider whose name starts with "ollama"
        │
        ▼
 optimize_tools_for_provider(provider_name, messages, tools)
        │
        ▼
 required = DESTRUCTIVE ∪ every MCP tool name        (guardrail floor —
        │                                              NEVER trimmed, any level)
        ▼
 optional = every other tool, ranked by keyword overlap
            with the user's message (cheap, no LLM call)
        │
        ▼
 TokenOptimizer.select(): LinearBanditCore picks a reduction level
    arms = "full" | "top12" | "top6" | "top3"
    context = [optional_tool_count/40, message_word_count/60]
        │
        ▼
 reduced = required ∪ optional[:keep_n]
        │
        ▼
 provider.chat(messages, reduced)  — stashed in client.tool_selections
 (in-memory only, never in the persisted tier_log)
        │
        ▼
 finish() — once retried/out_of_scope are known:
    reward = (1.0 if success else -0.5) + 0.3 × (tokens_saved / tokens_before)
    TokenOptimizer.record_outcome(selection, success)
        │
        ▼
 surfaced in the UI: "~N% fewer tokens" chip (ChatReply.token_stats)
```

## 6. RAG optimizer — `gmaofn/rag_optimizer.py`

```
 user message
        │
        ▼
 is_rag_relevant(message)?  ── no ──▶ skip retrieval entirely
        │ yes
        ▼
 refine_query_for_intent(message)   — strip filler, keep the keyword core
        │
        ▼
 RagOptimizer.select(): LinearBanditCore over arms "{store}::{algorithm}"
    stores:      pm-index (Chroma, real PM data) | default | any doc-ingested store
    algorithms:  vector | lexical_rerank | hybrid
    context = [optional_arm_count/10, message_word_count/30]
        │
        ▼
 store.search(query, k×3)  →  apply chosen algorithm, truncate to k
        │
        ▼
 build_context_with_included_hits()  — the ONE source of truth for
    "used" vs "considered but truncated by the char budget"
        │
        ▼
 context string → extra system message BEFORE perceive_intent's tool call
 (never replaces or gates it; input_guard/output_guard/requires_confirmation
  run identically regardless)
        │
        ▼
 finish() — once retried/out_of_scope are known:
    n_hits == 0 → reward -0.5 (dominant penalty, regardless of anything else)
    else        → reward = 1.0 if success else -0.2
        │
        ▼
 sources_used surfaced in the UI: "used N sources" chip, hover for titles
```

## 7. Hybrid mode — `llm/hybrid_router.py`

```
 ChatMessageIn.provider == "hybrid" ?
        │ yes                              │ no → pass through unchanged
        ▼
 hybrid_router.choose(message, tier)
    context = [word_count/50, clause_count/4, tier_is_delay_ok]
    arms = "local" | "hosted"
        │
        ▼
    ┌────────────┴────────────┐
    ▼                          ▼
 arm = "local"            arm = "hosted"
 preferred = "ollama"     preferred = None (normal chain already
                           tries hosted rungs before Ollama)
        │                          │
        └────────────┬─────────────┘
                     ▼
        get_llm_client(tier, preferred=...)  — from here on identical
        to picking one specific model; a bad pick still degrades
        through the normal fallback chain
                     │
                     ▼
        finish() — record_outcome:
           local success  → +1.0     hosted success → +0.5
           local failure   → -0.6     hosted failure → -0.3
```

## 8. Document ingestion → summary/prompts — `gmaofn/document_pipeline.py`

```
 folder or file (under RAG_INGEST_DIR, path-traversal guarded)
        │
        ▼
 ingest_path(): walk recursively, chunk_text() every .txt/.md,
    index into a FRESH dedicated store, register into rag_registry
        │
        ├──▶ immediately queryable by the LIVE PM chat too (§6 above)
        ▼
 summarize_document(store) / suggest_prompts(store)
        │
        ▼
 retrieve_with_optimizer(store_names=[store])  — RL picks the best
    ALGORITHM for this one store (no "which store" cold-start cost)
        │
        ▼
 real LLM call, "bulk" tier — grounded summary or N follow-up questions
        │
        ▼
 {"summary"|"prompts", "sources_used", "provider"}
```

## 9. Nightly PEFT fine-tune + regression gate — cron

```
 03:30 cron ──▶ nightly_finetune.sh
        │
        ▼
 1. docker cp ak-chatbot:.../finetune/ollama_turns.jsonl
        │  (no file yet? → exit 0, done)
        ▼
 2. finetune_local_slm.py — LoRA, reward-weighted (see §10 below)
        │  (< min_examples reward≥0.7? → exit 0, done)
        ▼ merged HF model dir
 3. convert_and_register_ollama_model.sh
        │
        ├─ 3a convert_hf_to_gguf.py → f16 GGUF
        ├─ 3b llama-quantize → Q4_K_M GGUF
        ├─ 3c ollama create <new-tag>          (base tag untouched)
        ├─ 3d sanity check: answers "Say OK."?  no → stop, don't register
        ├─ 3e regression_gate.py: new-tag vs base-tag
        │      (§candidate-eval harness, LOCAL ONLY, zero cost)
        │      worse? → stop, don't register
        │      couldn't run? → warn, register anyway this one time
        └─ 3f POST /api/admin/providers → new tag selectable NOW, no restart
        ▼
 4. prune out/ to the last 3 nights
```

## 10. PEFT mechanics — `scripts/finetune_local_slm.py`

```
 finetune_log.py's JSONL (every Ollama turn: messages, response,
 provider, reward = 1.0 − 0.3·retried − 0.5·out_of_scope)
        │
        ▼
 filter: reward ≥ 0.7, most-recent 500
        │
        ▼
 render through the TARGET model's own chat template;
 mask loss to -100 everywhere except the assistant's final turn
        │
        ▼
 LoRA adapters (r=8, alpha=16) on attention+MLP projections
        │
        ▼
 RewardWeightedTrainer.compute_loss():
    weighted_loss = Σ(per_example_loss × reward) / Σ(reward)
        │
        ▼
 train, bounded by WallClockStop (--max-minutes) AND --max-steps
        │
        ▼
 model.merge_and_unload() → one plain HF model directory
```

## 11. Candidate-selection eval pipeline — `scripts/eval/`

```
 eval/test_assets/*.yaml (§A, gold+rubric, 32 items)
        +
 eval/candidates.yaml (§B: local Ollama tags + OpenRouter candidates,
                         hosting: local|cloud explicit)
        │
        ▼
 harness.py (§C) — for each (candidate, item):
    deterministic (exact/regex/schema)  ──or──  llm_judge (judge.py,
                                                  bias guard: never told
                                                  which model answered)
        │
        ▼
 human_sample.py (§D 3rd lane) — ~12% sample, weighted to
    reasoning/planning/complex_reasoning
        │
        ▼
 aggregate.py (§E) — score matrix × eval/weights.yaml (locked in
    BEFORE scoring), latency p50/p95, $/1k requests
        │
        ▼
 decide.py (§F) — ranked top 1-2, cost/latency-adjusted, + manual
    license/vLLM-TGI checklist (deliberately not automated)
        │
        ▼
 regression_gate.py (§G) — same harness, reused nightly (see §9 above)
    for the cheap two-candidate PEFT promotion check
```

## 12. Voice — speech in, speech out, same pipeline

```
 mic audio ──▶ STT (browser/device native, or a hosted STT API)
        │
        ▼
 transcript text ──▶ EXACT SAME handle_message() as typed chat
        │              (voice is a transport, not a separate agent path)
        ▼
 AgentReply.reply ──▶ TTS (browser/device native, or a hosted TTS API)
        │
        ▼
 spoken audio out
```

## 13. MCP tools — external servers, permission-gated

```
 config.py's MCP_SERVERS (utility_server.py bundled example +
 mcp-server-fetch, Anthropic's reference server)
        │
        ▼
 mcp_client.mcp_tool_specs() → merged into TOOL_SCHEMAS at import time
 (perceive_intent and output_guard see them as just more tools)
        │
        ▼
 model picks an MCP tool ──▶ requires_confirmation(name)?
    (tools.DESTRUCTIVE ∪ mcp_client.mcp_tool_names() — EVERY MCP tool
     gets the same "ask permission first" gate a destructive PM action
     gets, since it calls a server this codebase doesn't control/audit)
        │ yes
        ▼
 confirm_with_user (§9 of the 17-step flow) — same UI/flow as a
 destructive PM action, no special-cased "external tool" path;
 confirm_prompt_for() shows ONLY the tool name + arguments, NEVER the
 MCP server's own `description` string (that text targets a model's
 reasoning, not a human's trust decision — the bundled mcp-server-fetch
 reference server's real description tries to talk a model into
 believing it now has internet access despite being told otherwise)
        │ user says yes
        ▼
 tools.execute() dispatches to mcp_client, real MCP server call,
 result formatted exactly like any other tool result
```

## 14. Query refinement layer — `gmaofn/query_refinement.py`

Runs from `routers/chat.py` (§2's step 0), ABOVE every agent track —
native/langgraph/adk all see the same refined text, so this logic lives
in exactly one place rather than duplicated per track (unlike Hybrid
mode, §7, which genuinely differs per track's own LLM call site).

```
 raw message (skipped entirely on a pinned confirmation/clarification
 reply — see §2's step 0 note)
        │
        ▼
 LinearBanditCore.select(["refine","skip"], context)
    context = [word_count/30, filler_phrase_count/3, missing_punctuation]
        │
   ┌────┴──────────────────────────────┐
   ▼ skip                              ▼ refine
 pass through unchanged        memory.lookup(raw) — cosine similarity vs
 (zero LLM cost)               QUANTIZED embeddings of past accepted
                               refinements (§ vector storage optimization,
                               reused here, not re-implemented)
                                        │
                          ┌─────────────┴──────────────┐
                          ▼ hit (sim ≥ 0.93)            ▼ miss
                    reuse cached refined text     ONE LLM call →
                    (zero LLM cost — the cache     {refined,
                    grows the longer the process    missing_clarifications,
                    runs: "refines as it goes")      suggested_queries}
                                                       │
                                         embedding_similarity(raw, refined)
                                         — HashEmbedder cosine, NOT a
                                           second LLM call
                                                       │
                                          ┌────────────┴────────────┐
                                          ▼ sim < 0.6               ▼ sim ≥ 0.6
                                    revert to raw              keep refined text
                                    drift_log.record()         memory.remember()
                                    (never remembered)          (quantized, for
                                                                  future lookups)
        │
        ▼
 record_outcome() — once out_of_scope is known (chat.py, after the
 agent track returns):
    skip:        +1.0 / −0.6      memory hit:  +1.0 / −0.1
    llm error:   −0.5             no-op refine (text unchanged): −0.3
    real refine: +1.0 / −0.2
        │
        ▼
 ChatReply.refinement (original/refined text, was_repaired,
 missing_clarifications, suggested_queries, needs_confirmation) —
 surfaced as a UI chip; never blocks the turn (a chat API has no
 synchronous confirm/edit round trip mid-message)
```

Two RL/embedding mechanisms compound here, not one: the bandit gate
learns whether refining is worth an LLM call AT ALL for a given kind of
message; `RefinementMemory` separately learns WHICH SPECIFIC queries
already have a known-good answer. `GET /api/chat/refinement-audit`
(Admin-only) lists every drifted case since the process started.

