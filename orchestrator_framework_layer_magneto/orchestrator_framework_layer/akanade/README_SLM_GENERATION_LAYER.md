# The SLM Generation Layer

ALGO AKANADE's model layer isn't just "call an LLM" — one rung in it,
Ollama, is a model ALGO AKANADE's own process actively *produces and
improves*, not just calls. That whole subsystem — collecting what the
model should learn, training it, quantizing it, exporting it, gating
it, and registering it back into the router every agent already
uses — is the **SLM Generation Layer**. This document is the single
place that says what's already real and running (§1) vs. what this
pass specifically added (§2), then how the result plugs into the
orchestrator (§3) and how to experiment with the idea without the full
nightly pipeline (§4). Deep mechanics for each stage are in
`README_SLM_PIPELINE.md`; this file is the layer-level map.

## 1. What's already used — the real, running pipeline

Everything below already exists, is wired together, and (per
`README_SLM_PIPELINE.md`'s own honesty section) is at minimum
syntax-verified and smoke-tested, with the pure-Python pieces
fully unit-tested:

| Stage | Where | What it does |
|---|---|---|
| Collect enterprise prompts | (you — a curated `list[str]`) | The deliberate questions the business actually wants the model to get right |
| Create ground truth | `app/framework/ground_truth.py`, `POST /api/v1/rag/ground-truth` | RAG-grounded (optional) or general-knowledge LLM answer per prompt, reviewed, logged at reward `1.0` |
| Data collection (incidental) | `app/agent/finetune_log.py` | Every real Ollama chat turn + hybrid-escalation turn + document Q&A turn, reward-scored |
| Data Engine | `scripts/data_engine.py` | PII scrub, cross-night dedup (`DataEngineStore`), curation |
| Training | `scripts/finetune_local_slm.py` | Reward-weighted LoRA/QLoRA fine-tune, optional pruning, `merge_and_unload()` |
| Hyperparameter selection | `scripts/peft_hyperparam_optimizer.py` | Bandit-picked LoRA rank/alpha/lr arm, learns from real gate outcomes |
| Conversion + export | `scripts/convert_and_register_ollama_model.sh` | GGUF conversion, Q4_K_M quantization, standalone artifact export |
| Gate | `scripts/eval/regression_gate.py` | New tag vs. base tag, same harness §8.4 of `ARCHITECTURE.md` uses to pick a base model |
| Registration | `POST /api/admin/providers` | Splices the new tag into the live router's fallback chains, no restart |
| Scheduling | `deploy/nightly_finetune.sh` (cron) | Wraps all of the above, prunes old training checkpoints |

## 2. What this pass adds

Three concrete, real additions on top of the pipeline above:

- **The "Collect enterprise prompts" / "Create ground truth" stages
  are now real code**, not just incidental chat logging — see the
  table above. Previously the only way data entered the corpus was
  organic usage or an ingested document; now you can also hand the
  pipeline the exact questions you want it to learn, get back a
  reviewable answer, and only log it once you (or an admin) trust it.
- **A genuine standalone export.** Before this, the quantized GGUF
  only ever existed inside a deleted temp directory and Ollama's own
  opaque blob store — there was no artifact you could actually copy to
  another machine, back up, or load into a different GGUF-capable
  runtime. `convert_and_register_ollama_model.sh` now copies the GGUF
  + the Modelfile that produced it + a `metadata.json` (base tag,
  quant type, sanity-check/gate outcome, whether it shipped) to
  `exported_models/<tag>/`, unconditionally — even a candidate that
  fails the gate stays there for inspection instead of vanishing.
- **One consistent product name: Algo-Akanade-SLM.** The output is no
  longer named after whichever base model happened to produce it
  (`qwen2.5-tuned`) — every base (Qwen, SmolLM, Llama, ...) that goes
  through `finetune_local_slm.py --base-model ...` produces a tag
  under the same `algo-akanade-slm:<date>` name and the same
  `"algo-akanade-slm"` registered provider.
- **A quickstart LoRA example** (`scripts/quickstart_lora_example/`,
  §4 below) for trying the "train a LoRA adapter on your own data"
  idea in minutes on a laptop GPU or a free notebook, without needing
  the full Docker/llama.cpp/regression-gate machinery running first.
- **One deterministic daily version, not an accumulating pile of
  identically-named ones.** Every earlier night's `POST
  /api/admin/providers` call created a brand-new DB row, all sharing
  the literal name `"algo-akanade-slm"` — which night's row actually
  answered chat turns depended on the DB scan's incidental return
  order, and every prior night's row (and its now-superseded Ollama
  tag) lingered forever. The script now looks up the existing
  `"algo-akanade-slm"` provider by name first and **PATCHes** its
  `model` field to tonight's tag instead — exactly one row ever exists,
  deterministically pointing at the latest candidate that passed the
  gate. First day: base Ollama tag answers (nothing registered yet).
  First successful night: PEFT produces a real candidate, the gate
  passes, `algo-akanade-slm` is created pointing at it. Every
  successful night after that: the SAME provider is updated in place —
  "then uses this one" going forward, not a second competing entry.
- **Local version history + Admin dashboard sync**
  (`app/routers/admin_slm.py`, `POST`/`GET /api/admin/slm-versions`).
  Every run — success or failure — reports its outcome (tag, base tag,
  sanity check, regression gate result, whether it was actually
  registered) to a local, inspectable version log (`store("SlmVersions")`,
  a plain JSON file under `DATA_DIR` when running the default local
  store, matching how every other table in this app persists). The
  Admin > AI Models page reads it and renders a version-history table
  underneath the provider list, newest first, so an operator can see at
  a glance which version is live right now and what happened on prior
  nights — without SSHing into the deploy box to read `exported_models/`
  or `ak-finetune.log` by hand.

Mapping the rest of what's sometimes proposed for a setup like this
onto what's already here, so nothing gets rebuilt twice:

| Proposed idea | Already covered by |
|---|---|
| Evaluation → Reward | `finetune_log.py`'s `compute_reward()` (retried/out-of-scope penalties) |
| Ground Truth → Training Dataset | §2's new `ground_truth.py` + `data_engine.py`'s curation |
| Nightly (Q)LoRA job | `scripts/finetune_local_slm.py`, scheduled by `deploy/nightly_finetune.sh` |
| Automatic Benchmark | `scripts/eval/regression_gate.py` (§8.4 of `ARCHITECTURE.md`) |
| Canary deployment | The base Ollama tag is never touched or removed; the new tag only reaches `POST /api/admin/providers` if it beats the base tag on the gate — production never sees an unvetted candidate |
| New Adapter → Router | `POST /api/admin/providers` → `llm/router.py`'s `_effective_chains()` (§3 below) |

## 3. How this connects to the orchestrator

Registering the trained model isn't a side channel — it's a real rung
in the same chains every agent already calls through:

```
   exported_models/<tag>/  (this layer's standalone artifact)
              │
              ▼
   ollama create <tag>          (Ollama-local, base tag untouched)
              │
              ▼
   regression_gate.py: better than base tag?  ── no ──▶ stop, stays local-only
              │ yes
              ▼
   "algo-akanade-slm" provider already exists?
     yes ─▶ PATCH /api/admin/providers/<id> {"model": "<tag>"}   (upsert —
     no  ─▶ POST /api/admin/providers {...}                       ONE row,
                                                                   ever)
              │
              ▼
   POST /api/admin/slm-versions  ── recorded either way, success or
                                     failure — the Admin dashboard's
                                     version history (§2)
              ▼
   llm/router.py _effective_chains()  ── splices the new provider into
              │                          the SAME realtime/bulk fallback
              │                          chains every tier already uses
              ▼
   get_llm_client(tier, preferred="algo-akanade-slm")   ◀── any of
              │                                              ALGO AKANADE's 9
              ▼                                              default agents
   TierClient.chat(node, messages)  ── identical call site whether the
                                        answering model is Groq, Claude,
                                        or Algo-Akanade-SLM
```

No orchestrator-side code changes, no redeploy, no restart — the
Adaptive Router (`ARCHITECTURE.md` §3), Agent Network, and Workflow
Engine above this layer are completely unaware of *which* concrete
model answered a given turn; they only see `LLMResult`. This layer's
whole job is making sure that when Algo-Akanade-SLM DOES answer, it's
already been proven at least as good as what it's replacing.

## 4. Quickstart: `scripts/quickstart_lora_example/`

A minimal, single-file way to see the core idea work end to end —
takes an open base model (`Qwen/Qwen3-0.6B` by default) and trains a
LoRA adapter on a handful of hand-written examples, in minutes on a
laptop GPU or a free Colab/Kaggle notebook:

```
scripts/quickstart_lora_example/
├── data/train.jsonl
├── train.py
├── inference.py
└── requirements.txt
```

```bash
cd scripts/quickstart_lora_example
pip install -r requirements.txt
python train.py        # writes ./algo-akanade-slm-quickstart/ (LoRA adapter)
python inference.py    # loads base model + adapter, runs one prompt
```

This is deliberately **separate** from the production pipeline: it
never touches `finetune_log.py`, `data_engine.py`, Ollama, or the live
chatbot, and for a real base model needs thousands of examples, not
four. It exists so you can iterate on a data format or sanity-check an
idea fast. To carry a real result forward into an actual
router-selectable model, merge the adapter into a full HF model
directory the way `finetune_local_slm.py`'s own `merge_and_unload()`
does, then hand that directory to
`scripts/convert_and_register_ollama_model.sh` — from there it goes
through the same export, gate, and registration steps as any nightly
candidate.

## 5. Domain-specific SLMs — the RL orchestrator that picks between them

Everything above produces ONE general-purpose Algo-Akanade-SLM. Real
deployments often want several — a legal-support model, a customer-
support model, a coding model — each fine-tuned on its own domain's
data via the exact same pipeline. `llm/domain_router.py` is the small
orchestrator that sits in front of however many of those are actually
registered and, per message, RL-picks which one is most likely to
answer it well.

**Producing a domain SLM.** No new pipeline — run the existing one with
a domain-filtered training log and set `DOMAIN` when registering:

```bash
# finetune_local_slm.py already accepts --log-path — point it at a
# log containing only that domain's turns (however you produce that
# filtered JSONL is up to you; the training/quantize/gate machinery
# doesn't change).
python scripts/finetune_local_slm.py --log-path legal_turns.jsonl --output-dir out/legal ...

DOMAIN=legal scripts/convert_and_register_ollama_model.sh out/legal/merged
# -> tag "algo-akanade-slm-legal:<date>", registered as provider
#    "algo-akanade-slm-legal" — same export/gate/upsert-by-name/
#    version-history steps as §1-§3, just domain-namespaced throughout
#    (exported_models/, the Ollama tag, the provider name, and the
#    /api/admin/slm-versions record's "domain" field all agree).
```

**Routing between them.** `ChatMessageIn.provider == "domain"` (a
fourth UI-selectable option, alongside Auto/a specific model/Hybrid —
see `ChatAssistant.tsx`'s model picker) intercepts the same way Hybrid
mode does, in all three agent tracks (native/ADK/LangGraph):

```
 domain_router.registered_domains()  ── reads live from
        │                               available_providers(), by naming
        │                               convention "algo-akanade-slm-<domain>"
        ▼
 no domains registered? ──▶ arm = "general" ──▶ preferred=None, normal chain decides
        │ yes
        ▼
 arm_ids = ["general", <every registered domain>]
        │
        ▼
 for each arm: LinearBanditCore.score(arm, [word_count, clause_count, tier])
        +  TaskMemory.bonus_for(message, arm)   ── embedding similarity to
        │                                          past (query, domain, reward)
        │                                          triples — THE real signal,
        │                                          since domain relevance is
        ▼                                          about content, not length
 argmax  ──►  winner (e.g. "legal")
        │
        ▼
 preferred_provider = "algo-akanade-slm-legal"  ──►  TierClient.preferred
        │                                             (degrades through the
        ▼                                             normal chain on failure,
 same LOCAL_TIMEOUT_SECONDS escalation as Hybrid       exactly like Hybrid)
        │
        ▼
 record_outcome(success)  ──►  updates BOTH the bandit's posterior for
                                that arm AND the embedding memory — the
                                next similar query routes better
```

Why this ISN'T just `hybrid_router.py`'s word/clause-count features
reused: local-vs-hosted genuinely correlates with message complexity,
but "which domain is this about" is a question about content, which a
word count can't answer. `framework/task_memory.py` — the same
embedding-similarity memory `router.py`'s own agent-selection bandit
uses to warm-start a brand-new agent — is reused here as the PRIMARY,
every-call signal instead of a one-time seed, because a domain SLM has
no structured per-arm signal (like agent routing's `task_type`) the
way agent selection does. Cold start (no memory yet) degrades to
picking among domains roughly at random, same as any bandit; it
sharpens as real `record_outcome()` calls accumulate.

**Not yet done, by design, to keep this addition scoped**: an
escalation from a domain SLM to a hosted model isn't logged as extra
fine-tune training data the way Hybrid mode's `log_hybrid_escalation_
turn()` does — worth adding later if domain routing sees real usage,
but it would need its own reward/labeling story rather than reusing
Hybrid's, since "the hosted model beat THIS domain SLM" isn't quite
the same claim as "the hosted model beat the general local SLM."

## 6. Configuring it from the UI, and running it without a host crontab line

Two more pieces close the loop between "here's the pipeline" and
"anyone can actually operate it":

**`POST`/`GET`/`PUT`/`DELETE /api/admin/finetune-config`**
(`app/routers/admin_finetune_config.py`) — what SHOULD get trained
tonight, one row per domain (`None` = the general-purpose model),
editable from Admin > AI Models > "Algo-Akanade-SLM nightly config" in
the UI instead of hand-editing `deploy/finetune.env` or shell-script
env vars on the deploy box. Each row is: a domain, any HuggingFace
`base_model` id, the `ollama_base_tag` it gets gated against, a
`max_minutes` training budget, and `enabled`. `PUT` is a genuine
upsert keyed by domain (`"general"` for the general-purpose row) —
there is at most one config per domain, mirroring
`convert_and_register_ollama_model.sh`'s own "one provider per domain,
ever" invariant. `deploy/nightly_finetune.sh` now fetches every
`enabled` row from `GET .../enabled` at the start of each run and
loops over them — an unconfigured deployment (nobody's touched the UI
yet, or the admin API is unreachable) falls back to exactly ONE run
using the script's own `FINETUNE_BASE_MODEL`/`OLLAMA_BASE_TAG` env
vars, the pre-multi-domain behavior, so nothing breaks for an existing
single-model deploy.

**`deploy/nightly_cron_daemon.py` + `deploy/Dockerfile.finetune-cron`**
— the no-crontab alternative to `nightly_finetune.sh`'s "add a line to
host crontab" one-time-setup step. A tiny, dependency-free (stdlib
only) Python scheduler that runs INSIDE its own always-on container
(`ak-finetune-cron`) and fires `nightly_finetune.sh` itself at the
configured time every day — the same script, unchanged, just triggered
by a container's own loop instead of a host cron line someone has to
remember to re-add if the box ever gets rebuilt:

```bash
docker build -t algo-kaizen-finetune-cron -f deploy/Dockerfile.finetune-cron .
sudo docker run -d --name ak-finetune-cron --network ak-net --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /opt/algo-kaizen:/opt/algo-kaizen \
  algo-kaizen-finetune-cron
```

**What this does NOT eliminate**: the exact same host-level
prerequisites `nightly_finetune.sh`'s own docstring already lists — a
real llama.cpp checkout + build, the `algo-kaizen-finetune` Docker
image already built, `finetune.env` filled in. Those are genuine
native-toolchain/credential requirements, not something a container
image can conjure away; this only removes the "remember to add a
crontab line" step, and makes WHICH model(s) get trained tonight a UI
concern instead of a shell one. The container needs the Docker socket
bind-mounted (same "run docker commands against the host's daemon"
pattern `nightly_finetune.sh` already uses via `sudo docker`) and
`/opt/algo-kaizen` mounted at the SAME path as the host — every `-v`
flag these scripts build gets resolved by the HOST's dockerd, not this
container's filesystem, so the paths have to agree; see the
Dockerfile's own comment for why. Verified: builds clean, and
`python3 /app/nightly_cron_daemon.py --now` correctly invokes
`nightly_finetune.sh`, which correctly no-ops (exit 0, "no data yet")
when `ak-chatbot` isn't reachable — the real end-to-end nightly run
against production data still awaits the first scheduled fire on a
real deploy box.

## 7. Serving directly, without Ollama at all

Every stage above still needs SOME runtime to actually answer chat
turns with the exported GGUF. Ollama is the default and the only one
this pipeline used to support — but Ollama is a convenience layer, not
a requirement: GGUF conversion (§3 of `README_SLM_ALL.md`) is already
model-family-agnostic, and llama.cpp's own server implements the exact
same OpenAI-compatible `/v1/chat/completions` shape `OpenAICompatible`
(`app/llm/providers.py`) already expects from ANY provider. So a
second `serve_backend` choice exists per `FinetuneConfig` row —
**"ollama"** (default, byte-for-byte the same behavior as before) or
**"llama_server"** (skip Ollama entirely, run the GGUF directly) —
editable from the exact same Admin UI form as every other config field,
or from the shell by exporting `SERVE_BACKEND=llama_server` before
calling `convert_and_register_ollama_model.sh`.

**Which one should a given domain actually use?** Concretely, for
"I want to add Qwen2.5-Coder as a coding-domain model": use **"ollama"**
(the default) — `ollama pull qwen2.5-coder:1.5b` once, then a
`FinetuneConfig` row with `domain: "coder"`, that tag as
`ollama_base_tag`, `serve_backend` left as `"ollama"`. Ollama already
ships an official template for the Qwen2.5-Coder family, so there's no
reason to bypass it. Reach for `"llama_server"` only when Ollama
DOESN'T have good template support for the model you want (an obscure
fine-tune, a brand-new architecture, a model you only have as a raw
GGUF already), or when you deliberately want a domain to run
completely independent of the shared `ak-ollama` daemon. Both modes
produce the exact same exported artifact (§3 of `README_SLM_ALL.md`)
and go through the exact same training/data/gate pipeline — this
choice ONLY affects how the final candidate gets served.

**`deploy/Dockerfile.llama-server`** builds llama.cpp's own
`llama-server` from source (no external registry image, no
uncertainty about a public tag — reproducible from the same upstream
repo the rest of this pipeline already depends on for `llama-quantize`).
Build once: `docker build -t algo-akanade-server -f
deploy/Dockerfile.llama-server .`

**Container lifecycle mirrors the Ollama path's own safety property,**
just expressed differently — an Ollama tag is immutable disk storage
until asked to run, so "old tag untouched" is automatic; a running
server isn't, so this mode uses TWO containers per domain instead of
one tag: a stable `ak-slm[-<domain>]` (the currently-live one) and a
throwaway `ak-slm[-<domain>]-candidate` for tonight's not-yet-proven
build. The candidate is health-checked and sanity-checked exactly like
an Ollama tag is; the regression gate scores the CURRENTLY-LIVE
container against the candidate (two separate servers, not two tags on
one server — `regression_gate.py` gained a `--new-base-url` flag
specifically for this, defaulting to `--ollama-base-url` so the normal
Ollama path is completely unaffected); only on a pass does the
candidate get renamed into the live slot, with the old one removed —
a failed candidate is discarded outright (unlike a failed Ollama tag,
which stays around inert; a full model SERVER left running forever for
every failed candidate would be a real resource leak a dormant Ollama
tag never was).

**A real bug this surfaced and fixed during development**: the first
version of `Dockerfile.llama-server` copied only the `llama-server`
binary into the runtime image — it failed to even start
(`error while loading shared libraries: libllama-server-impl.so`)
because llama.cpp's CMake build produces `llama-server` as a thin
binary linked against several separate `.so` files
(`libggml*`, `libllama*`, `libmtmd`, `libllama-server-impl`), plus
needed `libgomp1` (OpenMP) at the OS package level. Fixed by copying
the entire `build/bin/` directory and setting `LD_LIBRARY_PATH`.
**Verified for real**, not just by inspection: built the image, ran
`llama-server --help` successfully, downloaded a real (tiny,
1.2MB, `ggml-org/models`'s `tinyllamas/stories260K.gguf`) public GGUF,
ran it in two separate containers, and ran `regression_gate.py` with
`--new-base-url` against both — a genuine PASS, real per-item scores,
a real `eval_report.json` written — confirming the exact mechanism
`convert_and_register_ollama_model.sh`'s `llama_server` branch depends
on actually works end to end, not just that the shell logic parses.

## 7a. Step by step: adding a model to YOUR live `ak-*` server

Concrete steps on the actual deploy box (the one running `ak-backend`/
`ak-chatbot`/`ak-frontend`/`ak-ollama`/`ak-dynamo`), not a conceptual
overview — pick the Ollama or Direct path depending on the model.

**Path A — Ollama (the default; use this unless the model needs Path B):**

1. Pull the base tag into the running Ollama container:
   `sudo docker exec ak-ollama ollama pull <tag>` (e.g. `qwen2.5-coder:1.5b`).
2. In the chat widget, open Admin > AI Models > "Algo-Akanade-SLM
   nightly config" and click **+ Add config** (or edit the `general`
   row to switch what it trains). Set: `domain` (blank for
   general-purpose, or a name like `coder`), `base_model` (the
   HuggingFace id, e.g. `Qwen/Qwen2.5-Coder-1.5B-Instruct`),
   `ollama_base_tag` (the tag from step 1), `serve_backend: "ollama"`,
   `max_minutes`, `enabled: true`.
3. Trigger tonight's run: either wait for the scheduled cron/
   `ak-finetune-cron` fire, or run it right now —
   `sudo /opt/algo-kaizen/nightly_finetune.sh` (or, inside the
   `ak-finetune-cron` container: `python3 /app/nightly_cron_daemon.py --now`).
4. Watch it happen: `tail -f /opt/algo-kaizen/logs/ak-finetune.log`.
5. Confirm it's live: Admin > AI Models shows the new row in
   "Algo-Akanade-SLM versions" with **Live now**, and a matching row in
   "Nightly verification reports" with the per-prompt scores. The
   provider (`algo-akanade-slm[-<domain>]`) is now selectable in the
   chat widget's model picker, and Domain mode routes to it
   automatically if it's a named domain.

**Path B — Direct llama.cpp serving (only when Ollama doesn't template
the model well, or you want a domain fully independent of `ak-ollama`):**

1. One-time: build (or pull) the server image on the deploy box —
   `sudo docker build -t algo-akanade-server -f deploy/Dockerfile.llama-server .`
   (or just run `update_containers.sh` again — it now pulls
   `dockerhubalgo/algo-akanade-server:latest` and re-tags it
   automatically, best-effort).
2. Same config screen as Path A, but set `serve_backend` to **"Direct
   (llama.cpp server)"** instead. `ollama_base_tag` still needs to be a
   real Ollama tag — it's what the FIRST gate comparison for a NEW
   domain would use if one were needed, and keeps the field meaningful
   if you ever switch a domain back to Ollama later.
3. Same trigger step as Path A (step 3 above) — the pipeline is
   identical up through training; only the serving step differs.
4. Confirm it's live: `sudo docker ps` shows a new `ak-slm-<domain>`
   container (not `ak-ollama`) actually running, and Admin > AI Models
   shows the same "Live now"/verification-report rows as Path A — the
   dashboard doesn't distinguish which backend served it, only that it
   did.

Either path, the safety guarantee is the same: nothing already serving
is touched until the new candidate actually passes the regression gate
(§5 of `README_SLM_ALL.md`).

## What's verified vs. what still needs the real deploy box

Same standard `README_SLM_PIPELINE.md` holds itself to:

- **Fully test-verified**: `ground_truth.py`'s grounded/ungrounded
  answer generation and its `log_ground_truth_turn()` wiring
  (`tests/test_ground_truth.py`, 12 tests), and the version-history API
  (`tests/test_admin_slm.py`, 4 tests: role gate, record + list,
  a failed night still gets recorded, newest-first ordering) — no live
  network dependency, LLM calls monkeypatched. The Admin dashboard's
  version-history table (`AdminAIModels.tsx`) type-checks clean
  (`tsc -b`) against the same `SlmVersion` shape the backend returns,
  including its `domain` field. `domain_router.py`'s RL orchestrator
  (`tests/test_domain_router.py`, 8 tests: naming-convention discovery,
  general-arm fallback with zero domains registered, a registered
  domain is a reachable pick, provider-key mapping, the bandit AND the
  embedding memory both update on `record_outcome()`, a past failure
  lowers a similar future query's bonus, and the bandit measurably
  learns to prefer whichever arm keeps winning) — no live network
  dependency, `available_providers()` monkeypatched. The nightly-config
  API (`tests/test_admin_finetune_config.py`, 6 tests: role gate,
  upsert-in-place by domain, domain/key mismatch rejected, enabled-only
  filtering, delete) and its Admin UI form (`AddFinetuneConfigModal.tsx`)
  type-check clean against the same shape the backend returns, including
  its `serve_backend` field. The verification-report API
  (`tests/test_admin_eval_reports.py`, 6 tests: role gate, record +
  list, list is summary-only while GET .../{id} has full items, a
  failed gate is still recorded, domain round-trips, newest-first) and
  its dashboard table (with an expandable per-item view) type-check
  clean the same way.
- **Docker-build-verified AND run-verified against real servers**:
  `deploy/Dockerfile.finetune-cron` builds clean, and `python3
  /app/nightly_cron_daemon.py --now` inside it correctly invokes
  `nightly_finetune.sh`, which correctly no-ops when `ak-chatbot`/the
  Docker socket aren't reachable. `deploy/Dockerfile.llama-server`
  builds clean AND was run for real (see §7 above): two real
  llama-server containers serving an actual GGUF, `regression_gate.py
  --new-base-url` scoring them against each other, a real PASS with a
  real per-item report. Not yet run: either container as its own
  long-lived service on a real deploy box's `ak-net` (i.e. THIS specific
  local test used a temporary test network + published host ports,
  not the ak-net-internal-only setup `convert_and_register_ollama_
  model.sh` actually builds).
- **Syntax-verified, not yet run end-to-end on the real deploy box**:
  the export step in `convert_and_register_ollama_model.sh`, the
  upsert-by-name provider registration + `/api/admin/slm-versions`
  reporting it now does (needs a real llama.cpp checkout + at least two
  real nightly runs to confirm the SECOND night actually PATCHes the
  first night's row rather than creating a new one, and that
  `exported_models/` accumulates correctly across nights), the whole
  `SERVE_BACKEND=llama_server` SHELL ORCHESTRATION specifically (the
  underlying serve+gate mechanism is now real-verified per §7, but the
  candidate/promote/rename container lifecycle in the script itself
  still needs a real nightly run to confirm it end to end), and the
  `DOMAIN` env var end to end (needs two actually-registered domain
  SLMs on a real deploy box to confirm Domain mode routes between them
  live, not just in the monkeypatched unit tests). The quickstart
  example needs a real GPU to actually train — no CI coverage for it,
  by design, since it's a learning aid, not part of the tested
  production path.
