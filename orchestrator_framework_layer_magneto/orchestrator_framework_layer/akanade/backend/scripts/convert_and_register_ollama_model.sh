#!/usr/bin/env bash
# Stage 3 of the nightly local-SLM fine-tuning pipeline (stage 1:
# agent/finetune_log.py's data collection; stage 2:
# scripts/finetune_local_slm.py's LoRA training + merge). Takes the
# merged HF model directory stage 2 produced, converts it to a
# quantized GGUF, creates a NEW Ollama model tag from it (the existing
# base tag is never touched or removed), runs it through the
# candidate-selection pipeline's regression gate (eval/regression_gate.py
# — same test-asset harness as scripts/eval/, §G of the benchmarking
# architecture) against the current base tag, and only if it actually
# scores better registers that new tag with the live chatbot via its own
# admin-providers API — no restart, same "takes effect on the next chat
# turn" mechanism the Admin AI Models UI already uses for any other
# provider. A worse fine-tune never gets promoted.
#
# Requires llama.cpp checked out + built on this machine (its Python
# conversion script + the compiled llama-quantize binary) — a real,
# standard prerequisite for ANY Ollama custom-fine-tune workflow, not
# something pip-installable. Set LLAMA_CPP_DIR to that checkout. Also
# requires two Docker images already built: algo-kaizen-finetune
# (deploy/Dockerfile.finetune — the regression gate, §5, runs inside
# it) and algo-kaizen-ggufconvert (deploy/Dockerfile.ggufconvert — the
# GGUF conversion step, §1, runs inside it; kept as a SEPARATE image
# from algo-kaizen-finetune because llama.cpp pins specific
# transformers/numpy/gguf versions that can drift from that image's
# own, newer deps).
#
# Validated end-to-end on a real deploy box (worker-node, 2026-08-24):
# training was already smoke-tested; this script's own stages (Ollama
# tag creation, the regression gate, admin-API registration) ran for
# real against a real llama.cpp checkout + build on that box. GGUF
# conversion + a real multi-hour run against the actual production
# 1.5B/3B model specifically are the two pieces still awaiting the
# first real nightly cron run — check ak-finetune.log after it. Every
# step below fails loudly and stops rather than silently producing a
# broken model tag or touching anything already serving.
set -euo pipefail

MERGED_DIR="${1:?Usage: $0 <merged-model-dir> [new-tag]}"
# "algo-akanade-slm" (Algo-Akanade-SLM), not a base-model-specific name like "qwen2.5-tuned" —
# this pipeline supports any small base (Qwen, SmolLM, Llama, ...) via
# finetune_local_slm.py's own --base-model, so the OUTPUT gets one
# consistent product name regardless of which base produced it.
#
# DOMAIN (optional): running this SAME script against a merged model
# that was fine-tuned on DOMAIN-SPECIFIC data (e.g. only legal-team
# turns, filtered into their own finetune_local_slm.py --log-path) and
# setting DOMAIN=legal produces "algo-akanade-slm-legal" instead of
# the plain "algo-akanade-slm" — llm/domain_router.py auto-discovers
# any provider matching that "algo-akanade-slm-<domain>" naming
# convention and RL-routes each chat message to whichever registered
# domain SLM its own memory of past queries says handles that KIND of
# question best (README_SLM_GENERATION_LAYER.md §5). No code change
# needed to add a new domain — train it, run this script once with
# DOMAIN set, and it's selectable on the very next chat turn.
# SERVE_BACKEND (optional): "ollama" (default, unchanged behavior) or
# "llama_server" — skip Ollama entirely and serve the exported GGUF
# directly with llama.cpp's own OpenAI-compatible server
# (deploy/Dockerfile.llama-server). Genuinely "any SLM": GGUF
# conversion (steps 1-3 below) is already model-family-agnostic, so
# this isn't limited to whatever Ollama happens to ship a Modelfile
# template for. See README_SLM_GENERATION_LAYER.md §5.
SERVE_BACKEND="${SERVE_BACKEND:-ollama}"
DOMAIN="${DOMAIN:-}"
PROVIDER_NAME="algo-akanade-slm${DOMAIN:+-$DOMAIN}"
NEW_TAG="${2:-$PROVIDER_NAME:$(date -u +%Y%m%d)}"
DOMAIN_JSON="null"
[[ -n "$DOMAIN" ]] && DOMAIN_JSON="\"$DOMAIN\""
BASE_TAG="${OLLAMA_BASE_TAG:-qwen2.5:1.5b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
# llama-quantize is a HOST-built glibc binary (confirmed via `file`:
# interpreter /lib64/ld-linux-x86-64.so.2) but this script itself may
# run inside ak-finetune-cron, an Alpine/musl container — Alpine's own
# gcompat shim gets close but hits a hard `__res_init` symbol gap it
# can't fill (a documented gcompat limitation). Rather than execute the
# binary natively, run it inside a real glibc container, same "bind-mount
# the artifact into a purpose-built image" pattern step 1/7 already uses
# for GGUF conversion. See deploy/Dockerfile.llamaquantize for the full
# story and how this image was validated.
LLAMAQUANTIZE_IMAGE="${LLAMAQUANTIZE_IMAGE:-algo-kaizen-llamaquantize}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
FINETUNE_IMAGE="${FINETUNE_IMAGE:-algo-kaizen-finetune}"
# convert_hf_to_gguf.py needs torch + a specific transformers/numpy/gguf
# pin (llama.cpp's own requirements/) — a real Python 3.9+ dependency
# set, same story as scripts.finetune_local_slm.py, so it runs in its
# own Docker image rather than assuming the host's system Python has
# any of that (see deploy/Dockerfile.ggufconvert — kept separate from
# FINETUNE_IMAGE since llama.cpp's pins can drift from that image's
# newer deps). llama-quantize itself is a compiled native binary with
# no such dependency — it stays a plain host-side call.
GGUFCONVERT_IMAGE="${GGUFCONVERT_IMAGE:-algo-kaizen-ggufconvert}"
DOCKER="sudo docker"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Deploy layout is .../finetune/scripts/<this file> + .../finetune/finetune.env
# (one level up) — see deploy/nightly_finetune.sh's setup instructions.
FINETUNE_ENV_FILE="${FINETUNE_ENV_FILE:-$SCRIPT_DIR/../finetune.env}"
# Without this, the quantized GGUF only ever existed inside $WORK_DIR
# (deleted by the trap below) and Ollama's own opaque blob store —
# there was no actual standalone artifact a person could hand to
# another machine, back up, or load straight into llama.cpp/LM Studio
# without this exact Ollama install. Every night's Algo-Akanade-SLM
# candidate gets its own dated subdirectory here — GGUF + Modelfile +
# a metadata.json recording provenance and whether it actually shipped
# — regardless of whether the regression gate passed, so a failed
# candidate is still inspectable, not just discarded.
EXPORT_DIR="${EXPORT_DIR:-$SCRIPT_DIR/../exported_models}"

if [[ ! -x "$LLAMA_QUANTIZE_BIN" ]]; then
  echo "llama-quantize binary not found/executable at $LLAMA_QUANTIZE_BIN (set LLAMA_QUANTIZE_BIN)." >&2
  echo "Build it once from a llama.cpp checkout: cmake -B build && cmake --build build --target llama-quantize -j" >&2
  echo "(only that target — 'cmake --build build' with no --target can fail on llama.cpp's optional" >&2
  echo "web-UI sub-target on an older cmake; llama-quantize itself doesn't need it.)" >&2
  echo "Skipping tonight — the merged model is still at $MERGED_DIR, nothing already serving was touched." >&2
  exit 0
fi
if ! $DOCKER image inspect "$LLAMAQUANTIZE_IMAGE" >/dev/null 2>&1; then
  echo "$LLAMAQUANTIZE_IMAGE image not built yet — build it once from the repo root:" >&2
  echo "  docker build -t $LLAMAQUANTIZE_IMAGE -f deploy/Dockerfile.llamaquantize ." >&2
  echo "Skipping tonight — the merged model is still at $MERGED_DIR, nothing already serving was touched." >&2
  exit 0
fi
if ! $DOCKER image inspect "$GGUFCONVERT_IMAGE" >/dev/null 2>&1; then
  echo "$GGUFCONVERT_IMAGE image not built yet — build it once from the llama.cpp checkout:" >&2
  echo "  cp deploy/Dockerfile.ggufconvert <llama.cpp checkout>/ && cd <llama.cpp checkout> && docker build -t $GGUFCONVERT_IMAGE -f Dockerfile.ggufconvert ." >&2
  echo "Skipping tonight — the merged model is still at $MERGED_DIR, nothing already serving was touched." >&2
  exit 0
fi
if ! $DOCKER image inspect "$FINETUNE_IMAGE" >/dev/null 2>&1; then
  echo "$FINETUNE_IMAGE image not built yet — build it once: docker build -t $FINETUNE_IMAGE -f deploy/Dockerfile.finetune gmaofn/backend" >&2
  echo "Skipping tonight — the merged model is still at $MERGED_DIR, nothing already serving was touched." >&2
  exit 0
fi
CHATBOT_URL="${CHATBOT_URL:-http://localhost:8001}"
# See nightly_finetune.sh's own LOGIN_BASE_URL comment: in a "real app"
# deployment, ak-chatbot's admin routes accept a token from the real
# app's own login, not one ak-chatbot issues itself (current_user()
# forwards validation to the real app's /api/users/me either way).
# Defaults to $CHATBOT_URL so a deployment where login and the admin
# routes are the same service needs no extra configuration.
LOGIN_BASE_URL="${LOGIN_BASE_URL:-$CHATBOT_URL}"
# Fetched once, up front, so every exit path below (sanity failure, gate
# failure, or a full success) can report to /api/admin/slm-versions —
# the Admin dashboard's version history — not just the happy path. Empty
# TOKEN (missing env file, or a login failure) degrades every report_
# version() call below to a silent no-op, same "local artifact still
# exported, just not visible to the dashboard" story report_version()
# itself documents.
TOKEN=""
if [[ -f "$FINETUNE_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$FINETUNE_ENV_FILE"
  TOKEN="$(curl -sf -X POST "$LOGIN_BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\": \"$FINETUNE_ADMIN_EMAIL\", \"password\": \"$FINETUNE_ADMIN_PASSWORD\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null || true)"
  if [[ -z "$TOKEN" ]]; then
    echo "Could not log in with FINETUNE_ADMIN_EMAIL/FINETUNE_ADMIN_PASSWORD — check" >&2
    echo "$FINETUNE_ENV_FILE. Skipping registration and dashboard reporting; the Ollama" >&2
    echo "tag and exported artifact will still be produced below." >&2
  fi
else
  echo "Missing $FINETUNE_ENV_FILE (needs FINETUNE_ADMIN_EMAIL/FINETUNE_ADMIN_PASSWORD — an" >&2
  echo "Executive-role account, so the new model can register itself via the same admin" >&2
  echo "providers API the Admin AI Models UI uses). Skipping registration — the Ollama tag" >&2
  echo "will still be created below if the conversion succeeds; register it by hand via the" >&2
  echo "Admin UI, or create $FINETUNE_ENV_FILE and re-run this script." >&2
fi

# Reports one night's outcome to the Admin dashboard's version history —
# a no-op (not a failure) if TOKEN is empty, so a missing/broken
# FINETUNE_ENV_FILE never blocks the actual model pipeline above it.
# args: sanity_check("passed"|"failed")  regression_gate(JSON literal,
# e.g. '"passed"' or 'null')  registered(true|false)
report_version() {
  [[ -z "$TOKEN" ]] && return 0
  curl -sf -X POST "$CHATBOT_URL/api/admin/slm-versions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
    -d "{\"tag\": \"$NEW_TAG\", \"base_tag\": \"$BASE_TAG\", \"quant_type\": \"$QUANT_TYPE\", \
        \"sanity_check\": \"$1\", \"regression_gate\": $2, \"registered\": $3, \
        \"domain\": $DOMAIN_JSON}" >/dev/null \
    || echo "Couldn't reach $CHATBOT_URL/api/admin/slm-versions — this night's outcome won't show on the dashboard, but nothing else was affected." >&2
}

echo "== 1/7: Converting $MERGED_DIR to GGUF (f16) =="
# Deliberately NOT a plain `mktemp -d` (defaults under /tmp): this
# script mixes native file ops (cp, sed, shell redirects — resolved
# against THIS process's own filesystem) with `$DOCKER run -v ...`/
# `$DOCKER cp` calls, which resolve host paths against the SIBLING
# dockerd's filesystem (see the Dockerfile.finetune-cron header comment
# on this exact "host path vs container path" split). When this script
# runs natively on a bare host those are the same filesystem, so a /tmp
# dir works fine — but when it runs inside ak-finetune-cron, /tmp is
# the CONTAINER's own ephemeral overlay, invisible to the host's
# dockerd, which silently auto-creates a SEPARATE empty directory of
# the same name on the real host for any `-v $WORK_DIR:...` mount.
# Every `$DOCKER run`-based step then reads/writes that host-side
# ghost directory while every native step reads/writes the
# container-side one — found live via a real conversion run producing
# a valid quantized GGUF that step 3's native `cp` then reported as
# "No such file or directory". $SCRIPT_DIR/.. (.../finetune) is bind-
# mounted at the SAME path on both sides (same reasoning as
# EXPORT_DIR/MERGED_DIR above), so putting WORK_DIR there makes native
# and dockerized steps agree either way this script is invoked.
WORK_DIR="$(mktemp -d -p "$SCRIPT_DIR/..")"
trap 'rm -rf "$WORK_DIR"' EXIT
# Runs inside GGUFCONVERT_IMAGE (see the prerequisite check above) —
# $MERGED_DIR and $WORK_DIR both get bind-mounted in so the script can
# read the merged model and write the GGUF back out to a host path the
# rest of this script (running natively) can read directly afterward.
$DOCKER run --rm \
  -v "$MERGED_DIR:/model:ro" -v "$WORK_DIR:/out" \
  "$GGUFCONVERT_IMAGE" \
  /model --outfile /out/model-f16.gguf --outtype f16

echo "== 2/7: Quantizing to $QUANT_TYPE (matches the base model's own quantization) =="
# See the LLAMAQUANTIZE_IMAGE comment above: runs inside a real glibc
# container rather than natively, since this script itself may be
# executing inside an Alpine/musl container (ak-finetune-cron).
# $LLAMA_QUANTIZE_BIN's own directory is bind-mounted read-only (its
# sibling .so's, if any, need to be on the loader's path too, hence
# LD_LIBRARY_PATH) alongside $WORK_DIR for input/output.
LLAMA_QUANTIZE_BIN_DIR="$(cd "$(dirname "$LLAMA_QUANTIZE_BIN")" && pwd)"
LLAMA_QUANTIZE_BIN_NAME="$(basename "$LLAMA_QUANTIZE_BIN")"
$DOCKER run --rm \
  -v "$LLAMA_QUANTIZE_BIN_DIR:/qbin:ro" -v "$WORK_DIR:/work" \
  -e LD_LIBRARY_PATH=/qbin \
  "$LLAMAQUANTIZE_IMAGE" \
  "/qbin/$LLAMA_QUANTIZE_BIN_NAME" /work/model-f16.gguf "/work/model-$QUANT_TYPE.gguf" "$QUANT_TYPE"

echo "== 3/7: Exporting the standalone artifact to $EXPORT_DIR =="
# A real, portable model export — the GGUF + the Modelfile that
# produced this Ollama tag, saved OUTSIDE both $WORK_DIR (deleted by
# the trap above on exit) and Ollama's own internal blob store, so
# Algo-Akanade-SLM can be copied to another machine, backed up, or
# loaded directly by any GGUF-capable runtime (llama.cpp, LM Studio,
# `ollama create` elsewhere) without depending on this exact install.
TAG_DIR_NAME="$(echo "$NEW_TAG" | tr ':/' '__')"
THIS_EXPORT_DIR="$EXPORT_DIR/$TAG_DIR_NAME"
mkdir -p "$THIS_EXPORT_DIR"
cp "$WORK_DIR/model-$QUANT_TYPE.gguf" "$THIS_EXPORT_DIR/model.gguf"

if [[ "$SERVE_BACKEND" == "ollama" ]]; then
  echo "== 4/7: Creating the new Ollama tag ($NEW_TAG) — $BASE_TAG untouched =="
  # Reuse the base model's own Modelfile (template/params Ollama already
  # knows are correct for this model family) rather than hand-authoring a
  # chat template that could drift — just swap which weights FROM points at.
  $DOCKER exec ak-ollama ollama show "$BASE_TAG" --modelfile > "$WORK_DIR/Modelfile"
  sed -i "s|^FROM .*|FROM /tmp/finetune-model.gguf|" "$WORK_DIR/Modelfile"
  cp "$WORK_DIR/Modelfile" "$THIS_EXPORT_DIR/Modelfile"
  $DOCKER cp "$WORK_DIR/model-$QUANT_TYPE.gguf" ak-ollama:/tmp/finetune-model.gguf
  $DOCKER cp "$WORK_DIR/Modelfile" ak-ollama:/tmp/finetune-Modelfile
  $DOCKER exec ak-ollama ollama create "$NEW_TAG" -f /tmp/finetune-Modelfile
  $DOCKER exec ak-ollama rm -f /tmp/finetune-model.gguf /tmp/finetune-Modelfile

  echo "== 5/7: Sanity check — the new tag actually responds =="
  if ! $DOCKER exec ak-ollama ollama run "$NEW_TAG" "Say OK." >/dev/null 2>&1; then
    echo "WARNING: $NEW_TAG was created but didn't answer a trivial prompt cleanly." >&2
    echo "Not registering it with the chatbot — investigate before doing so by hand." >&2
    python3 -c "import json,sys; json.dump({'tag': sys.argv[1], 'base_tag': sys.argv[2], 'quant': sys.argv[3], 'domain': sys.argv[5] or None, 'sanity_check': 'failed', 'regression_gate': None, 'registered': False}, open(sys.argv[4], 'w'), indent=2)" \
      "$NEW_TAG" "$BASE_TAG" "$QUANT_TYPE" "$THIS_EXPORT_DIR/metadata.json" "$DOMAIN"
    report_version "failed" "null" "false"
    echo "Exported artifact + metadata still saved at $THIS_EXPORT_DIR for inspection."
    exit 1
  fi
elif [[ "$SERVE_BACKEND" == "llama_server" ]]; then
  LLAMA_SERVER_IMAGE="${LLAMA_SERVER_IMAGE:-algo-akanade-server}"
  # A stable, currently-live container name PER DOMAIN (mirrors
  # PROVIDER_NAME's own per-domain naming) plus a throwaway "-candidate"
  # name for tonight's not-yet-proven build — same "old never touched
  # until new is proven better" safety property the Ollama path gets
  # from never overwriting an existing tag, just expressed as container
  # lifecycle instead of Ollama tags.
  SERVE_CONTAINER="ak-slm${DOMAIN:+-$DOMAIN}"
  SERVE_CANDIDATE="${SERVE_CONTAINER}-candidate"
  if ! $DOCKER image inspect "$LLAMA_SERVER_IMAGE" >/dev/null 2>&1; then
    echo "$LLAMA_SERVER_IMAGE image not built yet — build it once from the repo root:" >&2
    echo "  docker build -t $LLAMA_SERVER_IMAGE -f deploy/Dockerfile.llama-server ." >&2
    echo "Skipping tonight — the exported GGUF is still at $THIS_EXPORT_DIR, nothing already serving was touched." >&2
    exit 0
  fi

  echo "== 4/7: Starting the candidate server ($SERVE_CANDIDATE) — $SERVE_CONTAINER untouched =="
  $DOCKER rm -f "$SERVE_CANDIDATE" >/dev/null 2>&1 || true
  $DOCKER run -d --name "$SERVE_CANDIDATE" --network ak-net --restart unless-stopped \
    -v "$THIS_EXPORT_DIR:/models:ro" "$LLAMA_SERVER_IMAGE" \
    -m /models/model.gguf --host 0.0.0.0 --port 8080

  # A throwaway container on ak-net probes the candidate — this script
  # runs on the HOST, which has no route to ak-net's internal DNS/IPs
  # (these servers deliberately have no -p published port, same
  # "ak-net-internal only" pattern ak-ollama itself already uses), so
  # the health check has to run FROM inside ak-net. Reuses
  # FINETUNE_IMAGE (already a required prerequisite, has Python) rather
  # than pulling in a curl image just for this one check.
  wait_for_llama_server() {
    local container="$1"
    for _ in $(seq 1 30); do
      if $DOCKER run --rm --network ak-net --entrypoint python "$FINETUNE_IMAGE" -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://$container:8080/health', timeout=3)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1; then
        return 0
      fi
      sleep 2
    done
    return 1
  }
  fail_llama_server_candidate() {
    python3 -c "import json,sys; json.dump({'tag': sys.argv[1], 'base_tag': sys.argv[2], 'quant': sys.argv[3], 'domain': sys.argv[5] or None, 'sanity_check': 'failed', 'regression_gate': None, 'registered': False}, open(sys.argv[4], 'w'), indent=2)" \
      "$NEW_TAG" "$BASE_TAG" "$QUANT_TYPE" "$THIS_EXPORT_DIR/metadata.json" "$DOMAIN"
    report_version "failed" "null" "false"
    $DOCKER rm -f "$SERVE_CANDIDATE" >/dev/null 2>&1 || true
    echo "$SERVE_CANDIDATE discarded — $SERVE_CONTAINER (if any) keeps serving, untouched."
    echo "The exported GGUF itself is still at $THIS_EXPORT_DIR for inspection."
  }

  echo "== 5/7: Sanity check — the new candidate actually responds =="
  if ! wait_for_llama_server "$SERVE_CANDIDATE"; then
    echo "WARNING: $SERVE_CANDIDATE didn't become healthy within 60s." >&2
    echo "Not registering it with the chatbot — investigate before doing so by hand." >&2
    $DOCKER logs --tail 30 "$SERVE_CANDIDATE" >&2 || true
    fail_llama_server_candidate
    exit 1
  fi
  if ! $DOCKER run --rm --network ak-net --entrypoint python "$FINETUNE_IMAGE" -c "
import json, urllib.request, sys
req = urllib.request.Request(
    'http://$SERVE_CANDIDATE:8080/v1/chat/completions',
    data=json.dumps({'model': 'candidate', 'messages': [{'role': 'user', 'content': 'Say OK.'}]}).encode(),
    headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req, timeout=30)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1; then
    echo "WARNING: $SERVE_CANDIDATE is up but didn't answer a trivial chat request cleanly." >&2
    echo "Not registering it with the chatbot — investigate before doing so by hand." >&2
    fail_llama_server_candidate
    exit 1
  fi
else
  echo "Unknown SERVE_BACKEND=$SERVE_BACKEND (expected 'ollama' or 'llama_server')." >&2
  exit 1
fi

echo "== 6/7: Regression gate (§G) — is $NEW_TAG actually better than $BASE_TAG? =="
# Runs inside the same finetune image scripts.finetune_local_slm used
# (scripts/eval/ is baked in at build time — see deploy/Dockerfile.finetune
# — no volume mount needed for it), on ak-net so it can reach ak-ollama
# (or, in llama_server mode, the two servers below) by network alias.
# NOT "http://localhost:..." for either backend — none of these
# containers publish a host port (see update_containers.sh: no -p flag
# on ak-ollama; the candidate above has none either), so a host-side
# "localhost" URL was never actually reachable; only containers on
# ak-net can talk to them.
FINETUNE_IMAGE="${FINETUNE_IMAGE:-algo-kaizen-finetune}"
# $THIS_EXPORT_DIR (not a throwaway temp dir) so the full per-item
# report lands right next to this candidate's GGUF/Modelfile/
# metadata.json — kept even on failure, same as every other export
# artifact here.
EVAL_REPORT_HOST_PATH="$THIS_EXPORT_DIR/eval_report.json"
GATE_ARGS=(--base-tag "$BASE_TAG" --new-tag "$NEW_TAG" --report-out /finetune/export/eval_report.json)
SKIP_GATE=false
if [[ "$SERVE_BACKEND" == "ollama" ]]; then
  OLLAMA_NETWORK_URL="${OLLAMA_NETWORK_URL:-http://ak-ollama:11434/v1}"
  GATE_ARGS+=(--ollama-base-url "$OLLAMA_NETWORK_URL")
else
  # Two SEPARATE servers, not two tags on one server — regression_gate.py's
  # --new-base-url exists specifically for this (see its own --help):
  # the "old" candidate's base_url is the CURRENTLY-LIVE container, the
  # "new" candidate's is tonight's not-yet-promoted one.
  if $DOCKER inspect -f "{{.State.Running}}" "$SERVE_CONTAINER" >/dev/null 2>&1; then
    GATE_ARGS+=(--ollama-base-url "http://$SERVE_CONTAINER:8080/v1" \
               --new-base-url "http://$SERVE_CANDIDATE:8080/v1")
  else
    echo "No currently-live $SERVE_CONTAINER yet (first night for this domain) — nothing to gate against." >&2
    SKIP_GATE=true
  fi
fi
if [[ "$SKIP_GATE" == "true" ]]; then
  GATE_EXIT=2
else
  set +e
  $DOCKER run --rm --network ak-net -v "$THIS_EXPORT_DIR:/finetune/export" "$FINETUNE_IMAGE" \
      scripts.eval.regression_gate "${GATE_ARGS[@]}"
  GATE_EXIT=$?
  set -e
fi

# Reports the full per-item verification report to the Admin dashboard
# — a no-op if TOKEN is empty or the report file wasn't produced
# (GATE_EXIT==2: scoring never ran, nothing to report yet).
report_eval() {
  [[ -z "$TOKEN" || ! -f "$EVAL_REPORT_HOST_PATH" ]] && return 0
  python3 -c "
import json, sys
doc = json.load(open(sys.argv[1]))
doc['domain'] = sys.argv[2] or None
json.dump(doc, open(sys.argv[1], 'w'))
" "$EVAL_REPORT_HOST_PATH" "$DOMAIN"
  curl -sf -X POST "$CHATBOT_URL/api/admin/slm-eval-reports" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
    --data-binary @"$EVAL_REPORT_HOST_PATH" >/dev/null \
    || echo "Couldn't reach $CHATBOT_URL/api/admin/slm-eval-reports — tonight's per-item report won't show on the dashboard, but the local copy at $EVAL_REPORT_HOST_PATH is still there." >&2
}
GATE_RESULT="passed"
if [[ "$GATE_EXIT" == "2" ]]; then
  echo "Regression gate couldn't run (see above) — registering unconditionally this time. " >&2
  echo "Fix the setup so future nightly runs are actually gated." >&2
  GATE_RESULT="could_not_run"
elif [[ "$GATE_EXIT" != "0" ]]; then
  if [[ "$SERVE_BACKEND" == "ollama" ]]; then
    echo "$NEW_TAG failed the regression gate — it stays in Ollama for manual inspection, but is "
    echo "NOT being registered with the chatbot. Nothing already serving was touched."
  else
    echo "$NEW_TAG failed the regression gate against $SERVE_CONTAINER — NOT being promoted."
  fi
  python3 -c "import json,sys; json.dump({'tag': sys.argv[1], 'base_tag': sys.argv[2], 'quant': sys.argv[3], 'domain': sys.argv[5] or None, 'sanity_check': 'passed', 'regression_gate': 'failed', 'registered': False}, open(sys.argv[4], 'w'), indent=2)" \
    "$NEW_TAG" "$BASE_TAG" "$QUANT_TYPE" "$THIS_EXPORT_DIR/metadata.json" "$DOMAIN"
  report_version "passed" "\"failed\"" "false"
  report_eval
  if [[ "$SERVE_BACKEND" == "llama_server" ]]; then
    $DOCKER rm -f "$SERVE_CANDIDATE" >/dev/null 2>&1 || true
    echo "$SERVE_CANDIDATE discarded — $SERVE_CONTAINER (if any) keeps serving, untouched."
  fi
  echo "Exported artifact + metadata still saved at $THIS_EXPORT_DIR for inspection."
  exit 0
fi

if [[ "$SERVE_BACKEND" == "llama_server" ]]; then
  echo "Promoting $SERVE_CANDIDATE -> $SERVE_CONTAINER (the gate passed, or couldn't run on a first night)."
  $DOCKER rm -f "$SERVE_CONTAINER" >/dev/null 2>&1 || true
  $DOCKER rename "$SERVE_CANDIDATE" "$SERVE_CONTAINER"
  REGISTER_BASE_URL="http://$SERVE_CONTAINER:8080/v1"
else
  REGISTER_BASE_URL="http://ak-ollama:11434/v1"
fi

echo "== 7/7: Registering $NEW_TAG with the live chatbot (no restart needed) =="
if [[ -n "$TOKEN" ]]; then
  # Upsert BY NAME, not a fresh POST every night — every prior POST here
  # created a brand-new DB row all sharing the literal same name, so
  # which night's row actually won that slot in llm/router.py's
  # provider registry depended on the DB scan's incidental return
  # order, and every earlier night's row (and its now-superseded Ollama
  # tag) lingered forever. Look up the existing row by name first and
  # PATCH its `model` to tonight's tag — exactly ONE "$PROVIDER_NAME"
  # provider ever exists, deterministically always pointing at the
  # latest tag that passed the gate (one row per DOMAIN, not one
  # overall — a "legal" night never touches the "support" row).
  EXISTING_ID="$(curl -sf "$CHATBOT_URL/api/admin/providers" -H "Authorization: Bearer $TOKEN" \
    | python3 -c "
import sys, json
rows = json.load(sys.stdin)
match = next((r for r in rows if r['name'] == sys.argv[1]), None)
print(match['id'] if match else '')
" "$PROVIDER_NAME" 2>/dev/null || true)"
  if [[ -n "$EXISTING_ID" ]]; then
    # base_url included here too, not just model — a config edit that
    # switches serve_backend for an already-registered domain (ollama
    # <-> llama_server) needs the provider's base_url to actually move
    # with it, not keep pointing at the old backend forever.
    curl -sf -X PATCH "$CHATBOT_URL/api/admin/providers/$EXISTING_ID" \
      -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
      -d "{\"model\": \"$NEW_TAG\", \"base_url\": \"$REGISTER_BASE_URL\", \"enabled\": true}" >/dev/null
    echo "Updated the existing '$PROVIDER_NAME' provider ($EXISTING_ID) to point at $NEW_TAG ($REGISTER_BASE_URL)."
  else
    curl -sf -X POST "$CHATBOT_URL/api/admin/providers" \
      -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
      -d "{\"name\": \"$PROVIDER_NAME\", \"base_url\": \"$REGISTER_BASE_URL\", \
          \"model\": \"$NEW_TAG\", \"api_key\": \"not-needed-ollama-ignores-this\", \
          \"tiers\": [\"realtime\", \"bulk\"]}" >/dev/null
    echo "Registered '$PROVIDER_NAME' for the first time, pointing at $NEW_TAG ($REGISTER_BASE_URL)."
  fi
  echo "$PROVIDER_NAME is a real rung in the realtime/bulk tier chains AND directly"
  echo "selectable in the chat widget's model picker now, no restart."
  REGISTERED=true
else
  echo "No admin token available (see above) — $NEW_TAG exists in Ollama but isn't registered"
  echo "with the chatbot. Register it by hand via Admin > AI Models, or fix"
  echo "$FINETUNE_ENV_FILE and re-run this script."
  REGISTERED=false
fi

python3 -c "import json,sys; json.dump({'tag': sys.argv[1], 'base_tag': sys.argv[2], 'quant': sys.argv[3], 'domain': sys.argv[7] or None, 'sanity_check': 'passed', 'regression_gate': sys.argv[4], 'registered': sys.argv[5] == 'true'}, open(sys.argv[6], 'w'), indent=2)" \
  "$NEW_TAG" "$BASE_TAG" "$QUANT_TYPE" "$GATE_RESULT" "$REGISTERED" "$THIS_EXPORT_DIR/metadata.json" "$DOMAIN"
report_version "passed" "\"$GATE_RESULT\"" "$REGISTERED"
report_eval
echo "Exported standalone artifact (GGUF + Modelfile + metadata.json + eval_report.json) to $THIS_EXPORT_DIR"
