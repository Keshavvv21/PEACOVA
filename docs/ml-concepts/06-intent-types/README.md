# Intent Recognition & Classification

## 1. What "intent" means in an NLU/agent system

In a natural-language-understanding (NLU) or agentic system, **intent** is
the underlying goal or task category behind a user's utterance, as opposed
to the utterance's literal surface text. "Why is this function crashing on
empty input?" and "This throws a stack trace when I pass an empty list" are
different sentences, but they express the same intent: the user wants
*debugging* help. A classifier that maps utterances to intents lets the
rest of the system branch on **what kind of thing is being asked for**
rather than trying to build one universal handler that does everything
reasonably for every possible phrasing.

Intent classification is almost always a means to an end: **routing**.
Once you know an utterance's intent, you can:

- Select a different tool set, retrieval strategy, or sub-agent.
- Inject task-specific instructions into a prompt (as opposed to a single
  generic system prompt trying to cover every case).
- Decide whether to ask a clarifying question, escalate to a bigger model,
  or short-circuit to a canned response.
- Log/measure what kinds of requests a system actually receives.

Intent is distinct from two things it's often confused with:

- **Entities/slots** — the specific parameters within an utterance (e.g.
  "refactor `parse_config` in `settings.py`" has intent=`refactor` and
  entities `function=parse_config`, `file=settings.py`). Intent answers
  "what should happen"; entities answer "to what."
- **Sentiment/complexity** — orthogonal signals. A prompt's intent
  (debug vs. generate) is a different axis than how *hard* it is to
  satisfy. A system can and often does compute both independently (this
  repo does exactly that — see §6.4).

## 2. Approaches to intent classification, simplest to most sophisticated

There is no single "correct" way to classify intent — the right choice is a
function of how many intents you have, how much labeled data you have, your
latency/cost budget, and how much accuracy actually matters for the
downstream decision. Below are five approaches in increasing order of
sophistication (and, roughly, cost/latency).

### 2a. Rule-based / regex / keyword matching

The classifier is a hand-written list of patterns (regexes, keyword sets,
or simple string containment checks), each mapped to an intent label. The
utterance is tested against each pattern in order; the first one that
matches wins.

**Strengths:**
- Zero training data required — you write the patterns from domain
  knowledge, not from examples.
- Runs in microseconds; no model inference, no network call, no GPU.
- Fully deterministic and directly unit-testable: a given input either
  matches a pattern or it doesn't, with no sampling variance.
- Trivial to audit — you can read every rule and know exactly why any
  input was classified the way it was.

**Weaknesses:**
- Brittle to phrasing variation. A pattern tuned to `\bfix (this|the|a)\b`
  won't fire on "please patch this" or "can you resolve the issue" unless
  someone thinks to add those synonyms too. Coverage only grows as fast as
  someone manually extends the pattern list.
- No generalization: an input just outside the pattern's vocabulary is
  invisible to the classifier no matter how semantically close it is to a
  known intent.
- **Ordering-sensitivity is a first-class design concern.** Because
  matching is "first pattern that hits wins," a specific-but-narrow
  pattern must be checked *before* a broad, generic catch-all — otherwise
  the catch-all will steal the specific pattern's inputs before the
  specific pattern ever gets a chance to run. This is not an
  implementation detail; it's the central design decision in any
  regex-classifier, and getting the order wrong silently misroutes
  inputs with no error thrown anywhere.

Example ordering bug: if a generic "write" pattern is checked before a
"debug" pattern, the utterance "write a test that reproduces this crash"
would match the generic pattern (because it contains "write") and never
reach the more specific and more useful `debug` classification — even
though a human reading the sentence would obviously call it a debugging
task, not a "please generate some code" task.

### 2b. Traditional ML classifiers (TF-IDF + logistic regression/SVM, or a small feedforward net over bag-of-words)

Here the utterance is converted into a fixed-size numeric feature vector —
classically TF-IDF weighted term frequencies, or a bag-of-words/one-hot
encoding — and fed into a conventional supervised classifier: logistic
regression, an SVM, a small feedforward network, or a gradient-boosted tree.

**Strengths:**
- Learns weighted associations between vocabulary and intents from labeled
  examples, so it generalizes better than hand-written regexes to
  phrasings that share vocabulary with the training set but weren't
  explicitly anticipated.
- Cheap to train and cheap to run at inference — no GPU needed, inference
  is a sparse matrix-vector multiply, sub-millisecond.
- Well-understood, interpretable-ish (you can inspect feature weights per
  class), mature tooling (`scikit-learn` and equivalents).

**Weaknesses:**
- Requires a labeled training set per intent — you must collect and
  annotate example utterances before you can train anything, and that
  labeling effort scales with the number of intents and how nuanced their
  boundaries are.
- Bag-of-words/TF-IDF representations discard word order and syntax, so
  they can be fooled by negation ("this is not a bug") or fine-grained
  compositional meaning that changes with word order.
- Adding a new intent, or meaningfully shifting an existing intent's
  boundary, requires re-collecting/re-labeling data and re-running the
  training pipeline — it's not a config-file edit the way adding a regex
  pattern is.
- Out-of-vocabulary terms at inference time (product names, jargon not
  seen in training) contribute nothing to the feature vector.

### 2c. Embedding-based similarity classification

The utterance is embedded into a dense vector space (via a sentence-embedding
model), and the classifier compares that vector's cosine similarity against
embedded canonical examples (or a learned centroid) for each intent. The
intent with the highest similarity wins — nearest-centroid classification,
or k-nearest-neighbors over a small few-shot example bank per intent.

**Strengths:**
- No training loop at all — you only need a *few* representative example
  utterances per intent (few-shot), not a large labeled training set.
  Adding a new intent means adding a handful of examples and re-embedding
  them, not retraining a model.
- Embeddings capture semantic similarity, not just lexical overlap, so
  this generalizes to paraphrases that share no vocabulary with the
  canonical examples at all (e.g. "the app keeps hanging" scoring close
  to a `debug` example that says "process appears to be deadlocked" even
  though they share zero words).
- Inference cost is one embedding call plus a handful of cosine-similarity
  dot products — fast, and embeddings can be cached/precomputed for the
  canonical examples.

**Weaknesses:**
- Needs an embedding model available at inference time (a small
  latency/infra dependency that regex and classic ML don't have).
- Quality is bounded by the embedding model's own semantic space — subtle
  or highly domain-specific distinctions may not separate cleanly in a
  general-purpose embedding space.
- Choosing good canonical examples per intent is itself a design problem;
  a poorly chosen example set can pull the centroid toward an
  unrepresentative region of the space.

### 2d. Fine-tuned transformer classifiers (e.g. BERT + classification head)

A pretrained transformer encoder (BERT, RoBERTa, DistilBERT, etc.) has a
classification head attached to its output and is fine-tuned end-to-end on
a labeled intent dataset.

**Strengths:**
- Typically the highest accuracy of the "classical" approaches — the model
  learns contextual, compositional representations of the input rather
  than relying on bag-of-words features or a fixed embedding space it
  wasn't specifically tuned for.
- Handles negation, compositionality, and long-range dependencies within
  the utterance far better than TF-IDF-based methods.
- Once trained, inference is fast (a single forward pass) relative to
  calling a large generative LLM.

**Weaknesses:**
- Requires a labeled dataset (often larger than what 2b/2c need to work
  well) and a real training/fine-tuning pipeline: GPU time, hyperparameter
  tuning, evaluation splits, versioning of trained checkpoints.
- Highest engineering overhead of the five approaches to stand up and
  maintain — you need MLOps around training data, retraining cadence, and
  model deployment, not just a config file or a prompt.
- Like 2b, adding/changing an intent requires new labeled data and
  retraining, not just an edit.

### 2e. LLM-based zero-shot/few-shot classification

A capable general-purpose LLM is prompted directly to classify the intent
— either zero-shot ("classify this utterance into one of: debug, refactor,
test, ... — respond with only the label") or few-shot (a handful of
labeled examples included in the prompt).

**Strengths:**
- Most flexible of all five: no training data and no training pipeline
  required at all. New intents can be added by editing the prompt's list
  of categories.
- Handles genuinely novel phrasing, compositional reasoning, and
  ambiguous cases better than any of the above, because it leverages the
  LLM's full general-language understanding rather than a
  purpose-built classifier.
- Can be asked to also explain its reasoning, handle multi-intent cases,
  or emit a calibrated-sounding confidence — all in the same call, since
  it's a general reasoning engine, not a fixed classifier head.

**Weaknesses:**
- **Highest latency and cost** of all five approaches — it requires a full
  model inference call (often to a hosted API), typically hundreds of
  milliseconds to multiple seconds, versus microseconds for regex or
  sub-millisecond for classical ML.
- **Non-deterministic**: without careful constraint (e.g. temperature 0,
  constrained decoding/grammar), the same input can be classified
  differently across calls, which makes the classifier harder to unit
  test and harder to reason about at the boundary between two intents.
- If intent classification is a *precursor* step before a second LLM call
  actually handles the request, this doubles the number of model calls
  per user turn — a real latency and cost multiplier at scale, and
  particularly damaging when the downstream model is itself
  latency-sensitive (see §6 — this is precisely the tradeoff this
  repository's own reasoning layer was designed to avoid).

## 3. Multi-intent and hierarchical intent

Real utterances don't always map cleanly to exactly one intent from a flat
list.

**Multi-intent**: a single utterance can legitimately span more than one
intent at once — e.g. "refactor this function and add tests for it" is
simultaneously a `refactor` request and a `test` request. Systems that need
to handle this either (a) run a multi-label classifier that can emit more
than one label per input, or (b) segment the utterance into clauses and
classify each clause independently, or (c) pick a single dominant intent
by some priority rule (which is effectively what an ordered first-match
regex classifier does implicitly, whether or not that's acknowledged as a
design choice — see §2a's ordering discussion and §6's discussion of the
same tradeoff in this repo).

**Hierarchical intent**: some systems organize intents in two (or more)
levels — coarse top-level categories with finer sub-intents nested beneath
them. For example, a coarse `code_task` category might have sub-intents
`debug`, `refactor`, `test`, while a coarse `conversation` category has
sub-intents `greeting`, `clarification_request`, `off_topic`. Hierarchical
schemes let a system make a cheap coarse-grained routing decision first
(e.g. "is this even a coding task?") and only pay the cost of finer
classification when the coarse category needs it — a useful pattern when
the coarse decision is common and cheap to get right, and the fine-grained
decision is rarer or more expensive to compute accurately.

## 4. Confidence scoring and fallback / "none of the above" handling

Every intent classifier — regardless of which of the five approaches above
it uses — needs an explicit escape hatch for inputs that don't confidently
belong to any known intent:

- **Classical ML and transformer classifiers** naturally produce a
  probability distribution over classes (softmax output). A common pattern
  is to threshold on the top class's probability: if it's below some
  cutoff, route to a fallback/"none of the above" intent (often called
  `unknown`, `out_of_scope`, `fallback`, or similar) rather than
  confidently misrouting into whichever class happened to score highest
  by a small margin.
- **Embedding-similarity classifiers** do the analogous thing with cosine
  similarity: if the best-matching intent's similarity score is below a
  threshold, nothing matched well enough to trust, and the system falls
  back.
- **LLM-based classifiers** can be prompted to emit an explicit
  "none of the above" / "unsure" option alongside the real intent list, or
  a confidence self-estimate, though self-reported LLM confidence is
  notoriously miscalibrated and shouldn't be trusted as a precise
  probability.
- **Regex/keyword classifiers** have no notion of graded confidence at
  all — a pattern either matches or it doesn't. Their fallback is
  structural rather than probabilistic: if no pattern in the ordered list
  matches, the classifier falls through to a default "general" bucket.
  This is a binary, not a threshold, but it serves exactly the same
  purpose as a low-confidence fallback in a probabilistic classifier: it
  guarantees the classifier always returns *something* usable rather than
  raising an error or leaving the caller with no signal at all.

The reason a fallback bucket is mandatory, not optional, in any of these
approaches: **the space of intents you enumerated in advance is never the
space of inputs users actually send.** Users ask off-topic questions, make
typos, mix languages, or simply phrase things in ways nobody anticipated
when the classifier was designed. A classifier without a fallback either
crashes, returns a null/undefined result the caller has to special-case
anyway, or — worse — confidently misroutes an unrelated input into
whichever intent's pattern happens to loosely match, silently corrupting
downstream behavior. A well-designed fallback intent turns an unhandled
case into a *handled, generic* case: something that still gets processed
usefully by a general handler, rather than falling through a hole in the
system entirely.

## 5. Diagrams

### 5.1 The five approaches on a cost/latency vs. accuracy/generalization spectrum

```
 accuracy /
 generalization
      ▲
      │                                                    ● (e) LLM zero/few-shot
      │                                                      classification
      │                                                      - most flexible, handles
      │                                                        novel phrasing & reasoning
      │                                                      - non-deterministic, highest
      │                                          ● (d) Fine- $$$ / slowest
      │                                            tuned transformer
      │                                            (BERT + head)
      │                                            - needs labeled data +
      │                                              training pipeline
      │                              ● (c) Embedding similarity
      │                                (nearest-centroid / few-shot)
      │                                - no training loop, generalizes
      │                                  to paraphrases via semantic space
      │                 ● (b) Traditional ML
      │                   (TF-IDF + logreg/SVM)
      │                   - needs labeled data,
      │                     lexical features only
      │  ● (a) Rule-based / regex
      │    keyword matching
      │    - zero training data,
      │      brittle to phrasing
      └──────────────────────────────────────────────────────────▶
        microseconds   sub-ms      ms          ms-100ms    100ms-s+
                         cost / latency (per classification)

  Rule of thumb: move right along this spectrum only as far as your
  accuracy requirement and label budget force you to — every step right
  trades determinism, latency, and simplicity for generalization.
```

### 5.2 The regex approach's decision flow

```
                     ┌───────────────────────────┐
                     │   Incoming utterance text  │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │ Ordered pattern list (most         │
                 │ specific → most generic):          │
                 │                                     │
                 │   1. pattern_A  (specific intent)   │
                 │   2. pattern_B  (specific intent)   │
                 │   3. pattern_C  (specific intent)   │
                 │   4. pattern_D  (generic catch-all) │
                 └─────────────┬───────────────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │  Walk the list top → bottom.  │
                    │  Test each pattern in order.  │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │ Does pattern_N match text? │
                     └─────┬───────────────┬──────┘
                        yes│               │no
                           ▼               ▼
                ┌─────────────────┐   try pattern_(N+1)
                │ STOP. Return     │   (loop back up)
                │ intent for       │
                │ pattern_N.       │
                │ FIRST MATCH WINS │
                │ — later patterns │
                │ never evaluated. │
                └─────────────────┘
                           │
                (reached end of list, nothing matched)
                           ▼
                ┌─────────────────────────┐
                │  Fallback intent         │
                │  ("general"/"unknown")   │
                └─────────────────────────┘

  Key property: because it's first-match-wins, ORDER encodes priority.
  A narrow, specific pattern placed above a broad, generic pattern will
  correctly claim inputs that satisfy both; the same two patterns in the
  opposite order would let the generic one silently swallow the specific
  one's inputs.
```

## 6. How this repository does it

### 6.1 Location and shape

The Coding Agent's intent classifier lives in
`orchestrator_framework_layer/akanade/backend/app/llm/intent_reasoner.py`.
It defines six fixed intent categories as module-level string constants
(`intent_reasoner.py:26-31`):

```python
INTENT_DEBUG = "debug"
INTENT_REFACTOR = "refactor"
INTENT_TEST = "test"
INTENT_REPO_ANALYSIS = "repo_analysis"
INTENT_GENERATE = "generate"
INTENT_GENERAL = "general_coding"
```

That's **six intents total**: `debug`, `refactor`, `test`,
`repo_analysis`, `generate`, and the fallback `general_coding`.

Each of the first five (all but `general_coding`) has an associated
compiled regex in an ordered list, `_PATTERNS`
(`intent_reasoner.py:38-58`), structured as `list[tuple[str,
re.Pattern]]` — pairs of `(intent_label, compiled_pattern)` checked in
list order:

1. `debug` (`intent_reasoner.py:39-43`) — matches on words like `debug`,
   `diagnos*`, `root cause`, `fix (this|the|a)`, `why (is|does|did)`,
   `error`, `bug`, `crash(es|ed|ing)?`, `exception`, `traceback`,
   `stack trace`, `fails?`/`failing`, `broken`, `not working`,
   `doesn't work`, `race condition`, `deadlock`, `memory leak`,
   `hangs?`/`stuck`, `regression`.
2. `refactor` (`intent_reasoner.py:44-47`) — `refactor`, `clean ?up`,
   `simplify`, `reorgani[sz]e`, `extract* (a |the )?(function|method|
   class|module)`, `rename`, `restructure`, `de-?duplicate`, `split
   (this|the) (function|file|class)`, `improve readability`.
3. `test` (`intent_reasoner.py:48-50`) — `unit tests?`, `write tests?`,
   `pytest`, `test coverage`, `test suite`, `assert*`, `mock*`, `edge
   cases?`.
4. `repo_analysis` (`intent_reasoner.py:51-54`) — `repository`,
   `repo-?level`, `codebase`, `across (the|this) (repo|codebase|
   project)`, `entire project`, `every file`, `whole project`, `this
   project's`.
5. `generate` (`intent_reasoner.py:55-57`) — `write`, `create`,
   `implement`, `generate`, `build`, `add (a|an|the)`, `design (a|an)`.

`classify_intent()` (`intent_reasoner.py:98-102`) is exactly the
first-match-wins loop described in §5.2:

```python
def classify_intent(text: str) -> str:
    for intent, pattern in _PATTERNS:
        if pattern.search(text):
            return intent
    return INTENT_GENERAL
```

If none of the five ordered patterns match, it returns
`INTENT_GENERAL` (`general_coding`) — the fallback bucket from §4,
guaranteeing the function always returns a usable string rather than
`None` or an exception, no matter how off-topic or unanticipated the
input is.

`guidance_for()` (`intent_reasoner.py:105-106`) is a plain dict lookup
into `_GUIDANCE` (`intent_reasoner.py:65-95`), a hardcoded
`dict[str, str]` mapping each intent to a short, imperative instruction
string — e.g. the `debug` entry (`intent_reasoner.py:66-71`) reads:

> " The user is asking you to DEBUG something real. Find the actual
> root cause before proposing a fix — read the file or run the code
> to see the real error first; do not guess at a fix from the
> description alone if you can verify it directly instead."

`general_coding` maps to the empty string (`intent_reasoner.py:94`) —
by design, a plain knowledge question gets no debug/refactor/test
framing bolted onto it, per
`README_SLM_REASONING_LAYER.md:79` ("No extra guidance — a plain
knowledge question doesn't need debug/refactor/test framing bolted
on").

### 6.2 The explicit ordering rule, and why it exists

The comment directly above `_PATTERNS` (`intent_reasoner.py:33-37`)
states the ordering rule in exactly the terms §2a and §5.2 describe
generically:

> "Order matters — checked top to bottom, first match wins.
> Debug/refactor/test/repo-analysis are checked BEFORE the generic
> "generate" bucket because a prompt like "write a test that
> reproduces this bug" should resolve to debug (the more SPECIFIC,
> more actionable intent), not the generic "write" verb that also
> appears in it."

This is the concrete instance of the ordering-sensitivity concern
raised in §2a and §3 (multi-intent handling via priority): "write a
test that reproduces this bug" genuinely contains vocabulary for at
least three of the five patterns (`write` → generate, `test` → test,
`bug` → debug). Because `debug` is checked first in `_PATTERNS`, it
wins — matching the repo's own stated judgment that the *debugging*
framing is the more actionable one for a prompt like this. The repo
has a regression test guarding exactly this ordering decision:
`test_debug_wins_over_generate_when_both_verbs_present` in
`tests/test_intent_reasoner.py`
(`README_SLM_REASONING_LAYER.md:81-86`).

### 6.3 Which of the five approaches this is, and the repo's own justification

This is unambiguously **approach 2a — rule-based/regex/keyword
matching**: a fixed, hand-written, ordered list of compiled regex
patterns, checked with `.search()`, first match wins, structural
fallback on no match. There is no training data, no model weights, no
embedding call, and no learned parameter anywhere in this file.

The module's own docstring (`intent_reasoner.py:1-21`) and
`README_SLM_REASONING_LAYER.md` are both explicit and specific about
*why* this choice was made, and the reasoning is narrower than a
generic "regex is fast" claim — it is tied directly to this system's
particular latency-sensitive path:

- The docstring states the layer is "a cheap, deterministic
  pre-generation step, not another LLM round trip"
  (`intent_reasoner.py:1-2`), and explains: "an SLM is already the
  latency-sensitive path (see providers.py's OLLAMA_KEEP_ALIVE comment,
  hybrid_router.py's LOCAL_TIMEOUT_SECONDS), so a design that 'reasons
  about intent' via yet another model call would directly fight the
  exact latency goal this layer exists to support"
  (`intent_reasoner.py:6-13`).
- `README_SLM_REASONING_LAYER.md:16-21` restates this as one of two
  explicit "what this is not" clarifications: "**Not** a second model
  call... adding a 'let me think about your intent first' round trip
  in front of every SLM generation would directly fight the whole
  reason this layer exists. It runs in the same single generation
  call, as prompt conditioning."
- The same document is equally explicit that the tradeoff is real, not
  hand-waved away: "**Not** a neural intent classifier. It's a
  regex/keyword heuristic — the same order of technique
  `llm/hybrid_router.py`'s own `_complexity_features()` already uses
  for its cheap, non-LLM complexity estimate. Cheap, fast
  (microseconds), and fully deterministic/testable, at the cost of
  being a heuristic rather than a learned classifier — a real,
  disclosed tradeoff, not a limitation hidden behind an impressive-
  sounding name" (`README_SLM_REASONING_LAYER.md:22-28`).

So the justification given is precisely the one described in the task
prompt: a second LLM call to classify intent would add latency and
cost that directly conflicts with the goal this whole layer exists to
serve (making a single SLM generation call better-targeted, in a
system where the SLM path is already the latency-sensitive one). It is
worth being precise about what the README does and does not claim: it
does not claim regex is *more accurate* than a learned classifier for
this task — it explicitly concedes the opposite framing ("at the cost
of being a heuristic rather than a learned classifier"). The
justification is a latency/cost/simplicity argument, not an accuracy
argument.

### 6.4 Where the classified intent goes: wiring into `coding_agent.py`

`coding_agent.py` imports the two public functions at the top of the
file:

```python
from ...llm.intent_reasoner import classify_intent, guidance_for
```

(`coding_agent.py:82`)

The classifier is invoked inside `CodingAgent.plan()`, immediately
before the outgoing system prompt is finalized
(`coding_agent.py:415-434`):

```python
def plan(self, task: Task, understanding: dict) -> dict:
    if understanding["confirming_write"]:
        return {"confirm_write": True}
    if understanding["canceling_write"]:
        return {"cancel_write": True}
    content = SYSTEM_PROMPT["content"]
    if task.metadata.get("client_tools"):
        content += _CLIENT_TOOLS_ADDENDUM
    # The intent-reasoning layer (llm/intent_reasoner.py) — a cheap,
    # deterministic classification of what KIND of coding task this
    # is (debug/refactor/test/repo-analysis/generate), appended as
    # intent-specific guidance rather than run as a second model
    # call, so it costs nothing in latency. See
    # README_SLM_REASONING_LAYER.md for the full architecture.
    intent = classify_intent(understanding["text"])
    content += guidance_for(intent)
    system = {**SYSTEM_PROMPT, "content": content}
    history = _sanitize_history(task.metadata.get("history"))
    return {"messages": [system, *history, {"role": "user", "content": understanding["text"]}],
           "intent": intent}
```

(`coding_agent.py:415-434`, classification call and guidance append at
`coding_agent.py:429-430`)

Concretely: `classify_intent()` runs against the raw user text
(`understanding["text"]`, itself just `task.text` per
`coding_agent.py:411`), and the resulting intent-specific guidance
string from `guidance_for()` is string-concatenated onto `content` —
the same `content` variable that already holds `SYSTEM_PROMPT`'s base
text (and the client-tools addendum, if applicable). That combined
string becomes the `content` field of the `system` message sent to the
model in the same `messages` list as the conversation history and the
current user turn. There is no branch, no separate call, and no
conditional model selection based on intent here — the classification
result only ever changes *what text is inside the one system message*
that the one, already-planned LLM call receives.

The `intent` string is also returned alongside `messages` in `plan()`'s
result dict and threaded through further down: `coding_agent.py:684-687`
surfaces it into the agent's result object so it can be inspected by
callers, and `coding_agent.py:822` includes it in a further result
payload. `README_SLM_REASONING_LAYER.md:60-62` explains what this is
for: `AgentResult.raw["intent"]` lets `scripts/run_slm_test_suite.py`
verify, per prompt, that the regex classifier's answer agrees with the
category a human assigned that prompt to when writing the test suite —
i.e., the classification is validated empirically against a 100-prompt
human-labeled catalog organized under the same six-category taxonomy,
rather than merely asserted to work.

One documented limitation worth surfacing: `classify_intent()`'s output
currently only conditions the *prompt content* sent to whichever
provider was already selected — it does not currently feed the
Auto-Hybrid local-vs-hosted routing decision itself
(`hybrid_router.choose()`), which is described as "a separate,
complementary signal (complexity, not intent)"
(`README_SLM_REASONING_LAYER.md:100-103`). The README flags escalating
specific intents (e.g. always routing `repo_analysis` — which needs more
context-window headroom than a small local model typically has — to a
hosted model) as "a natural next step, not yet built"
(`README_SLM_REASONING_LAYER.md:104-108`).

### 6.5 Why approach (a) is a reasonable fit here, and where its limits would show

Mapping this back to §2's five-approach spectrum: this system chose the
cheapest, fastest, most deterministic point on the spectrum (2a)
specifically because the *cost of the intent step itself* had to be
effectively zero relative to the SLM call it precedes — any of
approaches 2b-2e would have added meaningfully more latency (2b/2c
add a model/embedding call; 2d adds a training pipeline and inference
cost; 2e doubles the number of LLM calls per turn, the exact failure
mode the README explicitly rules out).

The tradeoffs this repository accepts by choosing 2a are exactly the
generic 2a weaknesses from §2a, made concrete for this domain: the
patterns cover only the vocabulary someone anticipated (e.g. a prompt
that says "please patch this" without any of `fix|bug|error|crash|...`
would not trigger `debug`), and any future new intent, or shift in how
an existing intent's boundary is drawn, requires editing the pattern
list and its ordering by hand rather than relabeling data and
retraining. The 15-case unit test suite in `tests/test_intent_reasoner.py`
and the 100-prompt live-agreement check in
`scripts/run_slm_test_suite.py` (`README_SLM_REASONING_LAYER.md:110-120`)
are the concrete mitigations for that brittleness: they don't make the
classifier smarter, but they do make its failure modes visible and
regression-testable rather than silent.
