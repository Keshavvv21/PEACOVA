# ALGO AKANADE — Explained: What, How, and the Code Line by Line

This is the "explain it to me" document — no arrow diagrams (see
`README_ARROW_DIAGRAMS.md`), no formula reference (see
`README_RL_ALGORITHMS.md`). Just: what is "intent," what is RAG, how
does the RL primitive actually work at the code level, and what does
each layer in this system actually do and why.

## What is "intent," in this system?

"Intent" is what `perceive_intent` (step 6 of the 17-step algorithm)
figures out: **which single tool the user's message is actually asking
for, with what arguments** — or, if none fit, that the message doesn't
map to any PM action at all.

Concretely: the LLM is given the full list of available tools (their
names, descriptions, and argument schemas) alongside the conversation,
and asked to either (a) call exactly one tool with extracted arguments,
or (b) reply in plain text if nothing fits. That's the entirety of
"understanding intent" here — it is deliberately **not** a general
free-form reasoning step. The model isn't asked "what should happen
next in the abstract"; it's asked "which of these specific, pre-defined
actions (if any) does this map to." That narrowness is what makes
`output_guard` possible at all: a tool call's shape can be mechanically
validated against a known JSON schema, and entity names can be
mechanically resolved against real data (step 7) rather than trusted
as-is — the model's job is classification + extraction, not open-ended
generation.

When nothing fits, that's `out_of_scope: true` — a real signal, not an
error. The system doesn't refuse; it answers as best it can in free
text, but flags that answer as "general knowledge, not grounded in your
PM data," and the UI gates showing it behind an explicit "show me
anyway."

## What is RAG, and why does this system have two versions of it?

**RAG = Retrieval-Augmented Generation.** The core idea, in one
sentence: instead of asking a language model to answer purely from
whatever it memorized during training, you first **retrieve** a handful
of relevant text passages from a knowledge base you control, hand those
passages to the model as extra context, and ask it to answer **grounded
in that context**. This matters for two reasons: the model's training
data is frozen at some past date and doesn't know about *your* data at
all (your tasks, your documents), and grounding an answer in retrieved
text makes it checkable — you can point at exactly which passage backed
a claim, instead of trusting an ungrounded assertion.

The mechanics, step by step:

1. **Chunk** — a long document is split into smaller passages (`rag.py`'s
   `chunk_text()`), because embedding and retrieving whole documents is
   both less precise and more expensive than retrieving the specific
   paragraph that actually answers the question.
2. **Embed** — each chunk is converted into a vector (a list of numbers)
   that captures its meaning, such that semantically similar chunks end
   up as nearby vectors. This system uses either a real sentence-
   embedding model or, when that's not installed, a dependency-free
   hash-based fallback embedder — same interface either way.
3. **Store** — those vectors go into a vector database (`VectorStoreAdapter`
   — Chroma-backed for the real PM data, or a pure-Python in-memory
   store for anything else).
4. **Retrieve** — given a new query, embed *it* too, and find the
   stored chunks whose vectors are closest (cosine similarity) — these
   are the passages most likely to be relevant.
5. **Rerank** (optional but real here) — the raw vector-similarity order
   isn't always the best order; re-scoring by keyword overlap too can
   surface a chunk that's an exact keyword match but scored lower on
   pure embedding similarity.
6. **Build context** — the top passages get concatenated (with source
   labels) into a text block, truncated to a token/char budget.
7. **Generate** — that context block is handed to the LLM as extra
   input alongside the actual question, with an instruction to answer
   using it.

**Why two "versions" exist in this codebase:**

- `gmaofn/rag.py` is the **generic, foundational layer** — the
  interfaces (`VectorStoreAdapter`), the two real backends, and the
  chunk/rerank/build-context pipeline stages. It has no opinion on
  *which* store to use or *how* to rank results beyond a plain re-sort.
- `gmaofn/rag_optimizer.py` sits **on top of** that generic layer and
  adds the parts a real deployment actually needs: deciding which
  knowledge source to query when more than one exists, deciding which
  ranking algorithm to apply (plain vector similarity isn't always
  best), skipping retrieval entirely for messages that clearly don't
  need it, and reporting exactly which passages made it into the final
  answer so a user can be told what was actually used. That "which
  source, which algorithm" decision is where the RL layer comes in —
  see the code walkthrough below and `README_RL_ALGORITHMS.md` §6 for
  the exact math.

## The RL algorithm, line by line — `gmaofn/bandit.py`'s `LinearBanditCore`

This is the ONE piece of math every RL layer in this system reuses.
Reading it once here means every other layer's description
("LinearBanditCore, arms = X, context = Y, reward = Z") is fully
specified — there's no other hidden mechanism.

```python
class LinearBanditCore:
    def __init__(self, feature_dim: int, *, ridge: float = 1.0,
                exploration_scale: float = 0.35, rng: np.random.Generator | None = None):
        self._d = feature_dim + 1  # +1 bias term
```
The bandit tracks a linear model `reward ≈ θ·x` for each arm. `+1` adds
a constant "1" feature to every context vector so the model can learn a
baseline reward independent of any real feature (a standard linear-
regression trick — without it, the line would be forced through the
origin).

```python
        self._ridge = ridge
        self._nu = exploration_scale
        self._B: dict[str, np.ndarray] = {}
        self._f: dict[str, np.ndarray] = {}
        self._rng = rng or np.random.default_rng()
```
`ridge` (λ) controls how strong the prior pulling untested arms toward
zero is — a bigger ridge means more data is needed before the model
trusts its own posterior. `exploration_scale` (ν) controls how much
each arm's *sampled* score can wander from its *mean* score — bigger
means more exploration. `_B`/`_f` are dictionaries because every arm
gets its own independent posterior, keyed by arm id (a plain string) —
that's the whole trick that lets one `LinearBanditCore` serve completely
unrelated decisions (which store, which reduction level, which
generation params) just by using different arm-id strings per caller.

```python
    def ensure(self, arm_id: str, seed_utility: float = 0.0) -> None:
        if arm_id in self._B:
            return
        self._B[arm_id] = self._ridge * np.eye(self._d)
        self._f[arm_id] = np.full(self._d, seed_utility * self._ridge / self._d)
```
The first time an arm is ever touched, it gets initialized:
`B = λI` (an identity matrix scaled by the ridge) — this is the
"we know nothing yet" starting covariance. `f` is set so that the
initial posterior mean works out to `seed_utility` for every feature
equally — a deliberately crude, easily-overridden guess, not a claim of
real evidence. If a caller doesn't pass `seed_utility`, it defaults to
0 (no bias at all, pure "I don't know").

```python
    @staticmethod
    def features(context_vector: list[float]) -> np.ndarray:
        return np.array(context_vector + [1.0])
```
This is where the "+1 bias term" from `__init__` actually gets appended
— every raw context vector a caller builds (e.g. `[0.4, 0.1]`) becomes
`[0.4, 0.1, 1.0]` before any math touches it.

```python
    def score(self, arm_id: str, context_vector: list[float], *,
             seed_utility: float = 0.0) -> tuple[float, float]:
        self.ensure(arm_id, seed_utility)
        x = self.features(context_vector)
        B = self._B[arm_id]
        f = self._f[arm_id]
        B_inv = np.linalg.inv(B)
        mu = B_inv @ f
        cov = (self._nu ** 2) * B_inv
        theta = self._rng.multivariate_normal(mu, cov)
        return float(theta @ x), float(mu @ x)
```
The heart of Thompson Sampling. `mu = B_inv @ f` is the posterior
**mean** weight vector — the best point estimate of "how much each
feature predicts reward for this arm," given everything observed so
far. `cov` scales that same inverse matrix by `ν²` to get the
posterior's spread. `theta = rng.multivariate_normal(mu, cov)` draws
ONE random sample from that whole distribution — not the mean itself,
a random point near it, wider when less is known. `theta @ x` (dot
product) turns that sampled weight vector into an actual predicted
reward number for the current context; `mu @ x` does the same with the
mean instead, returned alongside for callers (like
`posterior_mean`-based tests) that want the noise-free estimate.

```python
    def select(self, arm_ids: list[str], context_vector: list[float], *,
              seed_utilities: dict[str, float] | None = None) -> tuple[str, dict[str, float]]:
        if not arm_ids:
            raise ValueError("select() needs at least one arm")
        seeds = seed_utilities or {}
        scored = {a: self.score(a, context_vector, seed_utility=seeds.get(a, 0.0))[0] for a in arm_ids}
        winner = max(scored, key=scored.get)
        return winner, scored
```
Score EVERY candidate arm — each with its own independent random draw
from its own posterior — and pick whichever scored highest. Because
each arm's draw is independent and noisy, an arm with a wide (uncertain)
posterior occasionally wins even if its mean looks worse than a
well-established rival's — that's the actual exploration mechanism, and
it's *automatic*: nothing here has an explicit "explore X% of the time"
knob, it falls naturally out of sampling from real uncertainty.

```python
    def update(self, arm_id: str, context_vector: list[float], reward: float) -> None:
        self.ensure(arm_id)
        x = self.features(context_vector)
        self._B[arm_id] += np.outer(x, x)
        self._f[arm_id] += reward * x
```
After a real outcome is observed, `update()` folds it into the arm's
running statistics — this is the online Bayesian linear regression
update rule: `B` accumulates the outer product of every seen context
(this is what the posterior's covariance is derived from), and `f`
accumulates `reward × context` (this is what the posterior's mean is
derived from). No stored history of past contexts/rewards is needed —
these two small matrices/vectors are a complete, ever-shrinking
summary of everything ever seen for that arm.

```python
    def posterior_mean(self, arm_id: str) -> np.ndarray | None:
        if arm_id not in self._B:
            return None
        return np.linalg.inv(self._B[arm_id]) @ self._f[arm_id]
```
A convenience accessor for the noise-free point estimate (no random
sampling) — used mostly by tests that want to check "which arm does
this bandit currently believe is better," without the randomness
`select()` deliberately introduces.

That's the entire algorithm — roughly 60 lines. Every "RL layer" this
system has (agent routing, framework picking, generation-parameter
tuning, tool-schema sizing, RAG source/algorithm selection, hybrid
local/hosted routing) is this exact class, instantiated once, with a
different set of arm-id strings, a different small context vector, and
a different reward formula plugged in by the caller. See
`README_RL_ALGORITHMS.md` for each one's specific numbers.

## Each layer, what it does and why (plain-language pass)

- **Agent routing** (`gmaofn/router.py`) — when a task could be handled
  by more than one registered agent, this decides which one gets it,
  learning over time from real outcomes which agent is actually good at
  which kind of task, instead of a fixed, hand-written assignment rule.
- **Framework picker** (`agent/framework_stats.py`) — three
  *implementations* of the identical chat algorithm exist (native,
  LangGraph, ADK); this decides which implementation answers a given
  turn, purely on measured speed/reliability, never on behavior (they're
  required to behave identically).
- **Generation-parameter tuning** (`llm/optimization_router.py`) —
  every single model call could use different sampling settings
  (temperature, max tokens); this learns, per provider, which setting
  combination tends to actually finish fast and succeed.
- **Tool-schema size** (`llm/token_optimizer.py`) — the local model is
  slow specifically because of how many tokens it has to read per
  request; this learns how aggressively it can shrink the list of tools
  offered to that one model without breaking real requests, while never
  hiding a destructive or external action.
- **RAG source + algorithm** (`gmaofn/rag_optimizer.py`) — when there's
  more than one place to look something up, and more than one way to
  rank what comes back, this learns which combination actually finds
  useful information for which kind of question.
- **Local vs hosted (Hybrid mode)** (`llm/hybrid_router.py`) — a
  user-selectable mode that learns, per message, whether this looks
  simple enough to hand to the free local model or complex enough to
  warrant paying for a hosted one.
- **Multi-agent credit** (`gmaofn/shapley.py`) — when several agents
  worked on one task together, this fairly splits the single resulting
  reward among them based on actual marginal contribution, so the
  agent-routing bandit above learns the right lesson about who deserves
  more traffic, not just "whoever went last."
- **Reward + adaptive weights** (`gmaofn/reward.py`) — turns a raw
  outcome (quality, success, cost, latency, user feedback) into the one
  number every bandit actually optimizes for, and adjusts how much each
  of those factors matters over time based on what's actually
  correlating with good outcomes.
- **Nightly PEFT fine-tune** (`scripts/finetune_local_slm.py`) — the one
  loop that isn't picking between existing options: it genuinely
  updates the local model's own weights overnight, using real reward-
  scored production turns, gated by a regression check so a bad night's
  training run can never make it to production.
