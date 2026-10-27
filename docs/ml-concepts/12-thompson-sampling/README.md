# Thompson Sampling and Contextual Bandits

## 1. The multi-armed bandit problem

A **multi-armed bandit** is the simplest possible sequential decision problem that still contains
the whole of the explore/exploit tension. You face `K` actions ("arms" — the name comes from slot
machines, one-armed bandits). Each arm `a` has some unknown reward distribution with unknown mean
`μ_a`. On each round `t = 1, 2, 3, ...`:

1. You pick an arm `a_t ∈ {1, ..., K}`.
2. You observe a reward `r_t` drawn from arm `a_t`'s distribution.
3. You update whatever beliefs you keep about the arms.

There is no state transition — unlike a full Markov Decision Process, the world doesn't move to a
different "situation" as a consequence of your action (until we add context in §5, everything looks
identical on every round except for what you've learned). The only thing you are optimizing is
cumulative reward over `T` rounds:

```
maximize  E[ Σ_{t=1}^{T} r_t ]
```

or, equivalently, minimize **regret** — the gap between what you got and what an oracle that knew
every `μ_a` in advance would have gotten by always pulling the best arm `a* = argmax_a μ_a`:

```
Regret(T) = T·μ_{a*} − E[ Σ_{t=1}^{T} r_t ]
```

The entire difficulty is that you cannot know which arm is best without trying arms you're unsure
about, and every round spent trying an inferior arm to *learn* about it is a round of reward you
didn't collect from the arm you already believe is good. That is the **exploration/exploitation
tradeoff**:

- **Exploitation**: pull the arm whose reward estimate is currently highest. Maximizes reward
  *given what you currently believe*, but if your belief is wrong (too little data, unlucky early
  samples) you can get stuck exploiting a mediocre arm forever.
- **Exploration**: pull an arm to reduce uncertainty about it, even though your current best guess
  says another arm is better. Necessary to ever discover that a seemingly-average arm is actually
  the best one, but costly if you explore an arm that really is bad.

A bandit algorithm's whole job is to trade these off intelligently, and its quality is judged by
regret — a good algorithm has *sublinear* regret (e.g. `O(log T)` or `O(√T))`, meaning the average
regret per round shrinks to zero as you gather more data. An algorithm with *linear* regret is no
better, asymptotically, than a fixed non-adaptive policy.

## 2. Simple baselines: epsilon-greedy and UCB

Before getting to Thompson Sampling, it's worth working through two standard baselines and being
honest about why each one is unsatisfying — that's what motivates the Bayesian approach.

### 2.1 Epsilon-greedy

Maintain a running sample-mean reward estimate for each arm, `Q_a = (1/n_a) Σ observed rewards for a`.
On each round, flip a coin with probability `ε`:

```
a_t = { uniform random arm,        with probability ε
       { argmax_a Q_a,             with probability 1 − ε
```

This is trivial to implement and it does explore forever, which guarantees every arm keeps getting
sampled and `Q_a → μ_a` for all arms. But it has two structural weaknesses:

- **It explores blindly.** When it decides to explore (probability `ε`), it picks *uniformly* among
  all arms — including arms it is already extremely confident are bad. A 100-arm bandit where 99
  arms have been sampled a thousand times each and are obviously terrible still wastes `ε` fraction
  of all future rounds sampling any of them with equal probability. Exploration effort is not
  directed toward the arms that are actually still uncertain.
- **`ε` doesn't decay on its own.** A fixed `ε` gives linear regret in the limit (you keep exploring
  a constant fraction of forever, even once you've essentially solved the problem). Practitioners
  patch this with an annealing schedule (`ε_t = 1/t`, etc.), but now you've introduced a
  hyperparameter — and the schedule — that needs tuning per problem, with no principled way to pick
  it from first principles.

### 2.2 Upper Confidence Bound (UCB)

UCB1 (Auer, Cesa-Bianchi & Fischer, 2002) instead builds an **optimistic** estimate of each arm's
value — its sample mean plus a bonus that shrinks as the arm gets pulled more:

```
UCB_a(t) = Q_a  +  c · sqrt( ln(t) / n_a )
```

where `n_a` is the number of times arm `a` has been pulled so far, `t` is the total round count, and
`c` is a tunable exploration-strength constant. Pick `a_t = argmax_a UCB_a(t)`. The bonus term is
large when `n_a` is small (little data → wide confidence interval → optimistic) and shrinks toward
zero as `n_a → ∞` (lots of data → the estimate converges to `Q_a` itself). This is the "optimism in
the face of uncertainty" principle: arms are given the benefit of the doubt in proportion to how
little you know about them, so uncertain arms naturally get explored, and that exploration
automatically tapers off as evidence accumulates. UCB1 achieves the theoretically optimal
`O(log T)` regret for the classic stochastic bandit.

Its weaknesses are different in character from epsilon-greedy's:

- **It's deterministic, not just in expectation but literally.** Given the same history, UCB always
  makes the same choice. That's a real practical liability in any world with hidden structure or
  correlated arms — it can also make UCB brittle to model misspecification, since there's no
  built-in randomness to smooth over a bonus term that's a slightly wrong shape for the actual
  reward distribution.
- **The confidence bound is a hand-derived worst-case bound**, typically from a concentration
  inequality (Hoeffding's, in UCB1's case) that assumes rewards are bounded and models "how wide
  should this confidence interval be" in a way that's provably safe for *any* distribution in that
  bound's assumption class. That safety margin is exactly its cost: the bound is often needlessly
  wide for the *actual* distribution in front of you, making UCB systematically overconfident about
  how uncertain it should behave (it over-explores relative to what the true posterior uncertainty
  would justify). Extending it correctly to more complex settings — non-Bernoulli rewards, or
  richer models like linear regression over a context vector — requires re-deriving a new
  concentration bound and constant `c` for every new model class; there is no automatic way to
  "plug in a different reward distribution."

Both baselines work reasonably well and both are used in production systems, but neither one
*naturally* generalizes its exploration behavior to match the actual shape of the algorithm's
uncertainty. Thompson Sampling gets that behavior for free, from the model itself.

## 3. Thompson Sampling: probability matching

Thompson Sampling (Thompson, 1933 — genuinely the oldest of the three algorithms here, though it
took until the 2010s for its regret bounds to be proven) takes a fully Bayesian view. Instead of a
single point estimate `Q_a` per arm, maintain a full **posterior probability distribution** over
each arm's true (unknown) reward parameter `θ_a`, given everything observed so far:

```
P(θ_a | data so far)
```

On each round:

1. For every arm `a`, draw one sample `θ̃_a ~ P(θ_a | data)` from that arm's *current* posterior.
2. Pick the arm with the highest sampled value: `a_t = argmax_a θ̃_a` (or, in the reward-model form
   used below, `argmax_a` of the *predicted reward* implied by that sampled parameter).
3. Observe the real reward `r_t`, and Bayes-update arm `a_t`'s posterior with it. Every other arm's
   posterior is untouched — you only learn about the arm you actually pulled.

This decision rule is called **probability matching**: because `a_t` is chosen by drawing a random
sample from each posterior and taking the max, the probability that arm `a` gets selected on this
round is *exactly* the posterior probability that `a` is the true best arm:

```
P(a_t = a | data) = P(θ_a is the argmax among all θ_1, ..., θ_K | data)
```

That single sampling step is what elegantly resolves the exploration/exploitation split without any
`ε` or hand-derived bonus constant `c`:

- An arm with a **narrow, high-mean posterior** (you've pulled it a lot and it consistently pays
  off) will sample a value close to its mean almost every time, so it wins almost every round —
  that's exploitation, arising for free because a confident posterior barely varies from sample to
  sample.
- An arm with a **wide, uncertain posterior** (you've pulled it rarely, or its outcomes have been
  noisy) will occasionally sample a very high value purely from that width, even if its posterior
  *mean* is mediocre — and when it does, it wins the round and gets pulled, which is exactly the
  exploration you want, and it happens exactly as often as the posterior's own uncertainty
  justifies. As more data arrives, the posterior narrows, the sampled values cluster tighter around
  the true mean, and exploration of that arm tapers off automatically — no schedule required, no
  `ε` to anneal, no bound to re-derive.

There is no free tunable knob standing between "how much should I explore" and "what my model of
uncertainty actually says" — the two are the same object. That single property is Thompson
Sampling's whole appeal relative to §2's baselines, and it is also why it generalizes so cleanly:
whatever the reward model is (Bernoulli, Gaussian, linear regression, ...), as long as you can
maintain *some* posterior over its parameters and sample from it, you get an exploration strategy
matched to that exact model, automatically.

## 4. The classic case: Beta-Bernoulli bandits

The canonical worked example is a bandit where each pull of arm `a` returns a binary reward
`r ∈ {0, 1}` (e.g. click / no-click, success / failure), with unknown true success probability `p_a`.

### 4.1 The model

- **Likelihood**: `r | p_a ~ Bernoulli(p_a)`, i.e. `P(r = 1 | p_a) = p_a`, `P(r = 0 | p_a) = 1 − p_a`.
- **Prior**: model `p_a` itself as a random variable with a `Beta(α_a, β_a)` distribution:

```
P(p_a) = p_a^(α_a − 1) · (1 − p_a)^(β_a − 1) / B(α_a, β_a)
```

  where `B(α, β)` is the Beta function (a normalizing constant). `Beta(1, 1)` is the uniform
  distribution on `[0, 1]` and is the standard uninformative starting prior for each arm before any
  data is seen. Intuitively `α_a` counts "successes so far, plus a pseudo-count" and `β_a` counts
  "failures so far, plus a pseudo-count," and the mean of `Beta(α, β)` is `α / (α + β)`.

### 4.2 Why Beta is conjugate to Bernoulli

"Conjugate" means: if the prior is `Beta(α, β)` and the likelihood is Bernoulli, the *posterior*
after observing one data point is again a Beta distribution, in closed form, with no numerical
integration required. Here's the derivation. By Bayes' rule, the posterior after observing a single
reward `r` is proportional to prior times likelihood:

```
P(p | r) ∝ P(r | p) · P(p)
```

If `r = 1` (success):

```
P(p | r=1) ∝ p · [ p^(α−1) (1−p)^(β−1) ]  =  p^α (1−p)^(β−1)
```

which is exactly the unnormalized form of `Beta(α + 1, β)`. If `r = 0` (failure):

```
P(p | r=0) ∝ (1−p) · [ p^(α−1) (1−p)^(β−1) ]  =  p^(α−1) (1−p)^β
```

which is `Beta(α, β + 1)`. Both cases collapse into one update rule, since `r ∈ {0, 1}`:

```
α ← α + r
β ← β + (1 − r)
```

Because the posterior is Beta again — the same distributional *family* as the prior, just with
updated parameters — you never need to run any general-purpose Bayesian inference (no MCMC, no
variational approximation, no numerical integration over `p`). Each observation is a two-line
increment. This is the whole appeal of conjugate priors, and it's why Beta-Bernoulli is the
"hello world" of Bayesian bandits: the update is `O(1)` per observation and exact, not
approximate.

### 4.3 The full algorithm

```
Initialize α_a = 1, β_a = 1 for all arms a      # uniform prior

for each round t:
    for each arm a:
        sample  p̃_a ~ Beta(α_a, β_a)
    pull arm  a_t = argmax_a p̃_a
    observe reward r_t ∈ {0, 1}
    α_{a_t} ← α_{a_t} + r_t
    β_{a_t} ← β_{a_t} + (1 − r_t)
```

That's the entire algorithm — a few lines, no gradient descent, no learning rate, and it provably
achieves near-optimal (`O(log T)`) regret (Agrawal & Goyal, "Analysis of Thompson Sampling for the
Multi-Armed Bandit Problem," COLT 2012; Kaufmann, Korda & Munos, ALT 2012).

## 5. Contextual bandits

The plain multi-armed bandit assumes every round looks the same. Real routing problems usually
don't — the best arm to pick often depends on something you observe *before* choosing, called the
**context** (a feature vector `x ∈ ℝ^d` describing this round: a user's attributes, a query's
properties, anything). This is the **contextual bandit**: on each round you observe `x_t`, choose
an arm, and the reward distribution for that arm depends on `x_t`.

### 5.1 The linear reward model

The standard way to make this tractable is to assume each arm's expected reward is a *linear*
function of the context:

```
r = xᵀ θ_a + ε          ε ~ N(0, σ²) noise
```

where `θ_a ∈ ℝ^d` is an arm-specific, unknown weight vector to be learned. This is exactly linear
regression — one regression model per arm, mapping context to expected reward — and the bandit
problem becomes "learn each arm's `θ_a` online, while also deciding which arm to pull, using as few
suboptimal pulls as possible."

Two well-known ways to do this:

- **LinUCB** (Li, Chu, Langford & Schapire, WWW 2010) is the UCB idea (§2.2) lifted to this linear
  setting: compute a confidence *ellipsoid* around each arm's estimated `θ̂_a` (from ridge
  regression) and pick the arm maximizing `xᵀθ̂_a + α·sqrt(xᵀ A_a⁻¹ x)` — the point estimate plus an
  optimistic uncertainty bonus shaped by the context. It's deterministic, same as UCB1, with the
  same strengths and same brittleness discussed in §2.2, just generalized to vector-valued
  parameters.
- **Bayesian linear bandit with Thompson Sampling**: maintain a genuine Bayesian posterior over
  `θ_a`, typically multivariate Gaussian, and *sample* a full weight vector on every round rather
  than computing a bound.

### 5.2 Bayesian linear regression posterior, worked out

Assume a Gaussian prior `θ_a ~ N(0, λ⁻¹ I)` and Gaussian noise `r = xᵀθ_a + ε`, `ε ~ N(0, 1)` WLOG
(any noise variance folds into `λ`). After observing a sequence of `(x_i, r_i)` pairs for arm `a`,
the posterior over `θ_a` is again Gaussian (Gaussian is self-conjugate for linear-Gaussian models,
the same closed-form-update story as §4.2, just in matrix form):

```
θ_a | data  ~  N( μ_a ,  ν² B_a⁻¹ )

where   B_a = Σ_i x_i x_iᵀ + λI          (a d×d matrix — "information"/precision-like matrix)
        f_a = Σ_i r_i · x_i               (a d-vector — reward-weighted evidence)
        μ_a = B_a⁻¹ f_a                   (posterior mean weight vector)
```

`λI` is the ridge/prior term (keeps `B_a` invertible even with little data, and pulls `μ_a` toward
0 for an arm with no observations yet); `ν` is an exploration-scale multiplier on the posterior
covariance (Agrawal & Goyal's analysis sets `ν` from a formula involving `d`, `T`, and a failure
probability `δ` to get worst-case regret guarantees — in practice it's usually treated as a tunable
constant, since the "safe" theoretical value is typically far larger than what's used in practice).

Each new observation `(x, r)` for arm `a` updates the sufficient statistics in closed form, again
`O(d²)` per update with no re-fitting from scratch:

```
B_a ← B_a + x xᵀ
f_a ← f_a + r · x
```

Thompson Sampling's decision rule now samples an entire *vector* rather than a scalar:

```
for each arm a:
    θ̃_a ~ N(μ_a, ν² B_a⁻¹)        # sample a full plausible regression model
    ŷ_a = xᵀ θ̃_a                  # predicted reward for THIS context under that sampled model
pull arm  a_t = argmax_a ŷ_a
```

This is the exact generalization of §4's `p̃_a ~ Beta(α_a, β_a)`: instead of sampling a single
success probability, you sample an entire plausible weight vector consistent with everything
observed for that arm, then use it to predict a reward for the specific context in front of you
right now. An arm that's barely been tried has a wide covariance `ν² B_a⁻¹` (since `B_a ≈ λI` is
small in the directions with no data), so its sampled `θ̃_a` — and hence `ŷ_a` — varies a lot from
round to round, occasionally landing high enough to win and get explored. An arm with lots of data
in the relevant region of context-space has a tight covariance there and consistently predicts close
to its true mean.

## 6. The Thompson Sampling loop

```
   ┌────────────────────────────────────────────────────────────────┐
   │                         ROUND t BEGINS                         │
   └────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  observe context vector  x_t   │
                  └───────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │  FOR EACH ARM a IN {1 .. K}:                         │
        │                                                      │
        │   ┌────────────────────────┐   ┌───────────────────┐ │
        │   │ posterior P(θ_a|data)  │──▶│ sample θ̃_a ~ P(·)  │ │
        │   └────────────────────────┘   └───────────────────┘ │
        │                                        │              │
        │                                        ▼              │
        │                          ┌──────────────────────────┐ │
        │                          │ predict ŷ_a = f(x_t, θ̃_a) │ │
        │                          └──────────────────────────┘ │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │  pick  a_t = argmax_a  ŷ_a         │
                └───────────────────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │  execute action, observe reward r_t │
                └───────────────────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │  Bayes-update ONLY arm a_t's        │
                │  posterior P(θ_{a_t} | data ∪ r_t)  │
                └───────────────────────────────────┘
                                  │
                                  ▼
                     (loop back to ROUND t+1)
```

Note the two things that keep this cheap and self-correcting: only the *chosen* arm's posterior is
touched per round (unchosen arms carry their uncertainty forward unchanged — correct, since you
learned nothing new about them), and the whole exploration behavior is an emergent consequence of
sampling from real posteriors, not a separately engineered schedule.

## 7. How this repository actually uses it

Everything above is the textbook version. This section describes precisely what
`orchestrator_framework_layer/akanade/backend/app/framework/bandit.py` and
`orchestrator_framework_layer/akanade/backend/app/llm/hybrid_router.py` implement, with file:line
citations for every claim, so it's clear where the real code matches §5.2 and where it takes a
specific engineering shortcut.

### 7.1 `LinearBanditCore` — a Bayesian linear bandit, confirmed by direct reading

`LinearBanditCore` (`bandit.py:20-103`) is exactly the Bayesian linear regression Thompson Sampling
primitive of §5.2, not LinUCB and not a plain multi-armed (context-free) bandit. Concretely:

- **State per arm** (`bandit.py:32-33`): a `d×d` matrix `self._B[arm_id]` and a `d`-vector
  `self._f[arm_id]`, one pair per arm — exactly `B_a` and `f_a` from §5.2. `d = feature_dim + 1`
  (`bandit.py:29`), the `+1` being a bias/intercept term appended to every context vector by
  `features()` (`bandit.py:46-48`, `context_vector + [1.0]`).
- **Initialization** (`ensure()`, `bandit.py:36-44`): `B_a` starts at `ridge * I` (the `λI` prior
  term of §5.2, `bandit.py:39`), and `f_a` starts at a small vector derived from an optional
  `seed_utility` (`bandit.py:44`) rather than plain zero — this is the "warm start from a
  hand-designed prior" the module's own docstring describes (`bandit.py:40-43`): it nudges the
  posterior *mean* toward a heuristic guess while leaving the posterior's *width* (`B_a = λI`,
  maximally uncertain) untouched, so a few real observations can quickly override it.
- **Sampling and scoring** (`score()`, `bandit.py:50-61`): computes `B_inv = B_a⁻¹`
  (`bandit.py:57`), the posterior mean `mu = B_inv @ f` (`bandit.py:58`, i.e. `μ_a = B_a⁻¹f_a`
  exactly as in §5.2), and the sampling covariance `cov = ν² · B_inv` (`bandit.py:59`, where
  `self._nu` is `exploration_scale`, i.e. `ν`). It then draws
  `theta = rng.multivariate_normal(mu, cov)` (`bandit.py:60`) — a genuine Gaussian posterior sample
  `θ̃_a ~ N(μ_a, ν²B_a⁻¹)`, not an upper-confidence-bound calculation — and returns both the sampled
  predicted reward `theta @ x` and the posterior-mean predicted reward `mu @ x` (`bandit.py:61`).
- **Arm selection** (`select()`, `bandit.py:63-72`): scores every candidate arm with its own
  independently-sampled `θ̃_a` and returns `argmax` (`bandit.py:70-71`) — this is precisely §3's and
  §5.2's Thompson Sampling decision rule ("score every arm with its OWN fresh posterior sample,
  argmax," as the method's own docstring at `bandit.py:65-66` states), reusable for any number of
  named arms.
- **Posterior update** (`update()`, `bandit.py:74-78`): `self._B[arm_id] += np.outer(x, x)`
  (`bandit.py:77`) and `self._f[arm_id] += reward * x` (`bandit.py:78`) — exactly `B_a ← B_a + xxᵀ`
  and `f_a ← f_a + r·x` from §5.2, an `O(d²)` closed-form update with no refitting and no gradient
  descent. Only the arm actually passed to `update()` is touched.
- **Persistence** (`to_state()` / `load_state()`, `bandit.py:85-102`): the entire learned state is
  just each arm's `B` matrix and `f` vector serialized as nested lists, confirming there is no
  hidden model beyond these two per-arm sufficient statistics — this genuinely is Bayesian linear
  regression and nothing more elaborate.

`bandit.py:11-13`'s own docstring names the exact reference this implements: Agrawal & Goyal,
"Thompson Sampling for Contextual Bandits with Linear Payoffs," ICML 2013 — confirmed independently
by `framework/router.py:1-20`, the other real consumer of this same `LinearBanditCore`, whose module
docstring spells out the identical `N(μ_a, ν²B_a⁻¹)` posterior and the same `B_a = Σxxᵀ + λI`,
`f_a = Σr·x` sufficient statistics for its own use case (agent selection). So: **this is a genuine
Bayesian linear bandit solved with Thompson Sampling, not LinUCB and not a bare Beta-Bernoulli
bandit** — the reward model is linear-Gaussian over a real-valued context vector, matching §5.2
of this document exactly, down to the ridge prior and the exploration-scale-on-covariance
parameterization.

### 7.2 `HybridRouter` — two arms, a 3-feature context, asymmetric rewards

`HybridRouter` (`hybrid_router.py:77-108`) is a thin, task-specific wrapper around one
`LinearBanditCore` instance (`hybrid_router.py:81`, constructed with `_FEATURE_DIM = 3`,
`hybrid_router.py:48`) used to decide, per chat message, whether to prefer the free local Ollama
SLM or let the hosted-model fallback chain run normally. It has exactly two arms, the literal
strings `"local"` and `"hosted"` (`hybrid_router.py:89`) — never any other arm id.

**Context construction** — `_complexity_features()` (`hybrid_router.py:57-64`) builds a 3-dimensional
feature vector purely from cheap, non-LLM signals on the raw message text and the calling tier:

```python
def _complexity_features(message: str, tier: str) -> list[float]:
    words = message.split()
    clauses = len(re.split(r"[,;]| and | or ", message))
    return [
        min(len(words) / 50.0, 1.0),           # word count, clamped to [0, 1] at 50 words
        min(clauses / 4.0, 1.0),                # clause count, clamped to [0, 1] at 4 clauses
        1.0 if tier == "delay_ok" else 0.0,     # 1.0 iff the calling tier already tolerates delay
    ]
```

(`hybrid_router.py:57-64`). All three features are pure functions of information already on hand
before any model call — message length, a crude clause count from splitting on commas/semicolons/
"and"/"or", and a binary flag for whether the tier's own classifier already decided this request can
tolerate extra latency — deliberately avoiding a second LLM round-trip just to estimate complexity
(as the module docstring states at `hybrid_router.py:14-17`). `LinearBanditCore.features()`
(`bandit.py:46-48`) appends the `+1.0` bias term on top of these 3, making the true regression
dimension `d = 4` for this router.

**Arm selection** — `choose()` (`hybrid_router.py:84-90`):

```python
def choose(self, message: str, tier: str, local_provider: str = _LOCAL_PROVIDER_KEY) -> HybridSelection:
    context = _complexity_features(message, tier)
    if not _local_provider_available(local_provider):
        return HybridSelection(arm="hosted", context=context)
    with self._lock:
        winner, _ = self._core.select(["local", "hosted"], context)
    return HybridSelection(arm=winner, context=context)
```

(`hybrid_router.py:84-90`). Two things worth calling out precisely: first, there is a hard override
*before* the bandit ever runs — if the local Ollama provider isn't currently available
(`_local_provider_available()`, `hybrid_router.py:67-68`, checking `available_providers()` from
`router.py`), `choose()` returns `"hosted"` immediately (`hybrid_router.py:87`) without calling
`select()` at all, so an unavailable local model is never sampled as a candidate. Second, when the
bandit does run, it is the literal same `select(["local", "hosted"], context)` call from §7.1 —
each of the two arms gets its own fresh `θ̃_a ~ N(μ_a, ν²B_a⁻¹)` sample scored against this round's
3(+1)-dimensional context, and the winner is whichever arm's sampled predicted reward is higher
(`bandit.py:70-71`). The `threading.Lock` (`hybrid_router.py:82`, held across the `select()` call at
`hybrid_router.py:88-89`) exists because `hybrid_router` is a process-wide singleton
(`hybrid_router.py:112`) shared across concurrent chat requests, and both `_B`/`_f` mutation
(`update()`) and read-then-sample (`select()`) need to be serialized to avoid a torn read of the
posterior matrices.

Constructor defaults are `ridge=1.0, exploration_scale=0.4` (`hybrid_router.py:80-81`) — i.e. `λ = 1`
and `ν = 0.4` in §5.2's notation, both passed straight through to `LinearBanditCore.__init__`
(`bandit.py:27-34`).

**Feedback / posterior update** — `record_outcome()` (`hybrid_router.py:102-108`):

```python
def record_outcome(self, selection: HybridSelection, *, success: bool) -> None:
    if selection.arm == "local":
        reward = 1.0 if success else -0.6
    else:
        reward = 0.5 if success else -0.3
    with self._lock:
        self._core.update(selection.arm, selection.context, max(-1.0, min(1.0, reward)))
```

(`hybrid_router.py:102-108`). This is the only place a reward signal is ever computed, and it is
**deliberately asymmetric across the two arms**, exactly as the module docstring states at
`hybrid_router.py:19-24`: a successful local answer is worth `+1.0` (free and it worked — the best
outcome), a successful hosted answer only `+0.5` (it worked, but cost money), a failed hosted
attempt is `-0.3`, and a failed local attempt is punished hardest at `-0.6` (routing something local
couldn't handle wastes a whole turn a hosted model would likely have gotten right first try). The
reward is clamped to `[-1, 1]` (`hybrid_router.py:108`, redundant here since all four literal values
already sit inside that range, but consistent with the shared convention `router.py:69` documents
for reward scale across this codebase) and fed into `LinearBanditCore.update(arm, context, reward)`
(`bandit.py:74-78`), which applies the exact `B_a ← B_a + xxᵀ`, `f_a ← f_a + r·x` update from §5.2
to whichever single arm (`"local"` or `"hosted"`) was actually selected for that turn — the other
arm's posterior is left untouched, matching §6's diagram precisely ("Bayes-update ONLY arm a_t's
posterior").

Two points that matter for interpreting this as "real" Thompson Sampling rather than something
adjacent to it: (1) the *arm identity itself* — `"local"` vs `"hosted"` — carries no features of its
own; the context vector describes the *message*, not the arm, so what's actually being learned is
two separate regression surfaces, `θ_local` and `θ_hosted`, each mapping "how complex does this
message look" to "how rewarding has that arm been on messages like this," and picking whichever one
currently predicts higher for the message in front of it. (2) The asymmetric reward function itself
is a fixed, hand-authored utility design choice (§7.2's four constants) that sits entirely upstream
of the bandit — the bandit's job starts only once a scalar reward has already been decided; it has
no notion of "cost" or "wasted turn" beyond the number it's handed.

Also worth noting for completeness (mentioned in `hybrid_router.py:26-37`'s docstring, though outside
`bandit.py`/`hybrid_router.py` proper): if the "local" arm is chosen but the SLM doesn't answer within
`LOCAL_TIMEOUT_SECONDS = 10.0` (`hybrid_router.py:54`), `TierClient`'s existing fallback logic
escalates to a hosted rung for that turn regardless of what the bandit picked, and that escalation is
separately logged as a fine-tuning example — but that escalation path does not touch the bandit's
posterior directly; it's a downstream consumer of `HybridSelection`, not a different reward path
inside `record_outcome()`.

### 7.3 Summary of what's real vs. textbook here

| Textbook concept (§1-§6) | This repo's implementation |
|---|---|
| Arm `a ∈ {1..K}` | Exactly two arms, `"local"` and `"hosted"` (`hybrid_router.py:89`) |
| Context `x ∈ ℝ^d` | 3 heuristic features + bias, `d=4` (`hybrid_router.py:57-64`, `bandit.py:29,46-48`) |
| Reward model `r = xᵀθ_a + ε` | Same, one linear-Gaussian model per arm (`bandit.py:50-61`) |
| Posterior `N(μ_a, ν²B_a⁻¹)` | Implemented verbatim (`bandit.py:57-60`) |
| Prior / warm start | `B_a=λI`, `f_a` seeded from `seed_utility` (`bandit.py:36-44`); `HybridRouter` passes no seed, so effectively `f_a=0` at first use |
| Sample & argmax | `select()` (`bandit.py:63-72`), called by `choose()` (`hybrid_router.py:88-89`) |
| Bayes update | `update()` (`bandit.py:74-78`), called by `record_outcome()` (`hybrid_router.py:108`) |
| Reward signal | Hand-authored, asymmetric, arm-dependent constants: `+1.0 / -0.6` (local), `+0.5 / -0.3` (hosted) (`hybrid_router.py:104-106`) |

No part of this is epsilon-greedy or UCB/LinUCB — there is no random-uniform exploration branch and
no explicit confidence-bound term computed anywhere in either file; every exploratory decision comes
from the width of the sampled Gaussian in `score()` (`bandit.py:59-60`), which is exactly the
mechanism described in §5.2 and diagrammed in §6.

## 8. Practical notes and caveats

- **Sample efficiency vs. model correctness.** Thompson Sampling's regret guarantees rely on the
  posterior being a reasonably accurate description of the true reward-generating process. If the
  true reward isn't actually linear in the chosen features (a real risk with a heuristic 3-feature
  context like word/clause count), the algorithm still runs and still explores/exploits sensibly
  relative to *its own* (possibly wrong) posterior, but the asymptotic optimality guarantees no
  longer strictly apply — it becomes a heuristic with good qualitative behavior rather than a
  regret-optimal procedure. This is a reasonable trade for a cheap, no-extra-model-call heuristic.
- **Non-stationarity.** Every derivation above assumes each arm's true parameter `θ_a` is fixed for
  all time. In production, a hosted provider's latency/quality or a local model's competence can
  shift over time (a model upgrade, a load spike). Plain Bayesian linear regression accumulates
  evidence forever and reacts to a distribution shift only slowly, in proportion to how much old
  data still dominates `B_a`/`f_a`. Common fixes not present in this code (worth flagging as absent,
  not assuming they're implicitly handled) include a forgetting factor that decays `B_a`/`f_a`
  over time, or a sliding window of recent observations only.
- **Cold start.** `LinearBanditCore.ensure()`'s `seed_utility` parameter (`bandit.py:36-44`) exists
  precisely to avoid the classic bandit cold-start problem — an arm with zero data has a maximally
  wide posterior and would otherwise be picked almost at random on its very first exposures, which
  is fine in a big-K low-stakes setting but wasteful when a hand-designed prior (like
  `router.py`'s `UtilityWeights`, `router.py:47-92`) is cheaply available. `HybridRouter` does not
  pass a `seed_utilities` argument to `select()` (`hybrid_router.py:89` calls it with only two
  positional arguments), so both `"local"` and `"hosted"` start from an unseeded, zero-mean,
  maximally-uncertain posterior — the very first few real turns are effectively decided by
  posterior width alone (near coin-flip odds), and the bandit only starts differentiating the two
  arms once genuine outcomes accumulate.
