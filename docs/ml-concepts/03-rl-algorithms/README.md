# Reinforcement Learning Algorithm Family Tree (for LLMs)

This document traces the lineage of reinforcement learning (RL) algorithms that
matter for training large language models, from first-principles RL vocabulary
through bandits, policy gradients, actor-critic methods, PPO, full RLHF, and
finally DPO. The goal is to make the math behind "the model was fine-tuned
with RLHF" or "we used DPO instead" legible line by line.

Related documents in this set:

- `docs/ml-concepts/12-thompson-sampling/README.md` — Thompson Sampling is a
  Bayesian bandit algorithm used elsewhere in this doc set (e.g. for
  exploration in routing/allocation problems). It is only mentioned here for
  completeness in the bandit section; its derivation lives in that README.

---

## 1. Core RL Vocabulary

Reinforcement learning formalizes sequential decision-making under
uncertainty. The standard formalism is the **Markov Decision Process (MDP)**,
defined by a tuple:

```
MDP = (S, A, P, R, gamma)

S      : set of states
A      : set of actions
P(s'|s,a) : transition probability - probability of landing in state s'
            after taking action a in state s
R(s,a) or R(s,a,s') : reward function
gamma  : discount factor, 0 <= gamma <= 1
```

### 1.1 Agent and Environment

The **agent** is the decision-maker (e.g., the LLM being trained). The
**environment** is everything the agent interacts with and cannot directly
control (e.g., a user prompt, a simulator, a game). At each discrete timestep
`t`:

```
1. Agent observes state s_t
2. Agent selects action a_t ~ pi(a_t | s_t)
3. Environment transitions to s_{t+1} ~ P(s_{t+1} | s_t, a_t)
4. Environment emits reward r_t = R(s_t, a_t)
5. Repeat
```

This loop produces a **trajectory** (also called a rollout or episode):

```
tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)
```

### 1.2 State, Action, Reward

- **State `s`**: a (Markovian) description of the environment at time `t`.
  For an LLM being fine-tuned with RL, the "state" is typically the prompt
  plus all tokens generated so far (the partial completion).
- **Action `a`**: a choice the agent makes. For an LLM, the action space is
  the vocabulary — each generated token is an action.
- **Reward `r`**: a scalar signal indicating how good the outcome was. For
  RLHF, the reward is typically produced by a learned reward model (see
  Section 6) and often only given at the *end* of a full generation
  (sparse, terminal reward), rather than per-token.

### 1.3 Policy `pi(a|s)`

The **policy** is the agent's decision-making function — a (possibly
stochastic) mapping from states to a distribution over actions:

```
pi(a|s) = P(action = a | state = s)
```

For a neural network policy parameterized by `theta` (e.g., the weights of
an LLM), we write `pi_theta(a|s)`. For an LLM, `pi_theta(a_t | s_t)` is
literally the softmax probability the model assigns to the next token
`a_t`, given the context `s_t` (prompt + tokens generated so far).

A policy can be:
- **Deterministic**: `a = mu(s)` — a direct function from state to action.
- **Stochastic**: `a ~ pi(.|s)` — a distribution over actions, which is the
  standard for LLMs (sampling from the token distribution).

### 1.4 Value Functions

Value functions estimate *expected future return* from a state (or
state-action pair), under a given policy. They are the tool used to reduce
variance and to assign credit across long trajectories.

**Return** — the discounted sum of future rewards from time `t`:

```
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... = sum_{k=0}^{inf} gamma^k * r_{t+k}
```

**State-value function** — expected return starting from state `s`,
following policy `pi` thereafter:

```
V_pi(s) = E_{tau ~ pi} [ G_t | s_t = s ]
```

**Action-value function (Q-function)** — expected return starting from
state `s`, taking action `a`, then following `pi`:

```
Q_pi(s,a) = E_{tau ~ pi} [ G_t | s_t = s, a_t = a ]
```

**Advantage function** — how much better action `a` is than the policy's
average behavior in state `s`. This quantity is central to variance
reduction in policy gradients (Sections 3-4):

```
A_pi(s,a) = Q_pi(s,a) - V_pi(s)
```

### 1.5 The RL Objective

The fundamental objective of reinforcement learning is to find policy
parameters `theta` that **maximize expected cumulative (discounted) reward**
over trajectories induced by that policy:

```
J(theta) = E_{tau ~ pi_theta} [ sum_{t=0}^{T} gamma^t * r_t ]
         = E_{tau ~ pi_theta} [ G_0 ]
```

Everything that follows — bandits, REINFORCE, actor-critic, PPO, RLHF, DPO
— is a different strategy for estimating `J(theta)` and its gradient
`grad_theta J(theta)`, and then climbing that gradient.

---

## 2. Multi-Armed Bandits: RL Without State Transitions

The **multi-armed bandit (MAB)** problem is the simplest special case of RL.
It strips away state transitions entirely: there is one state (or the state
is irrelevant/i.i.d. each round), a fixed set of `K` actions ("arms"), and
an immediate reward for each action. There is no `P(s'|s,a)` to learn
because the "episode" is a single step — pull an arm, get a reward, repeat.
This isolates the **exploration vs. exploitation** tradeoff from the
credit-assignment problem that full RL must also solve.

Formally: at each round `t`, choose arm `a_t in {1, ..., K}`, and receive
reward `r_t ~ D_{a_t}`, where `D_a` is the (unknown) reward distribution of
arm `a`. The goal is to minimize **regret**:

```
Regret(T) = sum_{t=1}^{T} ( mu* - mu_{a_t} )

mu*    = max_a E[D_a]        (the best arm's mean reward)
mu_a_t = E[D_{a_t}]          (mean reward of the arm actually pulled)
```

### 2.1 Epsilon-Greedy

Maintain an estimate `Q_hat(a)` of each arm's mean reward (e.g., a running
sample average of observed rewards for arm `a`). At each step:

```
with probability epsilon:      a_t = random arm (explore)
with probability 1 - epsilon:  a_t = argmax_a Q_hat(a) (exploit)
```

`Q_hat(a)` is updated incrementally after pulling arm `a` and observing
reward `r`:

```
N(a) <- N(a) + 1
Q_hat(a) <- Q_hat(a) + (1 / N(a)) * ( r - Q_hat(a) )
```

Epsilon-greedy is simple but crude: it explores *uniformly at random*
regardless of how much is already known about an arm, and `epsilon` must be
hand-tuned or decayed over time (e.g., `epsilon_t = 1/t`) to achieve
sublinear regret.

### 2.2 Upper Confidence Bound (UCB)

UCB replaces random exploration with **optimism in the face of
uncertainty**: prefer arms whose estimated value has high upper-confidence
uncertainty, so under-explored arms get a boost proportional to how little
they've been tried. The canonical UCB1 action-selection rule:

```
a_t = argmax_a [ Q_hat(a) + c * sqrt( ln(t) / N(a) ) ]
```

- `Q_hat(a)`: current mean-reward estimate for arm `a`.
- `N(a)`: number of times arm `a` has been pulled so far.
- `t`: total number of rounds so far.
- `c`: exploration coefficient controlling how aggressively to explore.

The bonus term `c * sqrt(ln(t)/N(a))` shrinks as `N(a)` grows (an arm
pulled often is well-estimated, so its confidence interval narrows) and
grows with `ln(t)` (giving every arm a periodically renewed incentive to be
re-checked). UCB1 achieves logarithmic regret `O(ln T)` under standard
assumptions, versus epsilon-greedy's typically worse (often linear if
`epsilon` isn't decayed) regret.

### 2.3 Thompson Sampling — See Separate Doc

A third classic bandit strategy, **Thompson Sampling**, treats each arm's
reward distribution as a Bayesian posterior and samples from that posterior
to decide which arm to pull. It is not derived here — see
`docs/ml-concepts/12-thompson-sampling/README.md` for its full derivation
and use elsewhere in this documentation set.

### 2.4 Why Bandits Matter for the LLM RL Story

Bandits establish the exploration/exploitation vocabulary and the idea of
learning from scalar reward feedback with no ground-truth supervision —
both of which reappear, dramatically scaled up, in RLHF: there, "arms" are
effectively the space of possible LLM completions, and the "reward" comes
from a learned reward model rather than i.i.d. draws from `D_a`. The
critical complication that full RL (and RLHF) adds on top of bandits is
**state**: in an LLM rollout, generating token `a_t` changes the context
`s_{t+1}` for every subsequent token, so credit for the eventual reward
must be propagated back through a whole sequence of decisions rather than
attributed to one isolated action.

---

## 3. Policy Gradient Methods and REINFORCE

Once state transitions matter (multi-step trajectories), we need a way to
improve a stochastic policy `pi_theta(a|s)` using only the scalar rewards
it receives, without knowing the environment's transition dynamics
`P(s'|s,a)`. **Policy gradient methods** do this by directly computing
`grad_theta J(theta)` and performing gradient ascent — as opposed to
value-based methods (like Q-learning) that learn a value function first and
derive a policy from it.

### 3.1 Deriving the Policy Gradient (the Log-Derivative Trick)

We want `grad_theta J(theta)` where:

```
J(theta) = E_{tau ~ pi_theta} [ R(tau) ]  =  integral over tau of  P_theta(tau) * R(tau) dtau
```

Here `R(tau) = sum_t gamma^t r_t` is the total (discounted) return of
trajectory `tau`, and `P_theta(tau)` is the probability of trajectory `tau`
occurring under policy `pi_theta`.

Naively differentiating `integral P_theta(tau) R(tau) dtau` with respect to
`theta` is intractable because `R(tau)` doesn't depend on `theta`, but
`P_theta(tau)` does, and we cannot easily sample from `grad_theta
P_theta(tau)` directly (it isn't a probability distribution — it can be
negative). The **log-derivative trick** (also called the REINFORCE trick or
score-function estimator) fixes this using the identity:

```
grad_theta P_theta(tau) = P_theta(tau) * grad_theta log P_theta(tau)
```

This follows directly from the chain rule: `grad_theta log f(x) = grad_theta
f(x) / f(x)`, so `grad_theta f(x) = f(x) * grad_theta log f(x)`.

Substituting into the gradient of `J(theta)`:

```
grad_theta J(theta) = grad_theta  integral  P_theta(tau) R(tau) dtau
                     = integral  grad_theta P_theta(tau) * R(tau) dtau
                     = integral  P_theta(tau) * grad_theta log P_theta(tau) * R(tau) dtau
                     = E_{tau ~ pi_theta} [ grad_theta log P_theta(tau) * R(tau) ]
```

We have converted an intractable integral into an **expectation**, which we
can approximate with Monte Carlo samples (rollouts). Now we need
`grad_theta log P_theta(tau)`. The trajectory probability factors as:

```
P_theta(tau) = P(s_0) * prod_{t=0}^{T} pi_theta(a_t|s_t) * P(s_{t+1}|s_t,a_t)
```

Taking the log turns the product into a sum:

```
log P_theta(tau) = log P(s_0) + sum_t [ log pi_theta(a_t|s_t) + log P(s_{t+1}|s_t,a_t) ]
```

Critically, `P(s_0)` and `P(s_{t+1}|s_t,a_t)` are properties of the
**environment**, not of `theta` — the policy has no control over
transition dynamics. So their gradients with respect to `theta` are zero,
and they vanish:

```
grad_theta log P_theta(tau) = sum_{t=0}^{T} grad_theta log pi_theta(a_t|s_t)
```

This is the key result: **the gradient of the trajectory's log-probability
depends only on the policy's own action-selection log-probabilities**, not
on the (unknown, possibly non-differentiable) environment dynamics. Putting
it together gives the **REINFORCE policy gradient**:

```
grad_theta J(theta) = E_{tau ~ pi_theta} [ ( sum_{t=0}^{T} grad_theta log pi_theta(a_t|s_t) ) * R(tau) ]
```

or, per-timestep (using the fact that reward at time `t` should only be
attributed to actions at time `t' <= t`, i.e. the "reward-to-go" form):

```
grad_theta J(theta) = E [ sum_{t=0}^{T} grad_theta log pi_theta(a_t|s_t) * G_t ]

where  G_t = sum_{k=t}^{T} gamma^{k-t} * r_k     (reward-to-go from step t)
```

In practice, this gradient is estimated with a single rollout (or a batch
of `N` rollouts) via Monte Carlo:

```
grad_theta J(theta)  ~=  (1/N) * sum_{i=1}^{N} sum_{t=0}^{T} grad_theta log pi_theta(a_t^i | s_t^i) * G_t^i
```

and applied via standard gradient ascent: `theta <- theta + alpha *
grad_theta J(theta)`.

### 3.2 Why High Variance Is a Problem

REINFORCE's gradient estimator is **unbiased** (in expectation it equals
the true gradient) but suffers from **high variance**:

- `G_t` (the Monte Carlo return) is a noisy sample of a random variable —
  different rollouts from the same state and policy can yield wildly
  different total rewards, especially over long horizons (many LLM tokens)
  or with sparse/delayed rewards (a single scalar reward at the end of a
  whole generation, as in RLHF).
- Because `G_t` multiplies every `grad_theta log pi_theta(a_t|s_t)` term,
  a single unlucky (or lucky) rollout can produce a huge gradient step in
  the wrong direction, and gradient noise scales with trajectory length.
- High variance means many rollouts are needed per gradient step to get a
  reliable estimate, which is sample-inefficient — costly when each
  rollout requires an expensive LLM generation (and, in RLHF, a reward
  model forward pass).

### 3.3 Baseline Subtraction and the Advantage Function

A powerful variance-reduction trick: subtract a **baseline** `b(s_t)` from
the return, which does not change the expected value of the gradient (it
is unbiased) but can substantially reduce its variance. The unbiasedness
relies on the fact that, for any baseline function `b(s_t)` that does not
depend on `a_t`:

```
E_{a_t ~ pi_theta(.|s_t)} [ grad_theta log pi_theta(a_t|s_t) * b(s_t) ] = 0
```

Proof sketch: `E_{a~pi}[grad_theta log pi_theta(a|s)] = integral pi_theta(a|s)
* grad_theta log pi_theta(a|s) da = integral grad_theta pi_theta(a|s) da =
grad_theta integral pi_theta(a|s) da = grad_theta 1 = 0`, and multiplying
by a constant-in-`a` baseline `b(s)` preserves this zero.

So we can rewrite the gradient with a baseline subtracted, with zero bias
added:

```
grad_theta J(theta) = E [ sum_t grad_theta log pi_theta(a_t|s_t) * ( G_t - b(s_t) ) ]
```

The standard, most effective choice of baseline is the **state-value
function** `b(s_t) = V_pi(s_t)`, in which case `G_t - V_pi(s_t)` is an
estimate of the **advantage function**:

```
A_pi(s_t, a_t) = Q_pi(s_t, a_t) - V_pi(s_t)  ~=  G_t - V(s_t)
```

Intuitively, this reframes the learning signal from "how good was this
total return" (which is dominated by how good the *state* already was) to
"how much better or worse was this particular action than what the policy
would typically do from this state" — a much lower-variance, more
informative signal. This substitution:

```
grad_theta J(theta) = E [ sum_t grad_theta log pi_theta(a_t|s_t) * A_pi(s_t, a_t) ]
```

is the bridge directly into actor-critic methods.

---

## 4. Actor-Critic Methods

**Actor-critic** methods combine two learned components:

- **Actor**: the policy `pi_theta(a|s)`, updated via the policy gradient
  (as in REINFORCE), responsible for choosing actions.
- **Critic**: a learned value function `V_phi(s)` (or `Q_phi(s,a)`),
  responsible for estimating how good states (or state-action pairs) are,
  used to compute a low-variance advantage estimate in place of the raw
  Monte Carlo return `G_t`.

Instead of waiting for a full-episode return `G_t` (as REINFORCE does),
actor-critic uses the critic to bootstrap the advantage estimate from
partial rollouts, which further reduces variance at the cost of some bias
(from an imperfect critic). A common one-step (TD) estimator:

```
delta_t = r_t + gamma * V_phi(s_{t+1}) - V_phi(s_t)      (TD error, an estimate of A_pi(s_t,a_t))
```

**Actor update** (policy gradient using the critic's advantage estimate):

```
theta <- theta + alpha * grad_theta log pi_theta(a_t|s_t) * delta_t
```

**Critic update** (regress `V_phi` toward the observed TD target, via
squared-error loss):

```
L(phi) = ( r_t + gamma * V_phi(s_{t+1}) - V_phi(s_t) )^2
phi <- phi - beta * grad_phi L(phi)
```

A widely used middle ground between the high-variance, unbiased Monte
Carlo return and the low-variance, more-biased one-step TD estimate is
**Generalized Advantage Estimation (GAE)**, which exponentially averages
multi-step TD errors with a parameter `lambda`:

```
A_t^GAE(gamma, lambda) = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
```

`lambda = 0` recovers the one-step TD advantage (low variance, more bias);
`lambda = 1` recovers the full Monte Carlo advantage (unbiased, high
variance). GAE is the advantage estimator used in most modern policy
gradient methods, including PPO and RLHF pipelines.

Why this matters for LLM training: without a critic, updating an LLM
policy purely from full-sequence Monte Carlo rewards (REINFORCE-style) is
extremely noisy, because a single scalar reward for an entire generated
sequence must retroactively explain the "goodness" of every individual
token decision. A learned critic (a value head, often initialized from the
same base model) gives a much steadier, lower-variance training signal per
token.

---

## 5. PPO: Proximal Policy Optimization

Vanilla policy gradient / actor-critic methods have a practical failure
mode: taking a normal-sized gradient step can occasionally produce an
enormous change in the policy's behavior (since we're optimizing a ratio
of probabilities, not a bounded quantity), which can catastrophically
degrade performance and, because the next batch of data is collected using
the *new*, now-bad policy, the damage can compound and never recover. **PPO**
(Schulman et al., 2017) fixes this by explicitly constraining how far each
update is allowed to move the policy, using a simple, first-order-only
(no second derivatives, unlike TRPO) clipped objective.

### 5.1 The Probability Ratio

Define the ratio between the new (being optimized) policy and the old
policy (the one used to collect the current batch of rollouts):

```
r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
```

`r_t(theta) = 1` when the new policy matches the old policy exactly (no
change yet); `r_t(theta) > 1` means the new policy has become more likely
to take action `a_t` than the old policy was; `r_t(theta) < 1` means less
likely.

The **surrogate objective** without any constraint would simply be:

```
L^CPI(theta) = E_t [ r_t(theta) * A_t ]
```

(CPI = "conservative policy iteration"). This is a valid first-order
approximation of the true policy-gradient objective at `theta =
theta_old`, but maximizing it with no constraint can push `r_t(theta)`
arbitrarily far from 1 in a single update — exactly the destructive,
uncontrolled step PPO is designed to prevent.

### 5.2 The Clipped Surrogate Objective

PPO's core contribution is to clip the ratio so that the incentive to move
`r_t(theta)` far from 1 is removed once it exceeds a small trust region
`[1-epsilon, 1+epsilon]`:

```
L^CLIP(theta) = E_t [ min( r_t(theta) * A_t,  clip( r_t(theta), 1-epsilon, 1+epsilon ) * A_t ) ]
```

- `epsilon` is a small hyperparameter (commonly `0.1` to `0.2`).
- `clip(x, lo, hi)` clamps `x` into `[lo, hi]`.
- The `min(...)` is taken between the unclipped and clipped versions of the
  objective.

**Why the `min` and clip prevent destructively large updates** — consider
the two cases based on the sign of the advantage `A_t`:

```
Case A_t > 0 (action was better than average — we want to increase r_t(theta)):
  Unclipped term:  r_t(theta) * A_t          -> grows without bound as r_t increases
  Clipped term:    (1+epsilon) * A_t          -> flat once r_t(theta) > 1+epsilon
  min(...) = whichever is SMALLER -> once r_t(theta) > 1+epsilon, the clipped
             (flat) term is selected, so there is NO further gradient
             incentive to keep increasing r_t(theta) past 1+epsilon.

Case A_t < 0 (action was worse than average — we want to decrease r_t(theta)):
  Unclipped term:  r_t(theta) * A_t          -> grows more negative as r_t decreases
  Clipped term:    (1-epsilon) * A_t          -> flat once r_t(theta) < 1-epsilon
  min(...) = whichever is SMALLER (more negative) -> the unclipped term is
             smaller here as r_t -> 0, so clipping on this side instead acts
             as a safety net capping the objective's value; combined with
             the min, the gradient still vanishes once r_t(theta) < 1-epsilon.
```

In both cases, once the ratio moves outside the trust region `[1-epsilon,
1+epsilon]` *in the direction the unclipped objective would push it*, the
`min` selects the flat, clipped branch, which has **zero gradient** with
respect to `theta`. This removes the optimizer's incentive to keep pushing
the policy further away from `pi_theta_old` in that direction, effectively
creating a soft trust region using only first-order gradients — no
expensive KL-constrained conjugate-gradient step (as in TRPO), just clip
and take a normal SGD/Adam step. This is also why PPO permits multiple
epochs of minibatch SGD over the same batch of rollouts (data reuse) while
remaining relatively stable, whereas plain policy gradient typically
requires fresh on-policy data for every single update.

### 5.3 Why PPO Became the Default for RLHF

PPO became the workhorse of RLHF for LLMs for several concrete reasons:

1. **Stability at scale**: LLM policies have billions of parameters; a
   single destructively large update (as can occur in unclipped policy
   gradient or REINFORCE) can irrecoverably damage a model that took
   months to pretrain. PPO's clipping bounds the worst-case per-update
   policy shift.
2. **Simplicity relative to TRPO**: PPO achieves similar stability
   guarantees to Trust Region Policy Optimization (TRPO) without TRPO's
   expensive second-order (Fisher information / conjugate gradient)
   machinery — it only needs first-order gradients, which is essential
   when the policy is an LLM with billions of parameters.
3. **Sample reuse**: PPO's clipped objective tolerates multiple gradient
   epochs over the same rollout batch, which matters enormously when each
   "sample" is an expensive full LLM generation (and, in RLHF, an
   additional reward-model and value-model forward pass).
4. **Compatibility with the actor-critic + KL-penalty combination** needed
   for RLHF (Section 6) — PPO's objective is naturally augmented with an
   extra KL penalty term against a reference policy, giving fine-grained
   control over how far the fine-tuned model is allowed to drift from its
   starting point.

---

## 6. The RLHF Pipeline for LLMs, End to End

**RLHF (Reinforcement Learning from Human Feedback)** is the process of
using PPO (or another RL algorithm) to fine-tune an LLM's policy so that
its outputs are preferred by humans, using a *learned* reward signal
derived from human preference data (since there is no ground-truth reward
function for "is this response good"). The pipeline has three models and
two training stages.

### 6.1 Models Involved

```
1. pi_ref   - the "reference" policy: typically the supervised
              fine-tuned (SFT) model, FROZEN during RLHF.
2. pi_theta - the "policy" being trained via PPO; initialized from pi_ref.
3. R_phi    - the reward model, trained separately (Stage A below),
              FROZEN during PPO (Stage B).
```

### 6.2 Stage A: Training the Reward Model on Human Preference Pairs

Human annotators are shown a prompt `x` and two (or more) candidate
completions `y_w` ("winner"/chosen) and `y_l` ("loser"/rejected), and
indicate which they prefer: `y_w >_x y_l`. Directly asking humans for a
scalar reward score is noisy and hard to calibrate across annotators;
asking for pairwise *preferences* is far more reliable and consistent,
which is why RLHF collects comparisons rather than absolute scores.

The reward model `R_phi(x, y)` is a neural network (usually initialized
from the SFT model, with a scalar output head replacing the LM head) that
scores a `(prompt, completion)` pair. It is trained to be consistent with
the observed preferences using the **Bradley-Terry model** of pairwise
comparisons, which assumes:

```
P( y_w > y_l | x ) = sigma( R_phi(x, y_w) - R_phi(x, y_l) )

where sigma(z) = 1 / (1 + exp(-z))   (the logistic sigmoid)
```

That is, the probability that `y_w` is preferred over `y_l` is a sigmoid
of the *difference* in the reward model's scalar scores for the two
completions — a higher relative score means a higher probability of being
preferred. The reward model is trained by maximizing the log-likelihood of
the observed human preferences, i.e., minimizing the negative
log-likelihood (a binary cross-entropy / logistic loss) over the
preference dataset `D = {(x, y_w, y_l)}`:

```
L(phi) = - E_{(x,y_w,y_l) ~ D} [ log( sigma( R_phi(x,y_w) - R_phi(x,y_l) ) ) ]
```

Minimizing `L(phi)` pushes `R_phi(x, y_w)` up and `R_phi(x, y_l)` down
whenever the model's current scores disagree with (or are insufficiently
confident about) the human-labeled preference, without ever requiring an
absolute reward scale — only relative order matters, which is exactly what
Bradley-Terry captures.

### 6.3 Stage B: PPO Fine-Tuning Against the Reward Model, with a KL Penalty

Once `R_phi` is trained and frozen, the LLM policy `pi_theta` (initialized
from `pi_ref`) is fine-tuned via PPO to maximize the reward model's score,
**with an added penalty for diverging too far from the reference policy
`pi_ref`**. The per-token (or per-sequence) reward used in the PPO loop is:

```
r_total(x, y) = R_phi(x, y) - beta * KL( pi_theta(.|x) || pi_ref(.|x) )
```

Per-token form, commonly implemented as a token-level penalty added to the
terminal reward-model score:

```
r_t = -beta * log( pi_theta(a_t|s_t) / pi_ref(a_t|s_t) )     for t < T (intermediate tokens)
r_T = R_phi(x, y) - beta * log( pi_theta(a_T|s_T) / pi_ref(a_T|s_T) )   (final token, adds reward model score)
```

- `beta` is a coefficient (fixed or adaptively controlled) governing the
  strength of the KL penalty.
- `KL(pi_theta || pi_ref) = E_{a~pi_theta}[ log(pi_theta(a|s)/pi_ref(a|s)) ]`
  measures how much the current policy's token distribution has diverged
  from the frozen reference/SFT model's distribution.

This combined reward `r_total` (or the per-token decomposition above) is
what the actor-critic + PPO machinery from Sections 4-5 actually optimizes:
a critic `V_phi_v(s)` is trained to predict expected future `r_total`, a
GAE advantage is computed, and the PPO clipped surrogate objective (Section
5.2) is applied to update `pi_theta`.

**Why the KL penalty exists — preventing reward hacking and drift:**

1. **Reward hacking / over-optimization**: `R_phi` is only a learned
   *proxy* for genuine human preference, trained on a finite, imperfect
   sample of comparisons. Unconstrained optimization pressure will find
   and exploit any blind spot or systematic error in `R_phi` — e.g.,
   producing unnaturally long, repetitive, sycophantic, or oddly-formatted
   text that scores highly on `R_phi` without actually being higher
   quality to a human (Goodhart's law: "when a measure becomes a target,
   it ceases to be a good measure"). The KL penalty puts a cost on moving
   away from a policy (`pi_ref`) known to already produce fluent,
   reasonable text, which directly discourages the policy from wandering
   into these exploitable, out-of-distribution regions of output space
   where `R_phi`'s judgments are least reliable.
2. **Preserving language quality / capability retention**: `pi_ref` (the
   SFT model) already generates coherent, grammatical, broadly competent
   text. Optimizing purely for `R_phi` with no anchor can degrade fluency
   or general capability in pursuit of reward, since nothing in the
   objective otherwise values "still sounds like natural language."
   Penalizing KL divergence from `pi_ref` keeps `pi_theta` in the
   neighborhood of a model already known to behave reasonably.
3. **Stabilizing PPO itself**: since `R_phi` is trained on outputs from
   *around* `pi_ref`'s distribution, its scores become increasingly
   unreliable extrapolations the further `pi_theta` drifts from that
   distribution. The KL term keeps the PPO rollouts in-distribution for
   the reward model, which keeps the reward signal itself trustworthy
   throughout training — a self-reinforcing purpose distinct from (1) but
   closely related to it.

The reward model, reference model, and current policy together mean a full
PPO/RLHF update step requires forward passes through (at least) four
networks per batch: the policy `pi_theta` (generation + backward pass),
the frozen reference `pi_ref` (for the KL term), the frozen reward model
`R_phi` (to score completions), and a critic/value model `V` (for the
advantage estimate) — a major reason RLHF is compute- and
infrastructure-heavy relative to supervised fine-tuning.

---

## 7. DPO: Direct Preference Optimization

**DPO** (Rafailov et al., 2023) observes that the entire RLHF pipeline
(Sections 6.2-6.3) can be collapsed into optimizing a *single* closed-form
loss directly on preference pairs — no separate reward model, no PPO
rollouts, no critic, no reference-model KL bookkeeping during training
(the reference model is still needed, but only for a static log-probability
term, not an RL loop).

### 7.1 From the RLHF Objective to a Closed-Form Optimal Policy

The RLHF objective being optimized in Stage B (Section 6.3) is, in
expectation over prompts `x` and completions `y ~ pi_theta`:

```
max_theta  E_{x, y~pi_theta(.|x)} [ R_phi(x,y) ]  -  beta * KL( pi_theta(.|x) || pi_ref(.|x) )
```

This exact objective — reward maximization regularized by a KL penalty to
a reference policy — has a known **closed-form optimal solution** (a
standard result from the KL-regularized bandit / control-as-inference
literature):

```
pi*(y|x) = ( 1 / Z(x) ) * pi_ref(y|x) * exp( (1/beta) * R_phi(x,y) )

where  Z(x) = sum_y  pi_ref(y|x) * exp( (1/beta) * R_phi(x,y) )   (a normalizing partition function)
```

`Z(x)` is intractable to compute in general (it sums/integrates over every
possible completion `y`), which is precisely why RLHF resorts to PPO —
an RL algorithm that can optimize this objective *without* ever computing
`Z(x)` explicitly, by sampling rollouts instead.

DPO's key algebraic move is to **invert this relationship**: instead of
solving for the optimal policy given a reward function, solve for the
implied reward function given a policy. Rearranging the closed form above
for `R_phi(x,y)`:

```
R_phi(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)
```

This says: *any* reward function consistent with the optimal RLHF policy
`pi*` can be re-expressed purely in terms of the log-probability ratio
between that policy and the reference model, plus a term `beta * log
Z(x)` that depends only on the prompt `x` — not on the completion `y`.

### 7.2 Substituting into the Bradley-Terry Loss

Recall the Bradley-Terry preference model from Section 6.2:

```
P(y_w > y_l | x) = sigma( R_phi(x,y_w) - R_phi(x,y_l) )
```

Because the `beta * log Z(x)` term does not depend on `y`, it is identical
for `y_w` and `y_l` and **cancels exactly** when we take the difference
`R_phi(x,y_w) - R_phi(x,y_l)`:

```
R_phi(x,y_w) - R_phi(x,y_l)
  = [ beta*log(pi*(y_w|x)/pi_ref(y_w|x)) + beta*log Z(x) ]
  - [ beta*log(pi*(y_l|x)/pi_ref(y_l|x)) + beta*log Z(x) ]
  = beta * [ log(pi*(y_w|x)/pi_ref(y_w|x)) - log(pi*(y_l|x)/pi_ref(y_l|x)) ]
```

Substituting this directly into the Bradley-Terry preference probability
gives a preference model expressed **entirely in terms of the policy
itself** (no reward model at all):

```
P(y_w > y_l | x) = sigma( beta * [ log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)) ] )
```

(Here `pi*` has been replaced by `pi_theta`, the policy we are actively
optimizing — we are now fitting `pi_theta` directly to reproduce the
preference data, rather than fitting a reward model.)

### 7.3 The DPO Loss

Maximizing the likelihood of the observed human preferences under this
policy-parameterized preference model — exactly analogous to Stage A's
reward-model training, but now applied straight to the policy — gives the
**DPO loss**:

```
L_DPO(theta; pi_ref) = - E_{(x,y_w,y_l) ~ D} [
    log sigma(
        beta * log( pi_theta(y_w|x) / pi_ref(y_w|x) )
      - beta * log( pi_theta(y_l|x) / pi_ref(y_l|x) )
    )
]
```

This is a simple **binary classification loss** (logistic regression) on
preference pairs, computed using nothing but log-probabilities the policy
and reference model already assign to full sequences (summed token
log-probs). Its gradient has an intuitive form:

```
grad_theta L_DPO = - beta * E_{(x,y_w,y_l)~D} [
    sigma( r_hat(x,y_l) - r_hat(x,y_w) ) * ( grad_theta log pi_theta(y_w|x) - grad_theta log pi_theta(y_l|x) )
]

where  r_hat(x,y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )   (the "implicit reward")
```

Intuitively: increase the log-probability of the chosen response `y_w` and
decrease the log-probability of the rejected response `y_l`, weighted by
`sigma(r_hat(x,y_l) - r_hat(x,y_w))` — i.e., the update is scaled up
precisely when the model's *current* implicit reward ordering still
disagrees with (or is not yet confident about) the human preference, and
scaled down automatically once the model already ranks `y_w` above `y_l`
with high confidence.

### 7.4 What DPO Eliminates

DPO reformulates RLHF's two-stage, three-model pipeline (reward model
training + PPO rollouts against a frozen reward model and reference model)
into a **single supervised-style loss** computed directly on the offline
preference dataset:

- No explicit reward model `R_phi` is ever trained — the policy itself
  implicitly defines a reward via `r_hat(x,y) = beta * log(pi_theta/pi_ref)`.
- No online rollouts/sampling from the policy during training — DPO trains
  directly on the fixed, offline `(x, y_w, y_l)` dataset, exactly like
  supervised fine-tuning.
- No PPO clipping, no advantage estimation, no critic/value network, no
  GAE — the entire RL apparatus from Sections 4-5 is unnecessary.
- The reference model `pi_ref` is still required, but only as a frozen
  scorer providing `log pi_ref(y|x)` for each pair — not embedded in an RL
  loop.

---

## 8. Pipeline Comparison: RLHF/PPO vs. DPO

```
                         RLHF / PPO PIPELINE
                         ====================

  Preference data                     Stage A: Reward Model Training
  (x, y_w, y_l)   ------------------> +-----------------------------+
                                       |   R_phi(x, y)  (scalar)     |
                                       |   trained w/ Bradley-Terry  |
                                       |   loss on (y_w, y_l) pairs  |
                                       +-----------------------------+
                                                    |
                                                    | R_phi frozen
                                                    v
   Stage B: PPO RL Loop (repeated for many iterations)
  +-----------------------------------------------------------------+
  |                                                                   |
  |   prompt x --> [ pi_theta (ACTOR, being trained) ] --> y (rollout)|
  |                         |                                         |
  |                         v                                         |
  |               [ Critic V(s) ]  -->  GAE advantage A_t             |
  |                         |                                         |
  |        +----------------+-----------------+                      |
  |        v                                   v                      |
  |  [ R_phi(x,y) ]                   [ pi_ref(.|x) FROZEN ]          |
  |   reward score                     KL(pi_theta || pi_ref)         |
  |        |                                   |                      |
  |        +------------------+----------------+                     |
  |                            v                                      |
  |                r_total = R_phi(x,y) - beta*KL                     |
  |                            |                                      |
  |                            v                                      |
  |              PPO CLIPPED SURROGATE OBJECTIVE                      |
  |              L^CLIP(theta) = E[min(r_t*A_t, clip(r_t)*A_t)]       |
  |                            |                                      |
  |                            v                                      |
  |               update pi_theta (and critic V)                     |
  |                            |                                      |
  +----------------------------+--------------------------------------+
                                | (loop: new pi_theta generates next rollouts)
                                v
                       Final RLHF-tuned policy

  Models required simultaneously during training: pi_theta, pi_ref,
  R_phi, V(critic)  -->  4 networks


                            DPO PIPELINE
                            ============

  Preference data
  (x, y_w, y_l)
        |
        v
  +---------------------------------------------------------------+
  |  Compute, for both y_w and y_l:                                 |
  |     log pi_theta(y|x)   (policy being trained, forward pass)    |
  |     log pi_ref(y|x)     (frozen reference, forward pass)        |
  |                                                                   |
  |  r_hat(x,y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) )     |
  |                                                                   |
  |  L_DPO = -log sigma( r_hat(x,y_w) - r_hat(x,y_l) )               |
  +---------------------------------------------------------------+
        |
        v
   single backward pass, update pi_theta directly
   (exactly like supervised fine-tuning; no rollouts, no sampling
    from pi_theta during training, no PPO loop, no critic)

  Models required simultaneously during training: pi_theta, pi_ref
  --> 2 networks, no online generation
```

---

## 9. Comparison Table: RLHF/PPO vs. DPO

| Dimension | RLHF (Reward Model + PPO) | DPO |
|---|---|---|
| **Models required during training** | 4: policy (actor), critic/value model, frozen reward model, frozen reference model | 2: policy, frozen reference model |
| **Training data flow** | Preference pairs -> reward model (offline) -> online RL rollouts scored by reward model | Preference pairs used directly in a single offline classification loss |
| **Needs online sampling/generation during training?** | Yes — PPO requires generating rollouts from the current policy at every iteration | No — trains like supervised fine-tuning on a fixed dataset |
| **Implementation complexity** | High: reward-model training pipeline, PPO actor-critic loop, reward normalization, KL coefficient tuning, GAE, clipping hyperparameters | Low: one loss function, standard supervised training loop |
| **Training stability** | Can be unstable/sensitive to hyperparameters (KL coefficient `beta`, reward scaling, PPO clip range, value-function tuning); prone to reward hacking if under-regularized | Generally more stable — no moving RL target, no reward-model exploitation because there is no separate reward model to exploit |
| **Compute cost** | High — generation (rollouts) + reward model scoring + critic + policy forward/backward, repeated over many RL iterations | Lower — comparable to standard SFT; two forward/backward passes per pair (policy + reference), no generation loop |
| **Sample efficiency (data reuse)** | On-policy-ish (PPO tolerates a few epochs per batch); needs continual fresh rollouts as the policy shifts | Fully offline; can train multiple epochs over static preference data like any supervised dataset |
| **Flexibility / extensibility** | Reward model is reusable/inspectable independently (can score arbitrary text, be combined with rule-based rewards, used for other downstream tasks like best-of-N reranking) | Reward is only implicit in the policy; harder to separately audit, reuse, or combine with other reward signals |
| **Theoretical guarantee** | Approximate — PPO optimizes a clipped surrogate as a practical proxy for the true KL-regularized RL objective | Exact — under the derivation's assumptions, the DPO loss's global optimum corresponds exactly to the same optimal policy `pi*` that RLHF targets |
| **When preferred** | Large-scale frontier LLM training where extensive infra already exists, where reward-model reuse (e.g., best-of-N, other reward shaping) is valuable, or where reward signals beyond static preference pairs (rule-based, verifiable rewards, tool-use success) are needed | Most practical fine-tuning scenarios with a fixed preference dataset and no need for online/verifiable reward signals; favored for its simplicity, stability, and lower compute/engineering cost |

---

## 10. Summary of the Family Tree

```
                     RL objective: max_theta J(theta) = E[cumulative reward]
                                          |
                +-------------------------+-------------------------+
                |                                                     |
     Multi-armed bandits                                    Full sequential MDPs
     (no state transitions)                                  (state matters)
     epsilon-greedy, UCB,                                            |
     Thompson Sampling (-> see 12-thompson-sampling)                 |
                                                                      v
                                                      Policy Gradient (REINFORCE)
                                                      grad J = E[grad log pi * R]
                                                      (log-derivative trick)
                                                      high variance -> baseline/advantage
                                                                      |
                                                                      v
                                                             Actor-Critic
                                                      (learned critic V(s) lowers variance)
                                                                      |
                                                                      v
                                                        PPO (clipped surrogate objective,
                                                         first-order trust region)
                                                                      |
                                                                      v
                                        RLHF = PPO + Reward Model (Bradley-Terry) + KL penalty vs. pi_ref
                                                                      |
                                                                      v
                                        DPO = closed-form reparameterization of the same
                                          KL-regularized objective, collapsed into one
                                          classification loss on preference pairs
```

Every algorithm in this tree is, at bottom, a different answer to the same
question posed in Section 1.5: how to estimate and climb `grad_theta
J(theta)` using only sampled experience and scalar (or pairwise) feedback,
with each successive method trading off variance, bias, compute cost, and
implementation complexity differently.
