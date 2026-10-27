# Neural Network Optimizers — From Gradient Descent to AdamW

An optimizer is the algorithm that turns a gradient into a parameter update. Everything a large
language model "learns" during training is the accumulated effect of millions of these tiny,
mechanical update steps. This document builds the math up from first principles — plain gradient
descent, momentum, the adaptive-learning-rate family (AdaGrad, RMSProp, Adam), and finally AdamW,
the de facto standard for training transformers — and then covers the practical machinery
(learning rate schedules, warmup, gradient clipping) that makes those algorithms actually work at
scale. It closes by connecting all of this to
[LoRA/QLoRA fine-tuning](../08-lora-qlora/README.md), where the optimizer's memory footprint
becomes the binding constraint rather than an afterthought.

## Table of contents

1. [The optimization problem](#1-the-optimization-problem)
2. [Vanilla gradient descent and why it's insufficient](#2-vanilla-gradient-descent-and-why-its-insufficient)
3. [Momentum](#3-momentum)
4. [Adaptive learning rates: AdaGrad → RMSProp → Adam](#4-adaptive-learning-rates-adagrad--rmsprop--adam)
5. [AdamW: decoupled weight decay](#5-adamw-decoupled-weight-decay)
6. [Key hyperparameters and learning rate schedules](#6-key-hyperparameters-and-learning-rate-schedules)
7. [Gradient clipping](#7-gradient-clipping)
8. [The training loop and optimizer state](#8-the-training-loop-and-optimizer-state)
9. [Optimizers and LoRA/PEFT](#9-optimizers-and-loraqeft)
10. [Summary table](#10-summary-table)
11. [References](#11-references)

---

## 1. The optimization problem

Training a neural network is, mechanically, nothing more than solving an unconstrained
minimization problem:

```
theta* = argmin_theta  L(theta)
```

where `theta` is the full vector of model parameters (for an LLM, every weight and bias in every
attention and MLP block, flattened into one enormous vector — often tens to hundreds of billions
of scalars), and `L(theta)` is a scalar loss function measuring how badly the model with parameters
`theta` performs on the training data. For language modeling this is almost always the average
cross-entropy (negative log-likelihood) of the next token given its context:

```
L(theta) = -(1/N) * sum_{i=1}^{N} log P_theta(x_i | x_<i)
```

`L` is a function of billions of variables and is highly non-convex — it has saddle points, flat
regions ("plateaus"), sharp ravines, and countless local minima. There is no closed-form solution
and no way to evaluate `L` or its curvature analytically at the scale of an LLM. The only thing we
can compute cheaply is the **gradient** of `L` with respect to `theta` — the vector of partial
derivatives:

```
grad(L)(theta) = [ dL/dtheta_1, dL/dtheta_2, ..., dL/dtheta_n ]
```

This gradient points in the direction of steepest *ascent* of `L` at the current point. Moving a
small step in the opposite direction is guaranteed (for small enough steps, in a locally convex
neighborhood) to decrease the loss. This is the entire premise of gradient-based optimization: we
cannot solve for the minimum directly, so we walk downhill, one gradient at a time.

### Computing the gradient: backpropagation

The gradient is computed via **backpropagation**, which is just the chain rule applied
systematically to a computation graph. A neural network is a composition of functions:

```
L = loss( f_K( f_{K-1}( ... f_1(x; theta_1) ...; theta_{K-1}); theta_K) )
```

Backprop computes `dL/dtheta_k` for every layer `k` by propagating an error signal backward through
the graph, layer by layer, reusing the intermediate values (activations) that were cached during
the forward pass. For a layer computing `y = f(x; theta)`, given the upstream gradient `dL/dy`
(supplied by the layer after it), backprop computes:

```
dL/dtheta = (dL/dy) * (dy/dtheta)      # gradient w.r.t. this layer's parameters
dL/dx     = (dL/dy) * (dy/dx)          # gradient to pass further backward
```

Each layer only needs to know its own local Jacobian (`dy/dtheta` and `dy/dx`); the chain rule
handles composing them across the whole network. The cost of one backward pass is roughly 2x the
cost of one forward pass (in FLOPs), which is why training a model is roughly 3x the cost of pure
inference per token (1 forward + 2 backward-equivalent passes).

**Backpropagation produces the gradient. The optimizer decides what to do with it.** These are two
separate, decoupled concerns — you can swap in any optimizer without touching how gradients are
computed. This document is entirely about the second half: given `grad(L)(theta)` at each step,
how do we update `theta`?

---

## 2. Vanilla gradient descent and why it's insufficient

The simplest possible update rule is:

```
theta_{t+1} = theta_t - lr * grad(L)(theta_t)
```

where `lr` (the learning rate, also written `eta` or `alpha`) is a small positive scalar
controlling step size. This is **gradient descent**: at every step, move a small amount in the
direction opposite the gradient.

### Batch, stochastic, and mini-batch variants

The gradient `grad(L)(theta)` as defined above is an average over the *entire* training set. There
are three ways to approximate it in practice, trading off gradient accuracy against compute cost
per step:

**Batch gradient descent** computes the exact gradient using every training example before taking
a single step:

```
grad(L)(theta) = (1/N) * sum_{i=1}^{N} grad(L_i)(theta)
```

This is accurate but computationally infeasible for LLM-scale datasets (billions to trillions of
tokens) — you'd wait for a full pass over the corpus just to take one step.

**Stochastic gradient descent (SGD)** goes to the opposite extreme: it computes the gradient from a
single randomly sampled example and updates immediately:

```
theta_{t+1} = theta_t - lr * grad(L_i)(theta_t),   i sampled at random
```

This gives a very noisy, high-variance estimate of the true gradient, but it's cheap per step and
the noise itself can help escape shallow local minima. In practice it's too noisy and too
sequential (no batching = no GPU parallelism) to be used directly for deep learning.

**Mini-batch gradient descent** is the practical compromise, and what "SGD" almost universally
means in a modern deep learning context: compute the average gradient over a small batch `B` of
examples (typically tens to millions of tokens, depending on scale):

```
grad(L)(theta) = (1/|B|) * sum_{i in B} grad(L_i)(theta)
```

This is the version actually used to train every modern neural network, including LLMs (with
"batch" here usually meaning a fixed number of tokens packed into sequences, not documents).
Mini-batching gives a much lower-variance gradient estimate than single-example SGD (variance
scales as `1/|B|`) while remaining far cheaper than a full-dataset pass, and critically, it maps
directly onto GPU/TPU parallel compute — a batch of examples is processed as one parallel matrix
operation.

### Why vanilla (mini-batch) gradient descent is insufficient at scale

Plain gradient descent, even with mini-batches, has several structural problems that make it
impractical for training deep networks with hundreds of millions to hundreds of billions of
parameters:

1. **A single global learning rate is wrong for every parameter simultaneously.** Different
   weights have wildly different gradient magnitudes and curvatures (the loss surface is far more
   curved in some directions than others). A learning rate small enough to be stable for
   steep/sensitive directions will make no progress in shallow/flat directions, and a learning rate
   large enough to make progress in flat directions will diverge in steep ones.

2. **Ravines cause oscillation.** When the loss surface looks like a narrow, curved valley (steep
   walls, shallow floor — extremely common in deep networks), gradient descent zig-zags back and
   forth across the walls of the ravine instead of moving efficiently along the floor toward the
   minimum. Progress along the important (floor) direction is very slow.

3. **No memory across steps.** Each update depends only on the *current* gradient. There is no
   mechanism to build on consistent directional signal from previous steps or to damp
   noisy/inconsistent signal — every step reacts only to instantaneous, mini-batch-noisy gradient
   information.

4. **Saddle points and plateaus stall training.** In high-dimensional non-convex loss surfaces
   (the norm for deep networks), saddle points — where the gradient is near zero but it's not a
   minimum — vastly outnumber true local minima. Plain gradient descent has essentially no signal
   to escape a region where the gradient magnitude collapses.

These problems motivate two independent classes of improvement, which we build up next:
**momentum** (using the *history* of gradients to smooth the trajectory) and **adaptive learning
rates** (using the *history* of gradient magnitudes to rescale the step size independently for
each parameter). Adam and AdamW combine both.

---

## 3. Momentum

### Physical intuition

Momentum reframes optimization as physics: imagine a ball rolling down the loss surface. Instead
of a particle that only ever moves in the direction of the instantaneous local slope (as in vanilla
gradient descent), a ball has mass and therefore **velocity** — it accumulates speed as it rolls
downhill in a consistent direction, and it doesn't instantly change direction just because the
local slope wiggles slightly. This gives it two useful properties in a loss landscape:

- Along a **ravine**, the gradient's oscillating component (across the narrow walls) partially
  cancels out over successive steps because it keeps changing sign, while the consistent component
  (along the shallow floor, toward the minimum) keeps compounding. The net effect is much less
  zig-zag and much faster progress along the floor.
- Through a **shallow region or small local bump**, accumulated velocity from the runway before it
  can carry the ball through, rather than stalling out the moment the local gradient shrinks.

### The update equations

Momentum introduces a **velocity** term `v` (also called the momentum buffer) that is an
exponentially decaying accumulation of past gradients:

```
v_t     = beta * v_{t-1} + grad(L)(theta_{t-1})
theta_t = theta_{t-1} - lr * v_t
```

- `v_t` is the velocity at step `t`, initialized to `v_0 = 0`.
- `beta` (typically ~0.9) is the **momentum coefficient**: it controls how much of the previous
  velocity survives into the current step. `beta = 0` recovers vanilla gradient descent exactly
  (no memory). `beta` close to 1 means velocity decays very slowly, so the optimizer keeps "rolling"
  in a direction long after the gradient that originally caused it has changed.

Expanding the recursion shows explicitly that `v_t` is an exponentially weighted sum of *all*
past gradients, with the most recent gradients weighted most heavily:

```
v_t = grad_t + beta * grad_{t-1} + beta^2 * grad_{t-2} + beta^3 * grad_{t-3} + ...
```

A gradient component that points in a *consistent* direction across steps gets reinforced
additively (the geometric series sums to roughly `1/(1-beta)` times the steady-state gradient
magnitude — e.g. an effective ~10x amplification at `beta=0.9`). A component that *flips sign* every
step (the oscillation across a ravine wall) gets cancelled out because consecutive terms partially
subtract from the running sum. This is precisely why momentum damps oscillation while accelerating
consistent descent.

Some formulations (e.g. PyTorch's SGD with `nesterov=False`) write this with a `(1-beta)` mixing
term instead of a pure sum:

```
v_t = beta * v_{t-1} + (1 - beta) * grad(L)(theta_{t-1})
```

This is the exponential-moving-average form (used consistently for Adam's moment estimates in
Section 4) — it keeps `v_t` on the same scale as the gradient itself rather than growing with
`1/(1-beta)`, which matters once you also want to bias-correct it.

**Nesterov momentum** is a refinement that evaluates the gradient not at the current position but
at the position velocity would carry you to (a "look-ahead" gradient), giving a small but
consistent improvement in convergence rate; it is not essential to understand for the Adam family
covered next, but appears as an option in most SGD implementations.

Momentum alone solves problem #2 and #3 from Section 2 (oscillation, no memory) but does nothing
about problem #1 (a single global learning rate for parameters with wildly different gradient
scales). That's what the adaptive methods solve.

---

## 4. Adaptive learning rates: AdaGrad → RMSProp → Adam

### 4.1 AdaGrad — per-parameter learning rates from accumulated squared gradients

AdaGrad's idea: give each parameter its *own* effective learning rate, scaled down for parameters
that have historically had large gradients (they've already moved a lot, or are very sensitive)
and left larger for parameters that have historically had small gradients (they've moved little,
or are on a flat part of the surface). It does this by accumulating the **sum of squared
gradients**, per parameter, since the start of training:

```
G_t = G_{t-1} + grad(L)(theta_{t-1})^2          # element-wise square, accumulated
theta_t = theta_{t-1} - (lr / (sqrt(G_t) + epsilon)) * grad(L)(theta_{t-1})
```

- `G_t` is a running per-parameter sum of squared gradients (a vector the same shape as `theta`),
  initialized to `G_0 = 0`.
- `epsilon` (e.g. `1e-8`) is a small constant added purely for numerical stability, to avoid
  dividing by zero when `G_t` is still near zero early in training.
- All operations (`^2`, `sqrt`, `/`) are element-wise — every parameter gets its own independent
  scale factor `lr / (sqrt(G_t) + epsilon)`.

This works well for sparse features (AdaGrad was originally designed for convex problems with
sparse gradients, like NLP bag-of-words models) but has a fatal flaw for deep learning: **`G_t` is
a monotonically increasing sum that never resets or decays.** Since it only ever grows, the
effective learning rate `lr / sqrt(G_t)` only ever shrinks, and it shrinks roughly like `1/sqrt(t)`
over the course of training. For long training runs (and LLM pretraining runs can be hundreds of
thousands to millions of steps), this means AdaGrad's effective step size eventually decays toward
zero and the optimizer effectively stops learning, regardless of whether the loss has actually
converged.

### 4.2 RMSProp — fixing AdaGrad's decay with an exponential moving average

RMSProp's fix is simple but critical: replace the ever-growing *sum* of squared gradients with an
**exponentially decaying moving average** of squared gradients, so old gradient history is
gradually forgotten instead of accumulated forever:

```
v_t = beta2 * v_{t-1} + (1 - beta2) * grad(L)(theta_{t-1})^2
theta_t = theta_{t-1} - (lr / (sqrt(v_t) + epsilon)) * grad(L)(theta_{t-1})
```

- `v_t` is now a moving average, not a running sum — it has a roughly constant "memory window" of
  about `1 / (1 - beta2)` recent steps, rather than remembering every gradient ever seen with equal
  weight.
- `beta2` (typically ~0.99 in plain RMSProp) controls the length of that memory window: larger
  `beta2` means slower decay, so more history is retained.

Because `v_t` no longer monotonically grows, the effective per-parameter learning rate
`lr / sqrt(v_t)` can go back up again if recent gradients shrink, instead of being permanently
capped by ancient, possibly irrelevant gradient history. This one change — sum to exponential
moving average — is what makes adaptive methods viable for long, non-convex deep learning training
runs, and it's directly reused, unchanged in spirit, as the "second moment" in Adam below.

### 4.3 Adam — momentum + RMSProp-style adaptive scaling, with bias correction

Adam ("Adaptive Moment Estimation") combines the two ideas developed so far:

- Momentum's exponential moving average of the gradient itself (the **first moment**, an estimate
  of `E[grad]`) — this is exactly the `v_t` from Section 3, just renamed `m_t` and written in EMA
  form.
- RMSProp's exponential moving average of the squared gradient (the **second moment**, an estimate
  of `E[grad^2]`) — this is `v_t` from Section 4.2, renamed here to avoid clashing with the first
  moment's name.

Written out completely, step by step, for a single optimizer step `t` (with parameter gradient
`g_t = grad(L)(theta_{t-1})`):

**Step 1 — update the first moment estimate (momentum term):**

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
```

**Step 2 — update the second moment estimate (uncentered variance term):**

```
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

(`m_0 = 0`, `v_0 = 0`; both are vectors the same shape as `theta`; `g_t^2` is element-wise.)

**Step 3 — bias-correct both moments.**

Because `m_t` and `v_t` are initialized at zero and are exponential moving averages, they are
strongly biased toward zero during the early steps of training, especially when `beta1` and
`beta2` are close to 1 (slow-decaying averages take a long time to "warm up" away from their
zero initialization). Adam corrects for this analytically:

```
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
```

where `t` is the current step number (starting at 1). Early in training, `beta1^t` and `beta2^t`
are still close to 1, so the denominators `(1 - beta1^t)` and `(1 - beta2^t)` are small, which
inflates `m_hat_t` and `v_hat_t` back up to counteract the zero-initialization bias. As `t` grows,
`beta1^t` and `beta2^t` decay toward 0, the correction factors approach 1, and `m_hat_t → m_t`,
`v_hat_t → v_t` — the correction naturally fades out once the moving averages have had enough
steps to reflect real gradient statistics.

**Step 4 — the parameter update:**

```
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

Putting all four steps together, Adam's complete update per step is:

```
g_t     = grad(L)(theta_{t-1})
m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

Standard defaults (from the original Adam paper, and still near-universal today):
`beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`.

**Why this combination works so well:**

- `m_hat_t` (momentum) smooths out mini-batch gradient noise and damps oscillation across ravines,
  exactly as in Section 3.
- `sqrt(v_hat_t)` (adaptive scaling) gives every parameter its own effective step size, shrinking
  the step for parameters with large/volatile gradient history and growing it (relatively) for
  parameters with small/quiet gradient history — exactly as in Sections 4.1–4.2, but with the
  non-decaying-memory property that makes it stable over long training runs.
- Bias correction means Adam behaves reasonably from step 1, instead of taking an initial
  warm-up period during which `m_t` and `v_t` are artificially near zero and the effective updates
  are too large relative to the true gradient statistics.

This combination — first-moment momentum divided by the square root of a second-moment
adaptive-variance estimate, both bias-corrected — is what makes Adam converge fast and reliably
across an enormous range of architectures with minimal hyperparameter tuning, and it's the reason
Adam (and its descendant AdamW, next) became the default optimizer for essentially all transformer
training.

---

## 5. AdamW: decoupled weight decay

### The problem: L2 regularization and Adam interact badly

**Weight decay** is a regularization technique that discourages large weight magnitudes by adding
a penalty term to the loss, proportional to the squared norm of the parameters:

```
L_regularized(theta) = L(theta) + (lambda / 2) * ||theta||^2
```

When you differentiate this regularized loss to get the gradient used in an update, the
regularization term contributes an extra `lambda * theta` to the gradient:

```
grad(L_regularized)(theta) = grad(L)(theta) + lambda * theta
```

In classic SGD, folding weight decay into the gradient like this (this is standard **L2
regularization**) is mathematically equivalent to true weight decay (directly shrinking `theta` by
a fixed fraction each step), because SGD's update is just `theta - lr * grad`, so the extra
`lambda * theta` term ends up subtracted from `theta` scaled only by `lr` — a clean, direct
shrinkage.

**In Adam, this equivalence breaks.** The problem is that Adam's original formulation (and most
early implementations) added the L2 penalty directly to the gradient `g_t` *before* it entered the
momentum and variance accumulators:

```
g_t = grad(L)(theta_{t-1}) + lambda * theta_{t-1}     # L2 term folded into the gradient
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

Because the weight-decay contribution is now baked into `g_t`, it gets divided by `sqrt(v_hat_t)`
along with the "real" loss gradient. Parameters with large historical gradient magnitude (large
`v_hat_t`) end up having their weight-decay penalty *suppressed* (divided down), while parameters
with small historical gradient magnitude get their weight decay *amplified* relatively. The amount
of actual weight shrinkage a parameter experiences ends up entangled with its gradient statistics
in a way that has nothing to do with the regularization's intent — the whole point of weight decay
is to uniformly discourage large weights, not to do so unevenly based on unrelated gradient-scale
history. This coupling measurably hurts generalization and made proper weight-decay tuning
unreliable under Adam.

### AdamW's fix: decouple weight decay from the gradient

AdamW (Loshchilov & Hutter, 2017, "Decoupled Weight Decay Regularization") fixes this by removing
the weight-decay term from the gradient entirely — so it never touches `m_t` or `v_t` — and instead
applying it as a direct, separate shrinkage of the weights at the final update step:

```
g_t     = grad(L)(theta_{t-1})                          # NOTE: no weight decay folded in here
m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * ( m_hat_t / (sqrt(v_hat_t) + epsilon) + weight_decay * theta_{t-1} )
```

Equivalently, this is often implemented as two separate operations per step:

```
theta_t = theta_{t-1} - lr * weight_decay * theta_{t-1}          # (a) decoupled decay
theta_t = theta_t      - lr * m_hat_t / (sqrt(v_hat_t) + epsilon) # (b) the Adam step, on the decayed weights
```

The critical difference from Adam+L2 is just the position of the `weight_decay * theta` term:
**it is applied directly to the parameter update, scaled only by `lr`, completely independent of
`m_hat_t` and `v_hat_t`.** Weight decay now shrinks every parameter by the same proportional
fraction (`lr * weight_decay`) every step, regardless of that parameter's gradient history — which
is what "weight decay" was always supposed to mean, and restores the SGD-equivalent behavior that
Adam's naive L2-in-the-gradient approach had broken.

### Why this matters for training large transformers

AdamW is the de facto standard optimizer for essentially every modern LLM pretraining and
fine-tuning run (GPT, LLaMA, and most open transformer families are all trained with AdamW) for a
few concrete reasons:

- **Better, more predictable generalization.** Decoupling makes the regularization strength behave
  as intended and independently tunable from the learning rate and gradient statistics, which
  produces measurably better validation loss and downstream task performance than Adam+L2 at
  comparable settings — this was the original empirical finding that motivated the paper.
- **Weight decay counteracts weight growth from adaptive scaling.** Adam's adaptive per-parameter
  scaling (`1/sqrt(v_hat_t)`) can let weights in low-gradient-variance directions grow larger over
  a long training run than they would under plain SGD; decoupled weight decay provides a clean,
  uniform counter-pressure against that growth without distorting the adaptive scaling itself.
- **Decoupling makes weight decay and learning rate independently tunable**, which matters a lot in
  practice because LLM training uses learning rate schedules (Section 6) where `lr` changes by
  orders of magnitude over the course of training. Under Adam+L2, the effective regularization
  strength would swing with the schedule in an uncontrolled way; under AdamW it doesn't, because
  weight decay's *proportional* shrinkage rate is a separate, stable hyperparameter (though the
  absolute per-step decay `lr * weight_decay` still scales with `lr`, its interaction with the
  gradient-adaptive term is what's decoupled — that entanglement is what caused the original
  problem).
- **It's a near-drop-in replacement.** The change from Adam to AdamW is a small, well-isolated
  modification (move one term from inside the gradient to the final update), so it carries none of
  Adam's convergence-speed and robustness advantages while fixing this one specific defect — there
  is essentially no reason to prefer plain Adam-with-L2 for transformer training today.

---

## 6. Key hyperparameters and learning rate schedules

### The hyperparameters, and what each one actually controls

| Hyperparameter | Typical LLM value | What it controls |
|---|---|---|
| `lr` (learning rate) | peak ~1e-4 to 6e-4 (pretraining), 1e-5 to 2e-4 (fine-tuning) | Overall step size. Too high → loss diverges or spikes; too low → training is needlessly slow or gets stuck in poor regions. |
| `beta1` | 0.9 | Decay rate of the first-moment (momentum) estimate. Higher = smoother, longer-memory momentum; lower = more reactive to recent gradients, closer to plain SGD's instantaneous behavior. |
| `beta2` | 0.95–0.999 (LLM pretraining commonly uses 0.95, lower than the original 0.999 default) | Decay rate of the second-moment (variance) estimate. Lower `beta2` makes the adaptive scaling react faster to recent gradient-magnitude changes, which large-batch, high-variance transformer training tends to need; too low makes `v_hat_t` noisy and the effective step size unstable. |
| `epsilon` | 1e-8 (sometimes raised to 1e-6 for numerical stability in fp16/bf16 training) | Purely a numerical-stability floor to prevent division by ~0 when `v_hat_t` is very small (e.g. very early in training, or for parameters with genuinely tiny gradients). Not a tunable "performance" knob in the usual sense — it mostly matters when it's set too small for the numeric precision in use. |
| `weight_decay` | 0.01–0.1 | Strength of the decoupled L2-style shrinkage described in Section 5. Higher values regularize more aggressively; too high can prevent the model from fitting the data at all. |

### Learning rate schedules

A **constant** learning rate is almost never used for training an LLM from scratch. Instead, `lr`
is varied deliberately over the course of training according to a **schedule** — a function
`lr(t)` of the current step (or fraction of total training progress).

**Warmup** — Rather than starting training at the target peak learning rate, `lr` is ramped up
*linearly* from 0 (or a very small value) over the first `T_warmup` steps:

```
lr(t) = lr_peak * (t / T_warmup),     for t <= T_warmup
```

**Why warmup is critical for transformer training stability:** At the very start of training,
parameters are randomly initialized and the Adam second-moment estimate `v_t` has had almost no
gradients to average over yet — `v_hat_t` is a noisy, unreliable estimate of the true gradient
variance during exactly this period (this is the same early-training instability that bias
correction in Section 4.3 partially addresses, but bias correction alone is not sufficient at
transformer scale). If the full peak learning rate is applied immediately, the very large, poorly
calibrated early updates can push parameters into a bad region of the loss landscape from which
the model may never fully recover (in the worst case, this shows up as a loss spike or outright
divergence early in a pretraining run). This effect is empirically much more severe for
transformers than for smaller/older architectures, particularly because of the sensitivity of the
attention mechanism's softmax and normalization layers to very early, badly-scaled updates.
Warmup gives the optimizer's moment estimates time to stabilize against real gradient statistics
before the step size reaches its full, most aggressive value — every major transformer pretraining
run (GPT, LLaMA, etc.) uses some form of warmup, typically hundreds to a few thousand steps.

**Cosine decay** — After warmup, `lr` is decayed following one half-cycle of a cosine curve down
to a small minimum value `lr_min` (often `0`, or ~10% of `lr_peak`) by the end of training:

```
lr(t) = lr_min + 0.5 * (lr_peak - lr_min) * (1 + cos(pi * (t - T_warmup) / (T_total - T_warmup)))
```

Cosine decay is the most common schedule for LLM pretraining. It decreases slowly near the peak
(when big steps are still useful — the model is still far from convergence), decreases fastest
through the middle of training, and flattens out again near the end (allowing fine, stable
convergence into a good minimum without the last portion of training being wasted on a rate that's
still too high).

**Linear decay** — a simpler alternative, decreasing `lr` linearly from `lr_peak` to `lr_min` (or 0)
over the remaining steps after warmup:

```
lr(t) = lr_peak - (lr_peak - lr_min) * (t - T_warmup) / (T_total - T_warmup)
```

Linear decay is common in fine-tuning (including many LoRA fine-tuning setups) and in some
pretraining recipes; it's simpler to reason about than cosine but lacks the "flatten out near the
end" property that helps cosine's final convergence.

The general shape shared by all of these — **warmup, then decay** — is what actually matters:
start low to let the optimizer's internal state stabilize, ramp up to a peak once training dynamics
are healthy, then decay to let the model settle into a sharper, better-converged minimum by the end
of training.

---

## 7. Gradient clipping

### The problem: exploding gradients

Deep networks (transformers especially, due to their depth and the multiplicative interactions
across attention and residual layers) can occasionally produce gradients with enormous magnitude
for some batches — an "exploding gradient." This can happen from a single anomalous or
poorly-tokenized training example, numerical instability compounding across many layers, or just
an unlucky region of the loss landscape with very high local curvature. Left unchecked, an
oversized gradient produces an oversized parameter update, which can push weights far outside any
reasonable range in a single step — corrupting the model's learned representations (sometimes
irrecoverably) and frequently manifesting as a visible **loss spike** in the training curve, or
outright `NaN`/`Inf` values propagating through subsequent steps.

### The fix: clip by global norm

The standard mitigation is **gradient clipping by global norm**: compute the L2 norm of the entire
gradient vector (concatenated across every parameter in the model), and if it exceeds a threshold
`max_norm`, rescale the *entire* gradient vector down so its norm equals exactly `max_norm`:

```
g_norm = ||g||_2 = sqrt( sum_i g_i^2 )        # L2 norm over ALL parameters, flattened

if g_norm > max_norm:
    g = g * (max_norm / g_norm)
```

A few properties worth being precise about:

- Clipping is applied to the gradient **before** it reaches the optimizer's update rule — it
  operates on `g_t` directly, upstream of the momentum/variance accumulation in Adam/AdamW.
- It rescales **direction-preservingly**: every component of `g` is scaled by the same factor
  `max_norm / g_norm`, so the direction of the gradient is unchanged — only its magnitude is
  capped. This is why it's "clip by norm" and not "clip by value" (which clamps each individual
  gradient component independently and does distort direction).
- The norm is computed **globally**, across all parameters in the model at once, not per-layer or
  per-parameter-tensor — a single global scale factor is applied everywhere, which preserves the
  relative gradient magnitudes between different layers.
- Typical `max_norm` values for transformer training are around `1.0`, though this is tuned per
  model/setup.
- Clipping only ever *activates* on batches whose gradient norm actually exceeds `max_norm` — for
  the vast majority of well-behaved training steps it is a no-op, functioning purely as a safety
  valve against rare pathological gradients rather than as a constant damping force.

Gradient clipping is orthogonal to everything else in this document — it's a pre-processing step
applied to the raw gradient before it enters whichever optimizer (SGD, Adam, AdamW) is being used,
and virtually every LLM training run uses it as a standard defense against loss spikes.

---

## 8. The training loop and optimizer state

Putting everything together, one training step looks like this:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ONE TRAINING STEP                              │
│                                                                               │
│   ┌───────────────┐                                                         │
│   │  mini-batch   │                                                         │
│   │  of tokens    │                                                         │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────────────┐                                                 │
│   │      FORWARD PASS      │   y_hat = model(x; theta_t)                    │
│   │  (compute activations  │   caches intermediate activations for          │
│   │   layer by layer)      │   use during backward pass                    │
│   └───────────┬───────────┘                                                 │
│               │                                                             │
│               ▼                                                             │
│   ┌───────────────────────┐                                                 │
│   │     LOSS FUNCTION      │   L = cross_entropy(y_hat, y_true)             │
│   └───────────┬───────────┘                                                 │
│               │                                                             │
│               ▼                                                             │
│   ┌───────────────────────┐                                                 │
│   │     BACKWARD PASS      │   g_t = grad(L)(theta_t)   (backpropagation,   │
│   │   (chain rule through  │   layer by layer, using cached activations)    │
│   │    the computation     │                                               │
│   │    graph)               │                                               │
│   └───────────┬───────────┘                                                 │
│               │                                                             │
│               ▼                                                             │
│   ┌───────────────────────┐                                                 │
│   │   GRADIENT CLIPPING    │   if ||g_t|| > max_norm:                       │
│   │   (global L2 norm)     │       g_t *= max_norm / ||g_t||                │
│   └───────────┬───────────┘                                                 │
│               │                                                             │
│               ▼                                                             │
│   ┌───────────────────────────────────────────────────────────────┐        │
│   │                     optimizer.step()                            │        │
│   │  ┌─────────────────────────────────────────────────────────┐  │        │
│   │  │  READS  persistent optimizer state from step t-1:        │  │        │
│   │  │      m_{t-1}   (momentum / 1st-moment buffer)             │  │        │
│   │  │      v_{t-1}   (variance / 2nd-moment buffer)             │  │        │
│   │  ├─────────────────────────────────────────────────────────┤  │        │
│   │  │  m_t = beta1*m_{t-1} + (1-beta1)*g_t                      │  │        │
│   │  │  v_t = beta2*v_{t-1} + (1-beta2)*g_t^2                    │  │        │
│   │  │  m_hat_t = m_t / (1 - beta1^t)                             │  │        │
│   │  │  v_hat_t = v_t / (1 - beta2^t)                             │  │        │
│   │  │  theta_t = theta_{t-1}                                      │  │        │
│   │  │            - lr(t) * weight_decay * theta_{t-1}            │  │        │
│   │  │            - lr(t) * m_hat_t / (sqrt(v_hat_t) + eps)       │  │        │
│   │  ├─────────────────────────────────────────────────────────┤  │        │
│   │  │  WRITES updated state forward to step t:                  │  │        │
│   │  │      m_t   ──────────────────────────────►  m_{t}  ───┐   │  │        │
│   │  │      v_t   ──────────────────────────────►  v_{t}  ───┤   │  │        │
│   │  └─────────────────────────────────────────────────────┼───┘  │        │
│   └───────────────────────────────────────────────────────┼──────┘        │
│               │                                             │               │
│               ▼                                             │  (persists   │
│   ┌───────────────────────┐                                 │   into next  │
│   │   theta_{t+1} ready    │                                 │   step)      │
│   └───────────┬───────────┘                                 │               │
│               │                                             │               │
│               └──────────────── next mini-batch ◄───────────┘               │
│                                  (t = t + 1, loop back to FORWARD PASS)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

The detail worth internalizing from this diagram: **`theta` is not the only state that persists
across steps.** `m_t` and `v_t` (the momentum and variance buffers) are exactly as much a part of
"the optimizer" as the update formula itself, and they carry information forward from every
previous step of training. Concretely, this means an AdamW optimizer for a model with `N`
parameters requires:

```
theta:  N parameters      (the model itself, fp32 or bf16/fp16)
m_t:    N values           (first-moment buffer, one per parameter, kept in fp32)
v_t:    N values           (second-moment buffer, one per parameter, kept in fp32)
```

For a model trained in mixed precision, it's standard to also keep a full fp32 master copy of
`theta` for the optimizer to update precisely, plus the fp16/bf16 copy used for the forward/backward
compute — so AdamW's total memory footprint for optimizer state alone is commonly **8 bytes per
parameter** (4 bytes each for `m_t` and `v_t` in fp32) on top of the model weights themselves,
before counting gradients or the fp32 master-weight copy. For a 7B-parameter model, that's roughly
56 GB just for `m_t` and `v_t` — often larger than the memory used by the model weights and
gradients combined. This is exactly the cost that Section 9 shows PEFT methods like LoRA sidestep.

---

## 9. Optimizers and LoRA/PEFT

Everything above assumes the optimizer maintains momentum and variance buffers for *every trainable
parameter*. For full fine-tuning (or pretraining) of an LLM, "every trainable parameter" means
every parameter in the model — which, as shown in Section 8, makes the optimizer state alone one of
the largest memory consumers in the entire training pipeline (often exceeding the model weights
themselves for Adam/AdamW).

[LoRA](../08-lora-qlora/README.md) changes what "trainable parameter" means. As detailed there,
LoRA freezes the entire base model and injects small, trainable low-rank matrices (`A` and `B`,
with rank `r` typically in the range 4–64) alongside a handful of the original weight matrices,
replacing a full weight update `Delta_W` with a low-rank product:

```
W' = W_frozen + (alpha / r) * B @ A
```

`W_frozen` never receives a gradient — it's excluded from the optimizer entirely, not just "kept
from changing much." Only `A` and `B` (typically well under 1% of the base model's total parameter
count) are passed to the optimizer as trainable parameters.

This has a direct, mechanical consequence for the optimizer state described in Section 8: **AdamW's
`m_t` and `v_t` buffers are only ever allocated for `A` and `B`, not for `W_frozen` or any other
frozen base-model parameter.** For a 7B-parameter base model fine-tuned with LoRA adapters totaling
(for example) ~20M trainable parameters:

```
Full fine-tuning AdamW state:   ~7B params * 8 bytes  ≈ 56 GB  (m_t + v_t, fp32)
LoRA fine-tuning AdamW state:   ~20M params * 8 bytes ≈ 160 MB (m_t + v_t, fp32)
```

That's roughly a 350x reduction in optimizer-state memory for this example, and it compounds with
the other well-known LoRA benefits (far fewer gradients need to be computed and stored, since
`W_frozen` requires no gradient at all) and, in the QLoRA case, with quantizing `W_frozen` itself
down to 4-bit NF4 storage (see [Section 5 of the LoRA/QLoRA doc](../08-lora-qlora/README.md#5-qlora-fitting-a-7b70b-model-on-one-gpu)
for the full breakdown of QLoRA's memory math, including paged optimizers).

The optimization *algorithm* — AdamW, its bias-corrected moment estimates, its decoupled weight
decay, a warmup+cosine learning rate schedule, gradient clipping — is exactly the same algorithm
described in this document, applied without modification. What changes under LoRA is purely the
*size of the parameter vector `theta` that the optimizer is asked to track state for.* This is why
PEFT methods let you fine-tune a 7B–70B parameter model on a single consumer or prosumer GPU: not
because the optimizer got any smarter, but because it's only ever bookkeeping momentum and
variance for a tiny fraction of the model's total parameters.

---

## 10. Summary table

| Optimizer | Uses momentum? | Adapts per-parameter LR? | Decay of squared-gradient history | Weight decay handling |
|---|---|---|---|---|
| Gradient descent (batch/SGD/mini-batch) | No | No | N/A | None (or L2-in-gradient if added manually) |
| SGD + Momentum | Yes (`v_t`) | No | N/A | None (or L2-in-gradient if added manually) |
| AdaGrad | No | Yes | None — sum grows forever, LR decays to ~0 | L2-in-gradient if added |
| RMSProp | No (typically) | Yes | Exponential moving average (fixes AdaGrad's decay) | L2-in-gradient if added |
| Adam | Yes (`m_t`, bias-corrected) | Yes (`v_t`, bias-corrected) | Exponential moving average | Coupled: L2 folded into gradient, distorted by adaptive scaling |
| **AdamW** | Yes (`m_t`, bias-corrected) | Yes (`v_t`, bias-corrected) | Exponential moving average | **Decoupled**: applied directly to weights, independent of gradient/variance |

---

## 11. References

- Robbins & Monro, *A Stochastic Approximation Method* (1951) — foundational stochastic
  approximation theory underlying SGD.
- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* (1986) —
  backpropagation.
- Sutskever et al., *On the importance of initialization and momentum in deep learning* (2013) —
  momentum and Nesterov momentum in deep learning.
- Duchi, Hazan & Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic
  Optimization* (2011) — AdaGrad.
- Tieleman & Hinton, *RMSProp: Divide the gradient by a running average of its recent magnitude*
  (2012 lecture notes) — RMSProp.
- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014) — Adam.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2017/2019, ICLR) — AdamW.
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (2016) — cosine
  learning rate schedules.
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (2017) — learning rate
  warmup for large-batch training.
- Pascanu, Mikolov & Bengio, *On the difficulty of training Recurrent Neural Networks* (2013) —
  exploding gradients and gradient clipping by norm.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) — see also
  [`docs/ml-concepts/08-lora-qlora/README.md`](../08-lora-qlora/README.md) in this repo.
