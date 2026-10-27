from app.models import Candidate, RouteContext
from app.thompson import ARMS, TwoArmThompsonPolicy


def _fresh_policy() -> TwoArmThompsonPolicy:
    # A fresh instance per test (not the module singleton) — this
    # exercises the policy's own logic in isolation from whatever
    # persisted bandit state other tests in this session may have
    # written to the shared DATA_DIR (see conftest.py's own note on
    # session-scoped storage).
    return TwoArmThompsonPolicy()


def test_pick_only_ever_returns_a_designated_arm():
    policy = _fresh_policy()
    candidates = [
        Candidate(id="luna", capability_tier=0.9),
        Candidate(id="groq", capability_tier=0.7),
        Candidate(id="algo-akanade-slm", capability_tier=0.5),
        Candidate(id="deepseek", capability_tier=0.6),
    ]
    context = RouteContext(tier="realtime", est_tokens=500)
    for _ in range(20):  # sampling is stochastic — check it never drifts
        pick = policy.pick(candidates, context)
        assert pick in ARMS, f"TS picked {pick!r}, which isn't one of {ARMS}"


def test_pick_is_none_when_neither_arm_is_a_candidate():
    policy = _fresh_policy()
    candidates = [Candidate(id="luna", capability_tier=0.9), Candidate(id="deepseek", capability_tier=0.6)]
    context = RouteContext(tier="realtime", est_tokens=500)
    assert policy.pick(candidates, context) is None


def test_update_ignores_non_arm_candidate_ids():
    policy = _fresh_policy()
    context = RouteContext(tier="realtime", est_tokens=500)
    # Must not raise, and must not create bandit state for a
    # non-designated arm — only "groq"/"algo-akanade-slm" ever learn.
    policy.update("luna", context, ok=True, latency_ms=100.0)
    assert policy._core.posterior_mean("luna") is None


def test_pick_ucb_only_ever_returns_a_designated_arm():
    policy = _fresh_policy()
    candidates = [
        Candidate(id="luna", capability_tier=0.9),
        Candidate(id="groq", capability_tier=0.7),
        Candidate(id="algo-akanade-slm", capability_tier=0.5),
        Candidate(id="deepseek", capability_tier=0.6),
    ]
    context = RouteContext(tier="realtime", est_tokens=500)
    for _ in range(20):
        pick = policy.pick_ucb(candidates, context)
        assert pick in ARMS, f"UCB picked {pick!r}, which isn't one of {ARMS}"


def test_pick_ucb_is_none_when_neither_arm_is_a_candidate():
    policy = _fresh_policy()
    candidates = [Candidate(id="luna", capability_tier=0.9), Candidate(id="deepseek", capability_tier=0.6)]
    context = RouteContext(tier="realtime", est_tokens=500)
    assert policy.pick_ucb(candidates, context) is None


def test_pick_ucb_is_deterministic_given_the_same_posterior():
    # Unlike pick() (a posterior sample, stochastic), pick_ucb() is a pure
    # argmax over mean + confidence radius — repeat calls with no update()
    # in between must always agree.
    policy = _fresh_policy()
    candidates = [Candidate(id="groq", capability_tier=0.7), Candidate(id="algo-akanade-slm", capability_tier=0.5)]
    context = RouteContext(tier="realtime", est_tokens=500)
    picks = {policy.pick_ucb(candidates, context) for _ in range(20)}
    assert len(picks) == 1


def test_pick_ucb_favors_the_less_explored_arm_after_one_arm_learns_a_strong_reward():
    # After many real successes for "groq" alone, UCB's confidence radius
    # for the untouched "algo-akanade-slm" arm should keep it competitive
    # (or winning) purely from uncertainty, not learned reward — the
    # classic explore-the-unknown-arm behavior UCB is chosen for.
    policy = _fresh_policy()
    context = RouteContext(tier="realtime", est_tokens=500)
    for _ in range(30):
        policy.update("groq", context, ok=True, latency_ms=100.0)
    candidates = [Candidate(id="groq", capability_tier=0.7), Candidate(id="algo-akanade-slm", capability_tier=0.5)]
    assert policy._core.posterior_mean("algo-akanade-slm") is None
    # Never raises even though the untouched arm has no posterior yet —
    # ensure() inside select_ucb() must seed it on the fly.
    pick = policy.pick_ucb(candidates, context)
    assert pick in ARMS
