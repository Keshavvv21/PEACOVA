"""ak-router's deterministic strategy needs a capability_tier (0..1) per
candidate — see the router-service plan's Phase 1. Built-in providers
(luna/max/glm/hy4/deepseek/groq/algo-engine/ollama/rule — hardcoded in
llm/router.py's _providers(), never a row in the "Providers" DB table)
get a hand-designed default here, in the SAME spirit as
framework/router.py's UtilityWeights: transparent, hand-set numbers an
operator can read and argue with, not a black box.

Every DB-backed provider (admin-added via admin_providers.py, or a
nightly-registered "algo-akanade-slm[-<domain>]" tag) carries its OWN
capability_tier as a real field on its Providers row instead — see
admin_providers.py's ProviderIn/ProviderOut and _db_provider_entries()
in this package's router.py. That field is the "Router flag/update"
closed loop from the plan: seeded conservatively at registration,
raised once a real Phase-5 benchmark run or a nightly PEFT promotion has
actual quality numbers for it.
"""
from __future__ import annotations

# Deliberately mirrors CHAINS' own hand-curated cheapest/fastest-first
# ordering (llm/router.py's module docstring) rather than inventing new
# numbers from nothing — a rung CHAINS already trusts to answer well
# gets a correspondingly higher tier here.
_DEFAULT_TIERS: dict[str, float] = {
    "luna": 0.85,
    "max": 0.9,
    "glm": 0.8,
    "hy4": 0.75,
    "deepseek": 0.75,
    "groq": 0.7,
    "algo-engine": 0.6,
    "ollama": 0.5,
    "ollama-coder": 0.55,
    "claude": 0.95,
    "huggingface": 0.5,
    "custom": 0.5,
    "rule": 0.0,  # the deterministic floor is never a real model answer
}

_FALLBACK_TIER = 0.4  # an unlisted provider with no configured tier at all


def default_tier_for(provider_key: str) -> float:
    return _DEFAULT_TIERS.get(provider_key, _FALLBACK_TIER)
