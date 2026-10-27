"""Settings for the Optimization Dashboard — a small, deliberately
SEPARATE FastAPI service from Algo Akanade's own backend (its own login, its
own JWT secret, its own process). It reads Algo Akanade's durable state files
(app/db.py's LocalStore JSON tables) directly off disk rather than
calling Algo Akanade's HTTP API, so this dashboard keeps working even when
Algo Akanade's own process isn't currently running (a real requirement for a
"what happened yesterday" view) — see README_OPTIMIZATION_DASHBOARD.md
§ Architecture for why.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# .../performance_optimizer/dashboard/backend/app/config.py
#   parents[0]=app  [1]=backend  [2]=dashboard  [3]=performance_optimizer
#   [4]=orchestrator_framework_layer — from there, akanade/backend/data
# is this repo's actual (dev checkout) layout; ALGO_AKANADE_DATA_DIR overrides
# it for any other deployment. In the container this path never exists
# (the image only has /srv/app/app/... — four levels up runs off the end
# of the filesystem, index-erroring on Path.parents), so the fallback
# is computed defensively: real value in a dev checkout, a harmless
# "./data" default in a container, where ALGO_AKANADE_DATA_DIR is ALWAYS
# expected to be set explicitly via -e (see Dockerfile) anyway.
try:
    _REPO_DEFAULT_ALGO_AKANADE_DATA_DIR = str(
        Path(__file__).resolve().parents[4] / "akanade" / "backend" / "data"
    )
except IndexError:
    _REPO_DEFAULT_ALGO_AKANADE_DATA_DIR = "./data"


@dataclass
class Settings:
    ALGO_AKANADE_DATA_DIR: str = field(
        default_factory=lambda: os.getenv("ALGO_AKANADE_DATA_DIR", _REPO_DEFAULT_ALGO_AKANADE_DATA_DIR))

    # The Algo Engine's own OpenAI-compatible server (GET /stats) — best
    # effort: the GPU panel shows "not configured"/"unreachable" rather
    # than failing the whole dashboard when this is empty or down.
    ALGO_ENGINE_BASE_URL: str = field(default_factory=lambda: os.getenv("ALGO_ENGINE_BASE_URL", ""))

    # This dashboard's OWN admin credential — separate from every user
    # account in Algo Akanade's/Aksha PM's own Users table by design (the
    # explicit ask: "make a separate UI and password"). Change both via
    # env vars for anything beyond a laptop demo.
    DASHBOARD_ADMIN_USER: str = field(default_factory=lambda: os.getenv("DASHBOARD_ADMIN_USER", "admin"))
    DASHBOARD_ADMIN_PASSWORD: str = field(
        default_factory=lambda: os.getenv("DASHBOARD_ADMIN_PASSWORD", "admin"))

    DASHBOARD_JWT_SECRET: str = field(
        default_factory=lambda: os.getenv("DASHBOARD_JWT_SECRET",
                                          "dev-only-optimization-dashboard-secret"))
    DASHBOARD_JWT_TTL_HOURS: int = field(
        default_factory=lambda: int(os.getenv("DASHBOARD_JWT_TTL_HOURS", "12")))


@lru_cache
def settings() -> Settings:
    return Settings()
