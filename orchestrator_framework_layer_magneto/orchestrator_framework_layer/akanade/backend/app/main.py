"""ALGO AKANADE — Multi-Agent GenAI Network with Evolving Training and Orchestration.

The single FastAPI process (§2.1): the generic multi-agent orchestration
core (app/gmaofn/ — registry, adaptive router, workflow engine, feedback
loop), the PM domain built on top of it as one tenant (the mini Aksha PM
API + its chat agents), graph + vector retrieval, and the WebSocket
broadcast, all in-process. `uvicorn app.main:app`.

See app/gmaofn/README.md for the framework itself; this file just wires
its routers alongside the PM domain's own."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import seed
from .config import settings
from .framework.bootstrap import bootstrap_default_agents
from .framework.registry import registry as gmaofn_registry
from .retrieval.vectorstore import vector_index
from .routers import (acp, admin_eval_reports, admin_finetune_config, admin_ground_truth,
                     admin_model_benchmarks, admin_provider_budgets, admin_providers,
                     admin_router_config, admin_security, admin_slm, chat, chat_usage,
                     coding_usage, oauth, peft_jobs, peft_training, pm)
from .routers import v1 as gmaofn_v1

app = FastAPI(title=settings().APP_NAME, version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(pm.router)
app.include_router(coding_usage.router)
app.include_router(chat_usage.router)
app.include_router(chat.router)
app.include_router(admin_providers.router)
app.include_router(admin_security.router)
app.include_router(admin_slm.router)
app.include_router(admin_finetune_config.router)
app.include_router(admin_eval_reports.router)
app.include_router(admin_ground_truth.router)
app.include_router(admin_provider_budgets.router)
app.include_router(admin_router_config.router)
app.include_router(admin_model_benchmarks.router)
app.include_router(acp.router)
app.include_router(oauth.router)
app.include_router(peft_training.router)
app.include_router(peft_jobs.router)
for _router in gmaofn_v1.ALL_ROUTERS:
    app.include_router(_router)


@app.on_event("startup")
def _startup():
    if seed.seed():
        print("seeded demo data (users: aniruddha/meera/rahul/priya, pw demo123)")
    bootstrap_default_agents()


@app.get("/api/health")
def health():
    return {"status": "ok", "framework": settings().CHAT_AGENT_FRAMEWORK,
            "vector_backend": vector_index.backend,
            "embedder": vector_index.embedder.name,
            "storage": settings().STORAGE_BACKEND,
            "gmaofn_agents": [a.metadata.agent_id for a in gmaofn_registry.list_agents()]}


# dev convenience: SERVE_FRONTEND=1 mounts ../frontend at / so the whole
# stack runs as one uvicorn process without nginx (compose is the real
# two-app deployment; this flag is for laptops and CI).
import os
if os.getenv("SERVE_FRONTEND") == "1":
    from fastapi.staticfiles import StaticFiles
    _fe = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "frontend")
    if os.path.isdir(_fe):
        app.mount("/", StaticFiles(directory=_fe, html=True), name="frontend")
