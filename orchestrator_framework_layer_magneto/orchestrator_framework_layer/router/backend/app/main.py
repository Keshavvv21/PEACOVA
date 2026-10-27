"""ak-router — the standalone provider/model-ordering service. See the
implementation plan (Deterministic Baseline Router + Router Service +
Model Benchmarking) for the full architecture. Deliberately tiny: no
session state, no provider credentials, no chat/tool logic — just
"given these eligible candidates, what order should ak-chatbot try
them in."
"""
from __future__ import annotations

from fastapi import FastAPI

from .routes import router
from .routes_generic import router as generic_router

app = FastAPI(title="Algo Kaizen Router", version="1.0")
app.include_router(router)
app.include_router(generic_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
