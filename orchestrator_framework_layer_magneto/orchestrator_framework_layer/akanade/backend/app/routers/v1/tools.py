"""§11 — the Tool Registry's REST surface. Every tool any agent can call
right now: the PM domain's hand-written tools, its JSON plugins
(plugins.d/*.json), and any discovered MCP tool (mcp_client.py) — all
already merged into one schema set by agent/tools.py; this endpoint just
exposes that set generically, with each tool's real JSON-Schema
input/output shape."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...agent import tools
from ...auth import current_user

router = APIRouter(prefix="/api/v1/tools", tags=["gmaofn-tools"])


@router.get("")
def list_tools(_: dict = Depends(current_user)):
    return {"tools": [
        {"tool_id": name, "name": name, "description": schema["description"],
         "input_schema": {"type": "object", "properties": schema["properties"],
                          "required": schema["required"]},
         "destructive": name in tools.DESTRUCTIVE}
        for name, schema in tools.TOOL_SCHEMAS.items()
    ]}
