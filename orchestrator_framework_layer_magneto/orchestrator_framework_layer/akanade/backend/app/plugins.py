"""JSON tool plugins — extend the chat bot's PM actions by dropping a
`.json` file in `plugins.d/`, no Python required for the common case (call
one existing, whitelisted pm_service method with resolved entity ids).

Deliberately NOT free-form code execution: `service_call.method` must
already be in SERVICE_METHOD_REGISTRY below — a fixed whitelist, the same
"the bot never gets its own privileges, only a fixed named tool set"
principle (§11) that already governs every hand-written tool in
agent/tools.py. A JSON file wires a NEW PHRASE to an EXISTING, already-
reviewed, already-role-gated action; it can never invent a new action or
call arbitrary code.

Lives here — a sibling of agent/ and llm/, not inside either — so both
agent/tools.py (entity resolution + execution) and llm/providers.py
(rule-based trigger matching, the offline floor) can import it without an
agent<->llm import cycle: agent already depends on llm (native_agent.py
imports llm.router), so llm must never import from agent. See
plugins.d/README.md for the file format and a worked example
(rename_project.json), and the main README's "Adding a PM action via
JSON" section for the end-to-end picture.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .services import pm_service

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).parent / "plugins.d"

# The only pm_service callables a JSON plugin may invoke. Registering a new
# method here is still a Python-side, reviewed decision (is this action
# safe to expose to chat, and does it already do its own role check via
# _need?) — only the wiring from a chat phrase to an ALREADY-approved
# method is JSON-editable.
SERVICE_METHOD_REGISTRY: dict[str, Callable[..., dict]] = {
    "rename_project": pm_service.rename_project,
}


@dataclass
class ToolPlugin:
    name: str
    description: str
    response_template: str
    service_call: dict                        # {"method": str, "args": {kwarg: resolved_key}}
    required: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    trigger_patterns: list[re.Pattern] = field(default_factory=list)
    resolve: dict[str, dict] = field(default_factory=dict)  # arg_name -> {"entity": ..., "as": ...}
    destructive: bool = False

    @property
    def schema(self) -> dict:
        """The same {description, required, properties} shape every
        hand-written entry in agent.tools.TOOL_SCHEMAS uses — a plugin is
        indistinguishable from a built-in tool to perceive_intent/output_guard."""
        return {"description": self.description, "required": self.required,
                "properties": self.properties}


def _load_one(path: Path) -> ToolPlugin | None:
    try:
        raw = json.loads(path.read_text())
        method_name = raw["service_call"]["method"]
        if method_name not in SERVICE_METHOD_REGISTRY:
            logger.warning("plugin %s: unregistered service_call.method %r — skipped",
                           path.name, method_name)
            return None
        return ToolPlugin(
            name=raw["name"],
            description=raw["description"],
            response_template=raw["response_template"],
            service_call=raw["service_call"],
            required=raw.get("required", []),
            properties=raw.get("properties", {}),
            trigger_patterns=[re.compile(p, re.I) for p in raw.get("trigger_patterns", [])],
            resolve=raw.get("resolve", {}),
            destructive=bool(raw.get("destructive", False)),
        )
    except (json.JSONDecodeError, KeyError, re.error) as e:
        # A malformed plugin is a config error, not a reason the whole bot
        # fails to start — same "optional, additive" philosophy as
        # EXTRA_LLM_PROVIDERS (llm/router.py).
        logger.warning("plugin %s: malformed (%s) — skipped", path.name, e)
        return None


def load_plugins() -> dict[str, ToolPlugin]:
    if not PLUGIN_DIR.is_dir():
        return {}
    plugins: dict[str, ToolPlugin] = {}
    for path in sorted(PLUGIN_DIR.glob("*.json")):
        plugin = _load_one(path)
        if plugin is not None:
            plugins[plugin.name] = plugin
    return plugins


# Loaded once at import time — same lifecycle as agent.tools.TOOL_SCHEMAS,
# which merges this in (see agent/tools.py). Restart the process to pick up
# a new/edited plugin file, same as any other config-driven registry here.
PLUGINS: dict[str, ToolPlugin] = load_plugins()
