"""MCPUtilityAgent — an agent whose ENTIRE tool set comes from MCP
(agent/mcp_client.py), not the PM domain's tool schema, proving MCP is a
genuine capability source for a dedicated ALGO AKANADE agent — not just an
add-on merged into the PM agent's much larger tool list (where it's easy
to miss). Registers whichever tools config.py's MCP_SERVERS currently
exposes (the bundled utility_server.py by default: get_current_datetime,
days_between) — point MCP_SERVERS at a real external server and this
agent gains its tools with zero code changes here.
"""
from __future__ import annotations

from ...agent import mcp_client
from ...llm.router import get_llm_client
from ..agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from ..task import Task, TaskType

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You answer scheduling/date-math questions using the tools available "
              "to you. If no tool fits the question, say so plainly.",
}


class MCPUtilityAgent(Agent):
    def __init__(self):
        tool_names = tuple(mcp_client.mcp_tool_names())
        self.metadata = AgentMetadata(
            agent_id="mcp-utility-agent",
            name="MCP Utility Agent",
            description="Answers scheduling/date-math questions using ONLY "
                       "MCP-sourced tools (config.py's MCP_SERVERS) — no PM tools, "
                       "no domain logic. Its capability set IS whatever the "
                       "configured MCP server(s) currently expose.",
            capabilities=frozenset({"date_math", "scheduling_lookup"}),
            supported_tasks=frozenset({TaskType.GENERIC, TaskType.QUESTION_ANSWERING}),
            model="configurable (via llm/router.py's provider chain)",
            tools=tool_names,
            rag_enabled=False,
            cost_per_call_usd=0.001,
            latency_profile=LatencyProfile(p50_ms=400, p95_ms=1500),
            version="1.0.0",
            reliability_score=0.7,
            status=AgentStatus.ACTIVE,
        )

    def understand(self, task: Task) -> dict:
        return {"text": task.text}

    def plan(self, task: Task, understanding: dict) -> dict:
        return {"messages": [SYSTEM_PROMPT, {"role": "user", "content": understanding["text"]}]}

    def execute(self, task: Task, plan: dict) -> dict:
        client = get_llm_client("realtime", preferred=task.metadata.get("preferred_provider"))
        # ONLY MCP tool specs — not tools.openai_tool_specs()'s full PM
        # set — this agent's whole point is being scoped to MCP alone.
        result = client.chat("mcp_utility_respond", plan["messages"], mcp_client.mcp_tool_specs())
        return {"result": result}

    def validate(self, task: Task, result: dict) -> dict:
        raw = result["result"]
        if raw.tool_call:
            try:
                tool_result = mcp_client.call_mcp_tool(raw.tool_call["name"], raw.tool_call["arguments"])
                result["tool_text"] = tool_result["text"]
            except Exception as e:
                result["tool_text"] = f"Sorry — that failed: {e}"
        return result

    def respond(self, task: Task, validated: dict) -> AgentResult:
        raw = validated["result"]
        if "tool_text" in validated:
            return AgentResult(text=validated["tool_text"], ok=True, confidence=0.85,
                               tool_calls=(raw.tool_call["name"],), raw={"provider": raw.provider})
        ok = not bool(raw.error)
        text = raw.text if ok else f"Sorry — every model rung failed: {raw.error}"
        return AgentResult(text=text, ok=ok, confidence=0.6 if ok else 0.0)
