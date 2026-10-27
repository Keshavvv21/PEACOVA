"""Registers this deployment's default agents into the shared registry —
called once at process startup (see app/main.py). Adding a new agent to
a running ALGO AKANADE deployment never requires editing this file: it's just
`registry.register(MyAgent())` from wherever that agent's own module
lives (or a `POST /api/v1/agents/register` call at runtime, see
routers/v1/agents.py) — this module only seeds the agents this
particular deployment ships with by default:

  pm-agent                 — the pre-existing PM chatbot (§ pm_agent.py)
  pm-agent-adk              — the SAME PM chatbot, pinned to the "adk"
                              framework track specifically — Google ADK
                              exposed as its own independently-routable,
                              independently-scored ALGO AKANADE agent, not just
                              a query-param buried inside pm-agent
  openclaw-agent            — general-purpose, direct public Claude API
  coding-agent               — code generation + real ast.parse validation
  mcp-utility-agent           — tool set is ENTIRELY MCP-sourced
                                (agent/mcp_client.py) — proves MCP is a
                                real capability source, not PM-only
  langchain-research-agent     — a real langchain_core Runnable + a real
                                  BaseRetriever over ALGO AKANADE's RAG store
  code-security-agent         — real static pattern checks for common
                                Python vulnerability classes
  supply-chain-agent           — checks pinned dependencies against
                                 OSV.dev, a real public vulnerability DB
  cloud-config-agent           — real pattern checks over Dockerfile/
                                 compose/Kubernetes/Terraform content for
                                 known misconfigurations
"""
from __future__ import annotations

from .agents.cloud_config_agent import CloudConfigAgent
from .agents.code_security_agent import CodeSecurityAgent
from .agents.coding_agent import CodingAgent
from .agents.langchain_research_agent import build_langchain_research_agent
from .agents.mcp_utility_agent import MCPUtilityAgent
from .agents.openclaw_agent import OpenClawAgent
from .agents.pm_agent import PMAgent
from .agents.supply_chain_agent import SupplyChainAgent
from .network import network
from .registry import registry


def bootstrap_default_agents() -> None:
    agents = [
        PMAgent(),
        PMAgent(framework="adk", agent_id="pm-agent-adk", name="PM Agent (ADK)",
               description="The same PM chatbot as pm-agent, pinned to Google ADK's "
                          "Runner/LlmAgent machinery specifically — an 'outside' "
                          "agent framework integrated as its own independently-"
                          "routable, independently-scored ALGO AKANADE agent."),
        OpenClawAgent(),
        CodingAgent(),
        MCPUtilityAgent(),
        build_langchain_research_agent(),
        CodeSecurityAgent(),
        SupplyChainAgent(),
        CloudConfigAgent(),
    ]
    for agent in agents:
        if agent.metadata.agent_id not in registry:
            registry.register(agent)
        network.add_agent(agent.metadata)

    # A declared dependency edge, purely as a documented example of §13's
    # capability graph — the PM agent's own output is a natural input to
    # a follow-up code-generation ask ("now write the migration script
    # for that schema change"). Nothing currently forces this edge to be
    # walked; orchestrator.py still routes by the bandit's score, not by
    # graph traversal alone.
    network.declare_dependency("pm-agent", "coding-agent", weight=0.5)
