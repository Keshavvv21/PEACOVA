"""ALGO AKANADE — Multi-Agent GenAI Network with Evolving Training and Orchestration.

A generic, domain-agnostic layer for registering heterogeneous AI agents
(PM, Coding, Research, ...), routing a task to the best agent(s) via an
adaptive, self-learning policy, executing them (sequentially, in parallel,
hierarchically, with fallback/retry/consensus), and closing the loop with
real feedback that improves future routing.

This package does not know anything about project management specifically
— the existing PM chatbot (agent/native_agent.py, agent/langgraph_agent.py,
agent/adk_agent.py, agent/claude_agent.py, agent/tools.py, ...) is wired in
as ONE registered agent (gmaofn.agents.pm_agent.PMAgent) alongside others
(gmaofn.agents.coding_agent.CodingAgent,
gmaofn.agents.openclaw_agent.OpenClawAgent, ...) — proving the framework
is generic by construction, not by claim: the PM domain is a tenant of
ALGO AKANADE, not baked into it.

See README.md's "ALGO AKANADE" section for the full architecture, the routing
math (Thompson Sampling contextual bandit + Shapley-value credit
assignment), and how to register a new agent/model/tool/RAG backend
without touching this package's core.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
