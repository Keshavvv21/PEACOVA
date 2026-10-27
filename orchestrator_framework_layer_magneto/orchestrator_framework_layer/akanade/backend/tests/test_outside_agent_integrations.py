"""Proves the three "outside integration" agents bootstrap.py registers
are real and working, not just latent adapter modules:

  pm-agent-adk              — Google ADK, pinned as its own agent
  mcp-utility-agent          — MCP-sourced tools as an agent's ENTIRE capability
  langchain-research-agent    — a real langchain_core Runnable + BaseRetriever
"""
from __future__ import annotations

from app.framework.bootstrap import bootstrap_default_agents
from app.framework.registry import registry
from app.framework.task import Task, TaskContext, TaskType

bootstrap_default_agents()


def test_all_default_agents_are_registered():
    ids = {a.metadata.agent_id for a in registry.list_agents()}
    assert ids == {"pm-agent", "pm-agent-adk", "openclaw-agent", "coding-agent",
                   "mcp-utility-agent", "langchain-research-agent",
                   "code-security-agent", "supply-chain-agent", "cloud-config-agent"}


def test_adk_pm_agent_is_independently_routable_and_uses_adk():
    agent = registry.get("pm-agent-adk")
    assert agent.framework == "adk"
    assert agent.metadata.agent_id != registry.get("pm-agent").metadata.agent_id
    exec_user = {"sub": "aniruddha", "name": "Aniruddha", "role": "Executive"}
    result = agent.run(Task(text="list all projects", context=TaskContext(),
                            metadata={"user": exec_user}))
    assert result.ok is True


def test_mcp_utility_agent_capability_set_is_entirely_mcp_sourced():
    from app.agent import mcp_client

    agent = registry.get("mcp-utility-agent")
    assert agent.metadata.tools  # non-empty — real MCP tools were discovered
    assert set(agent.metadata.tools) <= mcp_client.mcp_tool_names()


def test_mcp_utility_agent_validate_and_respond_use_a_real_mcp_tool_call():
    """This suite runs fully hermetic (conftest.py blanks every LLM
    provider key at import time, session-wide) — so execute()'s own
    model call always bottoms out at the rule-based floor, which (having
    no hardcoded pattern for an MCP-discovered tool) can't actually pick
    days_between out of a free-text question; that requires a real
    model's function-calling. Proven live against a real provider before
    this test was written. What's tested here instead — hermetically,
    like the rest of this suite — is this agent's OWN validate()/
    respond() logic: given an already-selected tool call (exactly the
    shape a real model's LLMResult carries), does it genuinely invoke
    mcp_client (a real subprocess round-trip, not a mock) and phrase the
    result correctly."""
    from app.framework.agents.mcp_utility_agent import MCPUtilityAgent
    from app.llm.providers import LLMResult

    agent = MCPUtilityAgent()
    task = Task(text="ignored — tool_call below stands in for a real model's pick",
               context=TaskContext(task_type=TaskType.GENERIC))
    fake_model_result = {"result": LLMResult(
        provider="fake", tool_call={"name": "days_between",
                                    "arguments": {"start_date": "2026-08-01", "end_date": "2026-09-15"}})}
    validated = agent.validate(task, fake_model_result)
    result = agent.respond(task, validated)
    assert result.ok is True
    assert "45 day" in result.text
    assert result.tool_calls == ("days_between",)


def test_langchain_research_agent_is_a_real_runnable_with_real_retrieval():
    """Same hermetic constraint as above — the chain's FINAL answer-
    synthesis step needs a real model (verified live before this test
    existed); what's tested hermetically here is that it's a genuine
    langchain_core Runnable, and that its retrieval stage (a real
    BaseRetriever over a real in-memory vector store, no LLM involved)
    actually pulls the RIGHT document for the question — proving the
    "RAG" half of this integration works without needing model access."""
    from langchain_core.runnables import Runnable

    from app.framework.agents.langchain_research_agent import _KNOWLEDGE_BASE
    from app.framework.langchain_adapter import vector_store_to_retriever
    from app.framework.rag import InMemoryVectorStoreAdapter, RagDocument

    agent = registry.get("langchain-research-agent")
    assert isinstance(agent.runnable, Runnable)

    store = InMemoryVectorStoreAdapter()
    store.add_documents([RagDocument(text=t) for t in _KNOWLEDGE_BASE])
    retriever = vector_store_to_retriever(store, k=1)
    docs = retriever.invoke("How does ALGO AKANADE split credit across multiple agents?")
    assert docs
    assert "shapley" in docs[0].page_content.lower()
