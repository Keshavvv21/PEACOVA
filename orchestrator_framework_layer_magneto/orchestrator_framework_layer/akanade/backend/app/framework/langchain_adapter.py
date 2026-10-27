"""§8 — LangChain integration as an OPTIONAL layer, not a foundation:

    Framework Core
          ↓
    LangChain Adapter (this module)
          ↓
    LangChain Tools / Retrievers / Runnables

Nothing in core.py/registry.py/router.py/workflow.py imports LangChain —
this module is the only place that does, and it's imported lazily (only
when one of the three functions below is actually called), so a ALGO AKANADE
deployment that never touches LangChain doesn't pay for it and, per §16's
own separation principle, the core stays usable if this adapter is
deleted entirely. `langchain-core` and `langgraph` are already real
dependencies here (the PM domain's own "langgraph" agent framework track
uses them) — this module is what makes them usable OUTSIDE that one
track too, for any ALGO AKANADE agent/tool.

Three real, working bridges:
  gmaofn_tool_to_langchain  — any ALGO AKANADE tool (a name/description/JSON
      Schema + a callable) becomes a real `langchain_core.tools.
      StructuredTool`, usable in a LangChain/LangGraph agent executor.
  vector_store_to_retriever — any `rag.VectorStoreAdapter` (Chroma-backed
      or the in-memory one, or a future FAISS/Qdrant/... adapter) becomes
      a real `langchain_core.retrievers.BaseRetriever` — §9's RAG layer,
      usable anywhere LangChain expects a retriever.
  LangChainRunnableAgent    — wraps an arbitrary LangChain `Runnable`
      (a prompt|model chain, a full `AgentExecutor`, a compiled LangGraph
      graph) as a ALGO AKANADE `Agent`, so a LangChain-built agent can be
      registered into ALGO AKANADE's registry/router/workflow engine exactly
      like a native one (agents/pm_agent.py, agents/coding_agent.py, ...).
"""
from __future__ import annotations

from typing import Any, Callable

from .agent_base import Agent, AgentMetadata, AgentResult
from .rag import RagHit, VectorStoreAdapter
from .task import Task

_JSON_TO_PY_TYPE = {"string": str, "number": float, "integer": int,
                    "boolean": bool, "array": list, "object": dict}


def _json_schema_to_pydantic_model(name: str, schema: dict):
    from pydantic import create_model

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _JSON_TO_PY_TYPE.get(prop_schema.get("type", "string"), str)
        default = ... if prop_name in required else None
        fields[prop_name] = (py_type if prop_name in required else py_type | None,
                             default if prop_name in required else None)
    return create_model(f"{name}Args", **fields)


def gmaofn_tool_to_langchain(name: str, description: str, parameters_schema: dict,
                            fn: Callable[..., str]):
    """Wraps a ALGO AKANADE tool (agent/tools.py's TOOL_SCHEMAS shape, or an MCP
    tool from mcp_client.mcp_tool_specs()) as a real LangChain
    StructuredTool — `fn` is whatever already executes it
    (e.g. `lambda **kwargs: format_result(tools.execute(user, name, kwargs))`)."""
    from langchain_core.tools import StructuredTool

    args_model = _json_schema_to_pydantic_model(name, parameters_schema)
    return StructuredTool.from_function(func=fn, name=name, description=description,
                                        args_schema=args_model)


def vector_store_to_retriever(store: VectorStoreAdapter, *, k: int = 5, filters: dict | None = None):
    """A real `BaseRetriever` subclass over any VectorStoreAdapter — so
    §9's RAG layer (Chroma-backed PM index, the in-memory adapter, or
    any future one) is usable anywhere LangChain expects a retriever
    (a LangChain RAG chain, a LangGraph node, ...)."""
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class _VectorStoreRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *,
                                    run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
            hits: list[RagHit] = store.search(query, k=k, filters=filters)
            return [Document(page_content=hit.text, metadata={**hit.metadata, "score": hit.score})
                   for hit in hits]

    return _VectorStoreRetriever()


class LangChainRunnableAgent(Agent):
    """Wraps an arbitrary LangChain `Runnable` — a prompt|model chain, a
    full `AgentExecutor`, a compiled LangGraph graph, anything with
    `.invoke({"input": ...}) -> {"output": ...}`-shaped I/O — as a ALGO AKANADE
    Agent, so it's registerable/routable/composable exactly like a native
    one. `understand`/`plan`/`validate` are near-identity: whatever
    reasoning the wrapped Runnable does internally is opaque to ALGO AKANADE by
    design (that's the whole point of delegating to it)."""

    def __init__(self, metadata: AgentMetadata, runnable, *,
                input_key: str = "input", output_key: str = "output"):
        self.metadata = metadata
        self.runnable = runnable
        self.input_key = input_key
        self.output_key = output_key

    def understand(self, task: Task) -> dict:
        return {"input": task.text}

    def plan(self, task: Task, understanding: dict) -> dict:
        return {self.input_key: understanding["input"]}

    def execute(self, task: Task, plan: dict) -> dict:
        raw = self.runnable.invoke(plan)
        return {"raw": raw}

    def validate(self, task: Task, result: dict) -> dict:
        return result

    def respond(self, task: Task, validated: dict) -> AgentResult:
        raw = validated["raw"]
        if isinstance(raw, dict):
            text = raw.get(self.output_key) or raw.get("content") or str(raw)
        elif hasattr(raw, "content"):
            text = raw.content
        else:
            text = str(raw)
        return AgentResult(text=text, ok=True, confidence=0.7, raw={"langchain_output": str(raw)})
