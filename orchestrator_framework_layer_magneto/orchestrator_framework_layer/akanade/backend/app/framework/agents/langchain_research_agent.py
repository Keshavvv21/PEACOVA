"""LangChainResearchAgent — a real, working example of §8's LangChain
integration: an actual `langchain_core` `Runnable` pipeline (a
`ChatPromptTemplate` piped into a `RunnableLambda`), wrapped as a ALGO AKANADE
agent via `langchain_adapter.LangChainRunnableAgent` — not just an
adapter module sitting unused. Also builds a real
`langchain_core.retrievers.BaseRetriever` over ALGO AKANADE's in-memory RAG
store (rag.py) via `vector_store_to_retriever`, so the "RAG" half of
§8/§9's LangChain+RAG integration is exercised too: retrieved context is
injected into the prompt before the model ever sees the question.

The `RunnableLambda` bridges to `llm/router.py`'s own provider chain
rather than a LangChain chat-model integration (`langchain-openai`,
`langchain-anthropic`, ...) — this keeps the example dependency-light
(no extra API key needed beyond what's already configured) while still
being a genuine, composable LangChain `Runnable`: real
`ChatPromptTemplate.from_messages()`, real `|` piping, real
`.invoke()`. Swap the `RunnableLambda` for `ChatOpenAI`/`ChatAnthropic`/
`ChatOllama` (LangChain's own model wrappers) in a real deployment
without changing anything about how this agent is registered or routed.
"""
from __future__ import annotations

from ...framework.rag import InMemoryVectorStoreAdapter, RagDocument
from ...llm.router import get_llm_client
from ..agent_base import AgentMetadata, AgentStatus, LatencyProfile
from ..langchain_adapter import LangChainRunnableAgent, vector_store_to_retriever
from ..task import TaskType

# A tiny, self-contained knowledge base — enough to prove retrieval
# actually changes the answer, without depending on any external corpus.
_KNOWLEDGE_BASE = [
    "ALGO AKANADE routes tasks to agents using a linear contextual Thompson Sampling bandit.",
    "ALGO AKANADE splits credit across multiple agents on one task using the Shapley value, "
    "a 2012 Nobel Memorial Prize in Economic Sciences concept from cooperative game theory.",
    "ALGO AKANADE's reward function is R = alpha*Quality + beta*Success + gamma*Feedback "
    "- delta*Cost - epsilon*Latency - zeta*Failure, with adaptively-updated coefficients.",
]


def _build_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    store = InMemoryVectorStoreAdapter()
    store.add_documents([RagDocument(text=t) for t in _KNOWLEDGE_BASE])
    retriever = vector_store_to_retriever(store, k=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the given context. If the context doesn't "
                  "cover the question, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ])

    def _call_model(messages) -> dict:
        chat_messages = [{"role": "system" if m.type == "system" else "user", "content": m.content}
                        for m in messages.to_messages()]
        result = get_llm_client("realtime").chat("langchain_research_respond", chat_messages)
        return {"output": result.text if not result.error else f"Sorry — {result.error}"}

    def _with_context(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["input"])
        return {"input": inputs["input"], "context": "\n".join(d.page_content for d in docs)}

    return RunnableLambda(_with_context) | prompt | RunnableLambda(_call_model)


def build_langchain_research_agent() -> LangChainRunnableAgent:
    metadata = AgentMetadata(
        agent_id="langchain-research-agent",
        name="LangChain Research Agent",
        description="A real langchain_core Runnable (ChatPromptTemplate | RunnableLambda) "
                   "wrapped as a ALGO AKANADE agent, with a real BaseRetriever over an "
                   "in-memory RAG store injecting context before the model answers.",
        capabilities=frozenset({"research", "question_answering"}),
        supported_tasks=frozenset({TaskType.RESEARCH, TaskType.QUESTION_ANSWERING}),
        model="configurable (via llm/router.py's provider chain, bridged into LangChain)",
        rag_enabled=True,
        cost_per_call_usd=0.002,
        latency_profile=LatencyProfile(p50_ms=700, p95_ms=2500),
        version="1.0.0",
        reliability_score=0.7,
        status=AgentStatus.ACTIVE,
    )
    return LangChainRunnableAgent(metadata, _build_chain(), input_key="input", output_key="output")
