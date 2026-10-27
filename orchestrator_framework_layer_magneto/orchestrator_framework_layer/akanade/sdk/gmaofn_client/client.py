"""AKANADEClient — a thin httpx wrapper around a deployed ALGO AKANADE
chatbot's REST API. One class, one HTTP client, real endpoints only
(no guessing at paths — every method below matches an actual route in
akanade/backend/app/routers/).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

Framework = Literal["native", "langgraph", "adk"]


class AKANADEError(RuntimeError):
    """Raised for any non-2xx response — carries the real status code
    and response body so a caller can actually act on it, not just a
    generic message."""

    def __init__(self, status_code: int, body: Any):
        super().__init__(f"ALGO AKANADE API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


@dataclass
class ChatReply:
    reply: str
    pending_confirmation: bool
    confirm_prompt: str | None
    session_id: str
    framework: str
    out_of_scope: bool
    provider: str | None
    token_stats: dict | None
    sources_used: list[dict]
    tier_log: list[dict]
    # gmaofn/query_refinement.py's result for this turn — None when the
    # bandit gate chose "skip" (message was already clear).
    refinement: dict | None = None
    # The answering provider's own reported token usage for this turn —
    # {"prompt_tokens", "completion_tokens", "total_tokens"}, None for a
    # provider that doesn't report it (e.g. the local Ollama rung).
    usage: dict | None = None
    # True when an earlier provider in this turn's chain hit its own
    # context/output limit and the server escalated to a later one
    # instead of returning an incomplete answer — see llm/router.py's
    # `truncated` on the server side.
    context_limit_switch: bool = False
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def _from_json(cls, data: dict) -> "ChatReply":
        return cls(
            reply=data["reply"],
            pending_confirmation=data.get("pending_confirmation", False),
            confirm_prompt=data.get("confirm_prompt"),
            session_id=data["session_id"],
            framework=data["framework"],
            out_of_scope=data.get("out_of_scope", False),
            provider=data.get("provider"),
            token_stats=data.get("token_stats"),
            sources_used=data.get("sources_used", []),
            tier_log=data.get("tier_log", []),
            refinement=data.get("refinement"),
            usage=data.get("usage"),
            context_limit_switch=data.get("context_limit_switch", False),
            raw=data,
        )


class AKANADEClient:
    """Usage:

        client = AKANADEClient("https://your-deploy-host")
        client.login("aniruddha", "demo123")
        reply = client.send_message("what is overdue in Compass OS")
        print(reply.reply, reply.provider)

        # Hybrid mode — RL picks local SLM vs hosted model per message:
        client.send_message("summarize this quarter's velocity", provider="hybrid")

        # RAG: ingest a document (Admin role required server-side) and
        # summarize it:
        ingest = client.rag_ingest_path("reports/q3-notes.md")
        summary = client.rag_summarize(ingest["store"])
    """

    def __init__(self, base_url: str, *, timeout: float = 180.0):
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._token: str | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AKANADEClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        if not self._token:
            raise AKANADEError(401, "not logged in — call login() first")
        return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method: str, path: str, *, auth: bool = True, **kwargs: Any) -> Any:
        headers = kwargs.pop("headers", {})
        if auth:
            headers = {**headers, **self._headers()}
        resp = self._http.request(method, path, headers=headers, **kwargs)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise AKANADEError(resp.status_code, body)
        return resp.json() if resp.content else None

    # ── auth ──────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> dict:
        """POST /api/auth/login — stores the returned token for every
        subsequent call on this client instance. Returns
        {access_token, role, user_id, name}."""
        data = self._request("POST", "/api/auth/login", auth=False,
                             json={"username": username, "password": password})
        self._token = data["access_token"]
        return data

    def set_token(self, access_token: str) -> None:
        """Use an already-issued token instead of calling login()."""
        self._token = access_token

    # ── chat ──────────────────────────────────────────────────────────

    def send_message(self, message: str, *, session_id: str = "default",
                     framework: Framework | None = None,
                     provider: str | None = None) -> ChatReply:
        """POST /api/chat/message. `provider` may be a real provider id
        (see list_providers()), "hybrid" (RL picks local vs hosted per
        message), or omitted for Auto (the tier's normal fallback chain)."""
        params = {"framework": framework} if framework else {}
        body = {"message": message, "session_id": session_id}
        if provider:
            body["provider"] = provider
        data = self._request("POST", "/api/chat/message", params=params, json=body)
        return ChatReply._from_json(data)

    def list_providers(self) -> list[dict]:
        """GET /api/chat/providers — every provider in any tier's chain,
        with whether it's currently reachable. "hybrid" is a client-side
        sentinel (see send_message), not in this list."""
        return self._request("GET", "/api/chat/providers")["providers"]

    # ── RAG ───────────────────────────────────────────────────────────

    def rag_search(self, query: str, *, store: str = "default", k: int = 5) -> list[dict]:
        """POST /api/v1/rag/search."""
        return self._request("POST", "/api/v1/rag/search",
                             json={"store": store, "query": query, "k": k})["hits"]

    def rag_query(self, query: str, *, store: str = "default", k: int = 5,
                 max_context_chars: int = 4000) -> dict:
        """POST /api/v1/rag/query — search + rerank + build_context in one call."""
        return self._request("POST", "/api/v1/rag/query",
                             json={"store": store, "query": query, "k": k,
                                   "max_context_chars": max_context_chars})

    def rag_index_text(self, text: str, *, store: str = "default",
                       metadata: dict | None = None, chunk_size: int = 800) -> dict:
        """POST /api/v1/rag/index — index raw text (not a server file)."""
        return self._request("POST", "/api/v1/rag/index",
                             json={"store": store, "text": text,
                                   "metadata": metadata or {}, "chunk_size": chunk_size})

    def rag_ingest_path(self, path: str, *, store_name: str | None = None,
                        chunk_size: int = 800) -> dict:
        """POST /api/v1/rag/ingest-path — Admin role required server-side.
        `path` is relative to the server's RAG_INGEST_DIR, never an
        absolute path. Returns {"store": ..., "files_indexed": ...,
        "chunks_indexed": ..., "skipped_files": [...]}."""
        return self._request("POST", "/api/v1/rag/ingest-path",
                             json={"path": path, "store_name": store_name,
                                   "chunk_size": chunk_size})

    def rag_summarize(self, store: str, *, query: str = "summarize the main points",
                      k: int = 6, max_context_chars: int = 6000) -> dict:
        """POST /api/v1/rag/summarize. Returns {"summary", "sources_used", "provider"}."""
        return self._request("POST", "/api/v1/rag/summarize",
                             json={"store": store, "query": query, "k": k,
                                   "max_context_chars": max_context_chars})

    def rag_suggest_prompts(self, store: str, *, query: str = "what is this document about",
                           n: int = 3, k: int = 6, max_context_chars: int = 6000) -> dict:
        """POST /api/v1/rag/suggest-prompts. Returns {"prompts", "sources_used", "provider"}."""
        return self._request("POST", "/api/v1/rag/suggest-prompts",
                             json={"store": store, "query": query, "n": n, "k": k,
                                   "max_context_chars": max_context_chars})

    # ── admin: LLM providers (Executive role required server-side) ──────

    def admin_list_providers(self) -> list[dict]:
        return self._request("GET", "/api/admin/providers")

    def admin_create_provider(self, *, name: str, base_url: str, model: str,
                              api_key: str, tiers: list[str] | None = None) -> dict:
        return self._request("POST", "/api/admin/providers",
                             json={"name": name, "base_url": base_url, "model": model,
                                   "api_key": api_key, "tiers": tiers or ["realtime", "bulk"]})

    def admin_delete_provider(self, provider_id: str) -> None:
        self._request("DELETE", f"/api/admin/providers/{provider_id}")
