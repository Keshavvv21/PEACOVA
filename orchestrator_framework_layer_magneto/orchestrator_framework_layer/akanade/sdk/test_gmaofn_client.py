"""Smoke tests for gmaofn_client — no live server needed: httpx's
MockTransport intercepts every request and returns a canned response,
so these verify the client builds the right method/path/body and
parses the response correctly, not network behavior.

Run with: pip install httpx pytest && pytest sdk/test_gmaofn_client.py
"""
import httpx
import pytest

from gmaofn_client import AKANADEClient, AKANADEError


def _client(handler) -> AKANADEClient:
    c = AKANADEClient("http://fake-host")
    c._http = httpx.Client(base_url="http://fake-host",
                           transport=httpx.MockTransport(handler))
    return c


def test_login_stores_the_token_and_returns_the_full_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/login"
        assert b"aniruddha" in request.content and b"demo123" in request.content
        return httpx.Response(200, json={"access_token": "tok123", "role": "Executive",
                                         "user_id": "u1", "name": "Aniruddha"})

    client = _client(handler)
    data = client.login("aniruddha", "demo123")
    assert data["access_token"] == "tok123"
    assert client._token == "tok123"


def test_send_message_includes_hybrid_provider_and_parses_new_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "reply": "It's overdue.", "pending_confirmation": False, "confirm_prompt": None,
            "session_id": "s1", "framework": "native", "out_of_scope": False,
            "provider": "ollama-qwen2.5:1.5b",
            "token_stats": {"tokens_before": 800, "tokens_after": 320, "tokens_saved_pct": 60.0},
            "sources_used": [{"index": 1, "score": 0.9, "metadata": {"title": "notes"}}],
            "tier_log": [{"node": "perceive_intent", "provider": "ollama-qwen2.5:1.5b"}],
        })

    client = _client(handler)
    client.set_token("tok123")
    reply = client.send_message("what is overdue?", provider="hybrid")

    assert seen["path"] == "/api/chat/message"
    assert seen["auth"] == "Bearer tok123"
    assert reply.provider == "ollama-qwen2.5:1.5b"
    assert reply.token_stats["tokens_saved_pct"] == 60.0
    assert reply.sources_used[0]["metadata"]["title"] == "notes"


def test_calling_an_authenticated_method_before_login_raises_clearly():
    client = _client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(AKANADEError) as exc_info:
        client.send_message("hi")
    assert exc_info.value.status_code == 401


def test_a_non_2xx_response_raises_gmaofn_error_with_status_and_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No RAG store registered as 'x'"})

    client = _client(handler)
    client.set_token("tok123")
    with pytest.raises(AKANADEError) as exc_info:
        client.rag_summarize("x")
    assert exc_info.value.status_code == 404
    assert "No RAG store registered" in str(exc_info.value.body)


def test_rag_ingest_path_hits_the_right_endpoint_with_the_right_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json={"store": "doc-notes-abcd1234",
                                         "files_indexed": 1, "chunks_indexed": 3,
                                         "skipped_files": []})

    client = _client(handler)
    client.set_token("tok123")
    result = client.rag_ingest_path("reports/notes.md")

    assert seen["path"] == "/api/v1/rag/ingest-path"
    assert b"reports/notes.md" in seen["body"]
    assert result["store"] == "doc-notes-abcd1234"


def test_admin_create_provider_defaults_tiers_when_not_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(201, json={"id": "p1", "name": "custom", "available": True})

    client = _client(handler)
    client.set_token("tok123")
    client.admin_create_provider(name="custom", base_url="http://x", model="m", api_key="k")
    assert b'"realtime"' in seen["body"] and b'"bulk"' in seen["body"]
