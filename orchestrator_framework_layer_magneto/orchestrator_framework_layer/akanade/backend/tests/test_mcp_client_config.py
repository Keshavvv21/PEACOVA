"""app/agent/mcp_client.py's config parsing — specifically the `headers`
field added so a remote (url-based) MCP server that accepts a static
bearer token/API key can be configured without a full OAuth flow (a
genuinely different, bigger feature this does NOT implement — no
browser redirect/callback, no per-user token storage/refresh).
"""
from __future__ import annotations

import json

from app.agent import mcp_client


def test_configured_servers_parses_headers_for_a_url_server(monkeypatch):
    entries = [{"name": "notion", "url": "https://mcp.example.com/sse",
               "headers": {"Authorization": "Bearer test-token-123"}}]
    monkeypatch.setattr(mcp_client.settings(), "MCP_SERVERS", json.dumps(entries))
    servers = mcp_client._configured_servers()
    assert len(servers) == 1
    assert servers[0].headers == {"Authorization": "Bearer test-token-123"}
    assert servers[0].url == "https://mcp.example.com/sse"


def test_configured_servers_headers_default_to_none_when_omitted(monkeypatch):
    entries = [{"name": "public", "url": "https://mcp.example.com/sse"}]
    monkeypatch.setattr(mcp_client.settings(), "MCP_SERVERS", json.dumps(entries))
    servers = mcp_client._configured_servers()
    assert servers[0].headers is None


def test_configured_servers_headers_ignored_for_a_stdio_server(monkeypatch):
    """`headers` only makes sense for a `url` (remote) server — a stdio
    subprocess server just gets it parsed through harmlessly (stdio_client
    never reads it), consistent with `env` being the credential channel
    for that transport instead."""
    entries = [{"name": "utility", "command": "python", "args": ["-m", "app.mcp_servers.utility_server"]}]
    monkeypatch.setattr(mcp_client.settings(), "MCP_SERVERS", json.dumps(entries))
    servers = mcp_client._configured_servers()
    assert servers[0].headers is None
    assert servers[0].command == mcp_client.sys.executable
