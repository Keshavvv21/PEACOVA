"""A small, safe, fully offline example MCP server — the bundled default
for config.py's MCP_SERVERS, so the chatbot's MCP-client integration
(agent/mcp_client.py) has something real to talk to with zero external
network access or operator setup required.

Every tool here is read-only, deterministic (given its inputs), and touches
nothing but the local clock/interpreter — no filesystem, network, secrets,
or PM data. Swap MCP_SERVERS to point at a real external MCP server
(filesystem, GitHub, Slack, ...) instead of/alongside this one; this file
exists purely to prove the integration end-to-end out of the box.

Run standalone for a manual smoke test: `python -m app.mcp_servers.utility_server`
(speaks MCP over stdio — same way agent/mcp_client.py launches it, and the
same way Claude Desktop/Claude Code launch any local MCP server).
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

# FastMCP (not the mcp>=2's mcp.server.mcpserver.MCPServer — same @mcp.tool()/
# .run() surface, but requirements.txt pins mcp<2 because mcp-server-fetch,
# the OTHER MCP_SERVERS default, hard-requires it) is this package's 1.x
# high-level server API.
mcp = FastMCP("algo-kaizen-utility")


@mcp.tool()
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time in a given IANA timezone (e.g.
    "UTC", "Asia/Kolkata", "America/New_York"). Defaults to UTC.
    Useful for scheduling questions like "what time is it for the
    standup in Tokyo right now"."""
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return (f"Unknown timezone {timezone!r} — use an IANA name like "
                f"'UTC', 'Asia/Kolkata', or 'America/New_York'.")
    now = datetime.now(zone)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


@mcp.tool()
def days_between(start_date: str, end_date: str) -> str:
    """Number of calendar days between two ISO dates (YYYY-MM-DD).
    Useful for quick duration math the PM tools don't already cover,
    e.g. "how many days between 2026-08-01 and 2026-09-15"."""
    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
    except ValueError:
        return "Both dates must be in YYYY-MM-DD format."
    delta = (end - start).days
    return f"{abs(delta)} day(s) {'apart' if delta >= 0 else '(end_date is before start_date)'}"


if __name__ == "__main__":
    mcp.run()
