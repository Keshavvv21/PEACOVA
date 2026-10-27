"""§2.2 — the Framework Adapter: get_agent(framework) → AgentHandle with
one method. The router, ChatSessions, and the widget never know which
framework answered; the active one is config.CHAT_AGENT_FRAMEWORK, with
a per-request ?framework= override for the bake-off (§10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..config import settings
from .state import AgentReply


@dataclass
class AgentHandle:
    name: str
    _fn: Callable[..., AgentReply]

    def handle_message(self, user: dict, session_id: str, user_msg: str,
                       preferred_provider: str | None = None,
                       target: str | None = None) -> AgentReply:
        return self._fn(user, session_id, user_msg, framework_label=self.name,
                        preferred_provider=preferred_provider, target=target)


def get_agent(framework: str | None = None) -> AgentHandle:
    fw = (framework or settings().CHAT_AGENT_FRAMEWORK).lower()
    if fw == "langgraph":
        from . import langgraph_agent
        return AgentHandle("langgraph", langgraph_agent.handle_message)
    if fw == "adk":
        try:
            from . import adk_agent  # optional dependency (google-adk)
            return AgentHandle("adk", adk_agent.handle_message)
        except ImportError as e:
            raise RuntimeError(
                "ADK track requested but google-adk is not installed "
                "(pip install google-adk litellm)") from e
    if fw == "claude":
        from . import claude_agent
        return AgentHandle("claude", claude_agent.handle_message)
    from . import native_agent
    return AgentHandle("native", native_agent.handle_message)
