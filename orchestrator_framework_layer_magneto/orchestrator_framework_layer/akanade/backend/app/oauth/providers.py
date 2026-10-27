"""Static per-provider OAuth 2.0 registry — the config-driven-registry
shape config.py's EXTRA_LLM_PROVIDERS/MCP_SERVERS already use, just not
env-JSON since there are exactly three fixed providers rather than an
open-ended list an operator adds to.

GitHub and Linear are real, standard OAuth 2.0 Authorization Code flows
against their documented endpoints, verified against each provider's own
current developer docs (not assumed from training data). Google's OAuth
endpoints below are Google's own standard, well-documented ones too —
what's genuinely unverified is which real, current, actively-maintained
Gmail/Calendar MCP *server* to point at once a token exists (there is no
one official "Google MCP server" the way GitHub/Linear each have exactly
one) — see user_mcp_client.py's module docstring for that gap; `mcp_url`
is left None for google until that's resolved, which the discovery layer
already treats as "not wired yet," same as any other unconfigured rung.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings


@dataclass(frozen=True)
class OAuthProvider:
    name: str
    authorize_url: str
    token_url: str
    scopes: list[str]
    client_id: str
    client_secret: str
    # Extra query params folded into the authorize URL beyond the
    # standard response_type/client_id/redirect_uri/scope/state — e.g.
    # Google's access_type=offline+prompt=consent, needed to actually get
    # a refresh_token back (without them Google only returns one the
    # very first time an account ever grants consent).
    extra_authorize_params: dict[str, str] = field(default_factory=dict)
    # None = OAuth wiring exists but no MCP server is set up for it yet
    # (see module docstring re: Google) — discovery just yields no tools,
    # same as a misconfigured MCP_SERVERS entry, never a hard error.
    mcp_url: str | None = None
    # Real revoke endpoint semantics differ enough per provider (GitHub's
    # is a DELETE to a REST path with Basic auth, not a token-revoke POST;
    # Linear publishes no public revoke endpoint at all) that oauth_store's
    # revoke_connection() special-cases each provider by name rather than
    # this registry pretending they share one shape. This field exists
    # only for the one provider (Google) with a real, standard
    # POST-token-to-revoke endpoint.
    revoke_url: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


def _github() -> OAuthProvider:
    return OAuthProvider(
        name="github",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        # "repo" is the write-capable scope (open issues/PRs, push) —
        # GitHub has no separate read-only equivalent for private repos,
        # same tradeoff every "connect your GitHub" integration makes.
        scopes=["repo"],
        client_id=settings().GITHUB_OAUTH_CLIENT_ID,
        client_secret=settings().GITHUB_OAUTH_CLIENT_SECRET,
        mcp_url="https://api.githubcopilot.com/mcp/",
    )


def _linear() -> OAuthProvider:
    return OAuthProvider(
        name="linear",
        authorize_url="https://linear.app/oauth/authorize",
        token_url="https://api.linear.app/oauth/token",
        scopes=["read", "write"],
        client_id=settings().LINEAR_OAUTH_CLIENT_ID,
        client_secret=settings().LINEAR_OAUTH_CLIENT_SECRET,
        mcp_url="https://mcp.linear.app/sse",
    )


def _google() -> OAuthProvider:
    return OAuthProvider(
        name="google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        # Narrowest real scope per capability (verified against Google's
        # own scope docs, not guessed) rather than the broad gmail.modify/
        # calendar this used to request:
        #   gmail.readonly — read/summarize (agent/google_tools.py's
        #     read_emails)
        #   gmail.compose  — "Manage drafts and send emails" (Google's own
        #     description) — covers BOTH draft_email and send_email, no
        #     separate gmail.send needed
        #   calendar.events — view + create/update events on calendars the
        #     user can already access (narrower than the bare "calendar"
        #     scope, which also allows deleting whole calendars)
        #   chat.messages.create — "Compose and send messages" (send_chat_message)
        #   chat.spaces.readonly — "View chat and spaces", needed for
        #     spaces:findDirectMessage to resolve a DM space before
        #     posting to it (see google_api.send_chat_message)
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/chat.messages.create",
            "https://www.googleapis.com/auth/chat.spaces.readonly",
        ],
        client_id=settings().GOOGLE_OAUTH_CLIENT_ID,
        client_secret=settings().GOOGLE_OAUTH_CLIENT_SECRET,
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        revoke_url="https://oauth2.googleapis.com/revoke",
        # Still None — these are hand-written tools calling Gmail/Calendar/
        # Chat's real REST APIs directly (agent/google_tools.py,
        # services/google_api.py), not an MCP server. No official single
        # "Google MCP server" exists for these three products together, so
        # this deliberately isn't routed through mcp_client.py/
        # user_mcp_client.py at all — see google_tools.py's own docstring.
        mcp_url=None,
    )


def all_providers() -> dict[str, OAuthProvider]:
    """Rebuilt on every call (cheap: 3 dataclasses) rather than cached at
    import time, so a `settings()` change (e.g. a test's monkeypatch, or
    hot-reloading .env in dev) is reflected immediately — same reasoning
    admin_providers.py's DB-backed rows get re-read on every request."""
    return {"github": _github(), "linear": _linear(), "google": _google()}


def get_provider(name: str) -> OAuthProvider:
    providers = all_providers()
    if name not in providers:
        raise ValueError(f"unknown OAuth provider {name!r} — expected one of {list(providers)}")
    return providers[name]
