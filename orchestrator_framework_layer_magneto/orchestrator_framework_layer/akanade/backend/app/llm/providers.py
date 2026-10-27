"""LLM providers behind one interface.

Every provider implements chat(messages, tools) -> LLMResult, where the
result is either a tool call {name, arguments} or plain text. Providers:

  OpenAICompatible — Luna, Max, Deepseek V4 Flash, Groq, the future Algo
      Engine local tier, AND any operator-added EXTRA_LLM_PROVIDERS entry
      (see llm/router.py) are all OpenAI-compatible endpoints; one class,
      arbitrarily many configs (§3.2 + §19 of the design, plus the
      config-driven registry that extends it).
  LangChainOllama — a local small language model via Ollama, through
      LangChain's ChatOllama/.bind_tools()/.invoke() — a genuinely
      different client library, not another OpenAICompatible wearing a
      different name. The "add an SLM" tier; optional dependency.
  RuleBased — a deterministic, offline intent parser implementing the
      same interface. It is the bottom rung of every fallback chain, so
      the entire stack runs with zero API keys — and it doubles as the
      design's shadow-comparison baseline.
  AnthropicNative — the real Anthropic Claude API (the `anthropic`
      package's Messages API), for the "claude" agent framework
      (agent/claude_agent.py). Genuinely different wire format from every
      OpenAICompatible rung above — Claude's own tool_use/tool_result
      content-block shape, not OpenAI's tool_calls — so like
      LangChainOllama this is its own client, not OpenAICompatible again.
  HuggingFaceProvider — ALGO AKANADE's generic model layer's Hugging Face
      rung: the real `huggingface_hub.InferenceClient`, so ANY model
      hosted on HF Inference Providers (or a dedicated HF Inference
      Endpoint via `endpoint_url`) is addable through config alone —
      see config.py's HUGGINGFACE_MODEL/HUGGINGFACE_API_KEY.
  CustomProvider — a thin, explicitly-named alias of OpenAICompatible for
      "any custom or locally-hosted model that speaks the OpenAI chat-
      completions wire format" (vLLM, llama.cpp's server, LM Studio, a
      hand-rolled inference API, ...) — ALGO AKANADE's spec calls this rung out
      by name (ModelProvider.CustomProvider) even though, mechanically,
      it's the exact same OpenAI-compatible contract Luna/Max/Groq/
      already use; no new wire-format code needed for it.

The design's lint rule (§16): get_llm_client() in router.py is the ONLY
sanctioned entry to any of these — no provider import elsewhere.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import httpx

from ..plugins import PLUGINS
from .optimization import NONE_DECLARED, OptimizationProfile


@dataclass
class LLMResult:
    text: str = ""
    tool_call: Optional[dict] = None          # {"name":..., "arguments":{...}}
    provider: str = ""
    error: Optional[str] = None
    # Real token counts from the provider's own response, when it reports
    # them — {"prompt_tokens", "completion_tokens", "total_tokens"}, the
    # OpenAI-shaped names every provider below normalizes to regardless of
    # its own wire format (Anthropic's input_tokens/output_tokens included).
    # None for a provider that doesn't report usage (e.g. LangChainOllama).
    usage: Optional[dict] = None
    # True when the provider's OWN signal says the reply was cut off by
    # hitting its context/output limit (OpenAI-shape finish_reason=="length",
    # Anthropic stop_reason=="max_tokens") — a "successful" HTTP call that
    # still isn't a complete answer. TierClient.chat() treats this as a
    # reason to escalate to the next rung, same as a hard failure, but
    # logged distinctly so the UI can say WHY the model changed.
    truncated: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    """The one contract every provider — OpenAICompatible, LangChainOllama,
    RuleBased, and anything EXTRA_LLM_PROVIDERS constructs — must satisfy to
    be substitutable anywhere router.TierClient expects "a provider" (LSP,
    made structural/checkable rather than just an informal convention).
    Deliberately three members, nothing a caller doesn't need (ISP): no
    provider is forced to carry config or methods only some tiers use.
    """
    name: str

    def available(self) -> bool:
        """Cheap, local, no network call — can this rung even be tried?"""
        ...

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        """One turn. Raise on failure (network/5xx/timeout) rather than
        returning a sentinel — TierClient's chain walk relies on the
        exception to know to try the next rung (§6.1 step 12)."""
        ...


@dataclass
class LangChainOllama:
    """A local small language model via Ollama, through LangChain's
    ChatOllama — genuinely offline inference (no API key, runs on a laptop),
    the "add an SLM" tier. Real LangChain, not another OpenAICompatible:
    uses .bind_tools()/.invoke() and reads .tool_calls off the response,
    LangChain's own function-calling contract, so this is a legitimate
    second client library in the stack, not a relabeled HTTP POST.

    available() gates on both OLLAMA_BASE_URL being set AND langchain-ollama
    being installed (it's an optional dependency, see requirements.txt) — if
    either is missing this rung is just unavailable, same as any hosted tier
    with no key configured; the chain walks past it to rule-based.
    """
    name: str
    base_url: str
    model: str
    timeout: float = 30.0
    # Real, observed problem this fixes: Ollama's own default keep_alive
    # is 5 MINUTES — it unloads the model from memory after that much
    # idle time, and the next request pays a real ~10-15s reload cost
    # (confirmed live: a cold first call took 13.8s, the very next call
    # milliseconds later — model already warm — took 0.19s for the
    # IDENTICAL request). Since CodingAgent tries "ollama-coder" FIRST by
    # default (see coding_agent.py's _preferred_provider), and normal
    # usage has gaps well over 5 minutes (reading a reply, switching
    # tasks), this reload was firing on most real VS Code/mobile turns,
    # not just the first one ever — exactly what read as "the API is
    # slow" across both surfaces. Passed straight through to Ollama's own
    # API (LangChain's ChatOllama accepts the same keep_alive shape
    # Ollama's /api/chat does — a duration string or -1 for "never
    # unload") on EVERY call, not just once, since Ollama treats each
    # request's keep_alive as resetting that request's own idle timer.
    keep_alive: str = "30m"

    def available(self) -> bool:
        if not self.base_url:
            return False
        try:
            import langchain_ollama  # noqa: F401
        except ImportError:
            return False
        return True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        llm = ChatOllama(base_url=self.base_url, model=self.model,
                         timeout=self.timeout, keep_alive=self.keep_alive)
        if tools:
            llm = llm.bind_tools(tools)

        role_to_cls = {"system": SystemMessage, "assistant": AIMessage}
        lc_messages = [role_to_cls.get(m["role"], HumanMessage)(content=m["content"])
                      for m in messages]

        result = llm.invoke(lc_messages)
        if getattr(result, "tool_calls", None):
            tc = result.tool_calls[0]
            return LLMResult(provider=self.name,
                             tool_call={"name": tc["name"], "arguments": tc["args"]})
        return LLMResult(provider=self.name, text=result.content or "")


@dataclass
class OpenAICompatible:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0
    # True (default) for any hosted API that needs auth (Luna, Max, Deepseek,
    # Groq, an EXTRA_LLM_PROVIDERS entry like Gemini) — available() then
    # requires BOTH base_url and api_key, so a configured-but-keyless entry
    # is skipped rather than firing a doomed unauthenticated request on
    # every single turn. False for a self-hosted, unauthenticated rung
    # (, the future Algo Engine tier) where base_url alone is enough.
    requires_key: bool = True
    # Real pass-through, not description: merged verbatim into every
    # request payload — the one thing this PROCESS can actually hand to a
    # self-hosted server about how it should run a request (e.g. vLLM's
    # extra sampling params, a server-specific speculative-decoding
    # knob). See optimization.py's module docstring — everything else
    # about "model optimization strategies" is that server's own
    # deployment config, not a per-request flag.
    extra_body: dict = field(default_factory=dict)
    # Declarative only — what this endpoint's backend IS deployed with,
    # for GET /api/v1/models to report; never consulted to change how a
    # request is made (see optimization.py).
    optimization: OptimizationProfile = field(default_factory=lambda: NONE_DECLARED)

    def available(self) -> bool:
        if not self.base_url:
            return False
        return bool(self.api_key) if self.requires_key else True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        payload: dict = {"model": self.model, "messages": messages, **self.extra_body}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                       headers=headers, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        choice = body["choices"][0]
        msg = choice["message"]
        usage = body.get("usage")
        truncated = choice.get("finish_reason") == "length"
        if msg.get("tool_calls"):
            fn = msg["tool_calls"][0]["function"]
            return LLMResult(provider=self.name, tool_call={
                "name": fn["name"], "arguments": json.loads(fn["arguments"])},
                usage=usage, truncated=truncated)
        return LLMResult(provider=self.name, text=msg.get("content") or "",
                         usage=usage, truncated=truncated)


@dataclass
class AnthropicNative:
    """The real Anthropic Messages API — Claude's own tool_use/tool_result
    content-block format, not OpenAI's tool_calls, so this talks to the
    `anthropic` SDK directly rather than going through OpenAICompatible.

    available() gates on both an API key being configured AND the
    `anthropic` package being importable (an optional dependency, see
    requirements.txt) — same "just another unavailable rung" story as
    Ollama/LangChainOllama above if either is missing.
    """

    name: str
    api_key: str
    model: str
    max_tokens: int = 1024
    timeout: float = 30.0

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

        # Claude takes system instructions as a top-level `system` string,
        # never as a "system"-role turn in `messages` (the API rejects
        # that role outright) — every other provider here is OpenAI-shaped
        # and happily accepts inline system messages, so this translation
        # is specific to this one provider, not something router.py or
        # any caller needs to know about.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_parts) or None
        claude_messages = [{"role": m["role"], "content": m["content"]}
                          for m in messages if m["role"] in ("user", "assistant")]

        claude_tools = [{"name": t["name"], "description": t["description"],
                        "input_schema": t["parameters"]} for t in (tools or [])]

        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens,
                        "messages": claude_messages}
        if system:
            kwargs["system"] = system
        if claude_tools:
            kwargs["tools"] = claude_tools

        response = client.messages.create(**kwargs)
        # Normalized to the same {"prompt_tokens", "completion_tokens",
        # "total_tokens"} shape OpenAICompatible reports — Anthropic's own
        # names (input_tokens/output_tokens) are a provider-wire detail no
        # caller further up should need to know about.
        usage = ({"prompt_tokens": response.usage.input_tokens,
                 "completion_tokens": response.usage.output_tokens,
                 "total_tokens": response.usage.input_tokens + response.usage.output_tokens}
                if getattr(response, "usage", None) else None)
        truncated = response.stop_reason == "max_tokens"
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use:
            return LLMResult(provider=self.name,
                             tool_call={"name": tool_use.name, "arguments": dict(tool_use.input)},
                             usage=usage, truncated=truncated)
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResult(provider=self.name, text=text, usage=usage, truncated=truncated)


@dataclass
class CustomProvider(OpenAICompatible):
    """Named alias of OpenAICompatible — ALGO AKANADE's generic model layer
    calls this rung out explicitly (any custom/local model that speaks
    the OpenAI chat-completions wire format: vLLM, llama.cpp's server,
    LM Studio, a hand-rolled inference API, ...). Identical behavior to
    OpenAICompatible; this subclass exists purely so config/docs/API
    responses can say "custom" instead of the more generic base class
    name, matching the spec's own ModelProvider.CustomProvider."""


@dataclass
class HuggingFaceProvider:
    """Hugging Face Inference Providers (or a dedicated HF Inference
    Endpoint, via `endpoint_url`) through the real `huggingface_hub`
    SDK's `InferenceClient.chat_completion` — an OpenAI-shaped
    chat-completions call under the hood, but through HF's own client
    library (auth, retries, provider routing) rather than a raw httpx
    POST, the same "genuinely different client, not another
    OpenAICompatible" rationale LangChainOllama already follows here.

    available() gates on both an API token being configured AND
    `huggingface_hub` being importable — optional dependency, same
    "just another unavailable rung" story as Ollama/Claude above."""

    name: str
    api_key: str
    model: str
    endpoint_url: str = ""     # set for a dedicated Inference Endpoint instead of the shared API
    provider: str = "auto"     # HF's own inference-provider routing (auto|together|fireworks-ai|...)
    timeout: float = 30.0
    # Real pass-through — `InferenceClient.chat_completion`'s own
    # `extra_body` kwarg, merged into the actual request HF's client
    # sends. Same rationale as OpenAICompatible.extra_body: the one
    # thing this process can hand a self-hosted-behind-HF server about
    # how to run a request; see optimization.py.
    extra_body: dict = field(default_factory=dict)
    optimization: OptimizationProfile = field(default_factory=lambda: NONE_DECLARED)

    def available(self) -> bool:
        if not self.api_key or not self.model:
            return False
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return False
        return True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self.endpoint_url or self.model, token=self.api_key,
                                 provider=None if self.endpoint_url else self.provider,
                                 timeout=self.timeout)
        hf_tools = [{"type": "function", "function": t} for t in tools] if tools else None
        response = client.chat_completion(messages=messages, model=self.model,
                                          tools=hf_tools, max_tokens=1024,
                                          extra_body=self.extra_body or None)
        choice = response.choices[0]
        message = choice.message
        raw_usage = getattr(response, "usage", None)
        usage = ({"prompt_tokens": raw_usage.prompt_tokens,
                 "completion_tokens": raw_usage.completion_tokens,
                 "total_tokens": raw_usage.total_tokens} if raw_usage else None)
        truncated = getattr(choice, "finish_reason", None) == "length"
        if getattr(message, "tool_calls", None):
            call = message.tool_calls[0]
            args = call.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            return LLMResult(provider=self.name, tool_call={"name": call.function.name, "arguments": args},
                             usage=usage, truncated=truncated)
        return LLMResult(provider=self.name, text=message.content or "", usage=usage, truncated=truncated)


@dataclass
class RuleBased:
    """Deterministic intent parser — the offline floor of every chain.

    Speaks the same 'messages + tool schema in, tool call or text out'
    contract as a hosted model, using patterns over the user's message.
    Not smart; predictably correct on the phrasings the tests pin.
    """

    name: str = "rule-based"
    patterns: list = field(default_factory=list)

    def available(self) -> bool:
        return True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        user = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")
        low = user.lower()

        def q(name):  # quoted 'x' / "x" or capitalized tail capture helper
            m = re.search(r"['\"]([^'\"]+)['\"]", user)
            return m.group(1) if m else None

        # A trailing "with email x@y" clause is stripped up front (not
        # threaded through each create_user regex's own $ anchor) so it
        # composes with all three create_user phrasings below at once.
        email = None
        if em := re.search(r"\bwith email\s+(\S+@\S+)\s*$", user, re.I):
            email = em.group(1)
            user = user[:em.start()].rstrip()
            low = user.lower()

        def with_email(args: dict) -> dict:
            return {**args, "email": email} if email else args

        # ordered rules: first match wins. Verb alternations are picked
        # per-tool to avoid shadowing an earlier pattern — notably "add" is
        # never used as a create_task/create_project/create_space synonym
        # since add_member's own catch-all ("add X to Y", below) would
        # intercept it first; same reason create_sprint never accepts
        # "start" (start_sprint owns that verb).
        if m := re.search(r"(?:create|make|set up|start) (?:a |an )?(?:new )?space "
                          r"(?:called |named |titled )?(.+?)"
                          r"(?:\s+with\s+(.+?)\s+as\s+(?:the\s+)?(?:senior\s+)?manager)?$", low):
            args = {"name": q(user) or m.group(1).strip()}
            if m.group(2):
                args["manager"] = m.group(2).strip()
            return self._call("create_space", args)
        if m := re.search(r"(?:create|make|set up|provision) (?:a |an )?(?:new )?user "
                          r"(?:named |called )?"
                          r"(.+?) (?:as|with role) (?:an? )?"
                          r"(member|manager|admin|executive)\s*$", low):
            return self._call("create_user", with_email({
                "name": (q(user) or m.group(1)).strip(), "role": m.group(2)}))
        if m := re.search(r"(?:create|make|set up|provision) (?:a |an )?(?:new )?user "
                          r"(member|manager|admin|executive)"
                          r"(?: (?:named|called) (.+))?\s*$", low):
            return self._call("create_user", with_email({
                "name": (q(user) or m.group(2) or "").strip(), "role": m.group(1)}))
        if m := re.search(r"(?:create|make|set up|provision) (?:a |an )?"
                          r"(member|manager|admin|executive) "
                          r"(?:user|account)(?: (?:named|called) (.+))?\s*$", low):
            return self._call("create_user", with_email({
                "name": (q(user) or m.group(2) or "").strip(), "role": m.group(1)}))
        if m := re.search(r"(?:add|create|make) (?:a )?subtask "
                          r"(?:called |named )?(.+?) to (?:task )?(.+)", low):
            return self._call("create_subtask", {
                "title": (q(user) or m.group(1)).strip(), "task": m.group(2).strip()})
        # Checked BEFORE the generic "mark task ... as done" pattern below
        # — that one's own permissive (.+?) task-name capture would
        # otherwise swallow "subtask 'x' on 'y'" whole as one task title
        # instead of recognizing it as a subtask on a task.
        if m := re.search(r"(?:mark|complete|check off|finish) (?:the )?subtask "
                          r"(?:called |named )?(.+?) (?:on|under|in|for) (?:task )?"
                          r"(.+?)(?:\s+as\s+(done|not done|undone))?$", low):
            done = m.group(3) not in ("not done", "undone") if m.group(3) else True
            return self._call("complete_subtask", {
                "subtask": (q(user) or m.group(1)).strip(),
                "task": m.group(2).strip(), "done": done})
        if m := re.search(r"(?:give me |show me |show |list |what are )?"
                          r"(?:the )?subtasks (?:of|for|in|under) (?:task )?(.+)", low):
            return self._call("list_subtasks", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"(?:add|make) (.+?) (?:as |a )?"
                          r"(owner|co[- ]?owner|manager|member) "
                          r"(?:to|of|for|on) (?:project )?(.+)", low):
            return self._call("add_member", {
                "user": (q(user) or m.group(1)).strip(),
                "role": m.group(2).replace("-", "").replace(" ", ""),
                "project": m.group(3).strip()})
        if m := re.search(r"add (.+?) to (?:project )?(.+?) as (?:an? )?"
                          r"(owner|co[- ]?owner|manager|member)", low):
            return self._call("add_member", {
                "user": (q(user) or m.group(1)).strip(),
                "project": m.group(2).strip(),
                "role": m.group(3).replace("-", "").replace(" ", "")})
        if m := re.search(r"add (.+?) to (?:project )?(.+)", low):
            return self._call("add_member", {
                "user": (q(user) or m.group(1)).strip(), "project": m.group(2).strip()})
        if m := re.search(r"(?:create|make|set up) (?:a )?(kanban|scrum)? ?project "
                          r"(?:called |named )?(.+?) in (?:space )?(.+?)"
                          r"(?:\s+with\s+(.+?)\s+as\s+(?:the\s+)?(?:senior\s+)?manager)?"
                          r"(?:\s+and\s+(.+?)\s+as\s+(?:the\s+)?team\s+lead)?$", low):
            args = {"name": (q(user) or m.group(2)).strip(),
                    "space": m.group(3).strip(),
                    "board": (m.group(1) or "kanban").capitalize()}
            if m.group(4):
                args["manager"] = m.group(4).strip()
            if m.group(5):
                args["team_lead"] = m.group(5).strip()
            return self._call("create_project", args)
        if m := re.search(r"(?:create|make|new) (?:a )?task (?:called |named )?"
                          r"(.+?) in (?:project )?(.+?)(?:\s+with due date\s+"
                          r"([\d./-]+))?$", low):
            args = {"title": (q(user) or m.group(1)).strip(),
                    "project": m.group(2).strip()}
            if m.group(3):
                args["due_date"] = m.group(3)
            return self._call("create_task", args)
        if m := re.search(r"(?:assign|give|hand) (?:task )?(.+?) to (\w+)", low):
            return self._call("assign_task", {
                "task": (q(user) or m.group(1)).strip(), "assignee": m.group(2)})
        if m := re.search(r"(?:set|change|update) (?:the )?due date (?:of|for) "
                          r"(?:task )?(.+?) to ([\d-]+)", low):
            return self._call("set_due_date", {
                "task": (q(user) or m.group(1)).strip(), "due_date": m.group(2)})
        if m := re.search(r"(?:mark|move|change|set) (?:task )?(.+?) (?:to|as) "
                          r"(done|in ?progress|todo|backlog)$", low):
            status = {"done": "Done", "inprogress": "InProgress",
                      "in progress": "InProgress", "todo": "Todo",
                      "backlog": "Backlog"}[m.group(2)]
            return self._call("change_task_status", {
                "task": (q(user) or m.group(1)).strip(), "status": status})
        if m := re.search(r"(?:move|put) (?:task )?(.+?) (?:to|in|into) (sprint .+)", low):
            return self._call("move_task_to_sprint", {
                "task": (q(user) or m.group(1)).strip(),
                "sprint": m.group(2).strip()})
        if m := re.search(r"(?:create|make|set up) (?:a )?sprint (?:called |named )?"
                          r"(.+?) in (?:project )?(.+)", low):
            return self._call("create_sprint", {
                "name": (q(user) or m.group(1)).strip(),
                "project": m.group(2).strip()})
        if m := re.search(r"(?:start|kick off) (sprint .+?)(?: in (?:project )?(.+))?$", low):
            return self._call("start_sprint", {
                "sprint": m.group(1).strip(), "project": (m.group(2) or "").strip()})
        if m := re.search(r"(?:complete|end|finish|close|wrap up) (sprint .+?)"
                          r"(?: in (?:project )?(.+))?$", low):
            return self._call("complete_sprint", {
                "sprint": m.group(1).strip(), "project": (m.group(2) or "").strip()})
        # Checked BEFORE the generic delete_task catch-all right below, since
        # that one's optional "task" keyword would otherwise swallow "delete
        # user X" as a task named "user X".
        if m := re.search(r"(?:delete|remove) (?:the )?user(?: account)? "
                          r"(?:named |called )?(.+)", low):
            return self._call("delete_user", {"user": (q(user) or m.group(1)).strip()})
        # Same "checked before the generic catch-all" precaution as
        # delete_user just above: with no delete_space/delete_project
        # tool to route to instead, the safest thing this pattern can do
        # for "delete the space X"/"delete the project X" is NOT match
        # (falling through to the honest "I don't know how to do that
        # yet" catch-all at the bottom of this method) rather than
        # confidently misinterpret X as a TASK name and ask "which task
        # did you mean?" about something that was never a task.
        if m := re.search(r"(?:delete|remove|trash|discard) (?:task )?"
                          r"(?!(?:the )?(?:space|project)\b)(.+)", low):
            return self._call("delete_task", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"archive (?:project )?(.+)", low):
            return self._call("archive_project", {"project": (q(user) or m.group(1)).strip()})
        if m := re.search(r"who is working on (?:task )?(.+)", low):
            return self._call("task_info", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"(?:what(?:'s| is)) (?:the )?status of (?:task )?(.+)", low):
            return self._call("task_info", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"(?:details|info(?:rmation)?) (?:on|about|for) "
                          r"(?:task )?(.+)", low):
            return self._call("task_info", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"why (?:is|has|are) (?:task )?(.+?) (?:been )?"
                          r"tak(?:ing|en) (?:so )?long", low):
            return self._call("task_duration_check", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"is (?:task )?(.+?) (?:running (?:over|late|behind)|"
                          r"behind schedule)\??$", low):
            return self._call("task_duration_check", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"how long (?:has|is) (?:task )?(.+?) "
                          r"(?:been running|(?:been )?taking)", low):
            return self._call("task_duration_check", {"task": (q(user) or m.group(1)).strip()})
        if m := re.search(r"(?:which|what) tasks? (?:are )?(?:taking too long|"
                          r"running (?:over|long|behind))(?: (?:in|for)(?: project)? "
                          r"(.+))?$", low):
            return self._call("task_duration_check", {"project": (m.group(1) or "").strip()})
        if m := re.search(r"what(?:'s| is) running (?:behind|over|long)"
                          r"(?: (?:in|for)(?: project)? (.+))?$", low):
            return self._call("task_duration_check", {"project": (m.group(1) or "").strip()})
        # Checked BEFORE message_task's own catch-all right below — "the X
        # project chat" must not be swallowed there as a task literally
        # named "X project chat".
        if m := re.search(r"(?:send |post |leave )?(?:a )?(?:message|comment|note) "
                          r"(?:to|on) (?:the )?(.+?)(?:'s)? project chat\s*"
                          r"(?:saying|that says|:)\s*(.+)$", user, re.I):
            return self._call("message_project", {
                "project": m.group(1).strip(), "text": m.group(2).strip()})
        if m := re.search(r"(?:send |post |leave )?(?:a )?(?:message|comment|note) "
                          r"(?:to|on) (?:task )?(.+?)\s*(?:saying|that says|:)\s*(.+)$",
                          user, re.I):
            return self._call("message_task", {
                "task": m.group(1).strip(), "text": m.group(2).strip()})
        if m := re.search(r"move (?:my )?personal task (.+?) to "
                          r"(do[ _]?first|schedule|delegate|eliminate)", low):
            quadrant = "do_first" if m.group(2).startswith("do") else m.group(2)
            return self._call("move_personal_task", {
                "task": (q(user) or m.group(1)).strip(), "quadrant": quadrant})
        # Explicit-content phrasing only ("save note: X") — a bare "save
        # this to my notes" with no content of its own needs the LLM tiers
        # above (which see conversation history and can tell what "this"
        # refers to); the offline floor here can't reach back into context.
        if m := re.search(r"(?:save|add|create) (?:a )?note[:\s]+(.+)", user, re.I):
            return self._call("save_note", {"body": (q(user) or m.group(1)).strip()})
        if m := re.search(r"(?:note|remember) that (.+)", user, re.I):
            return self._call("save_note", {"body": (q(user) or m.group(1)).strip()})
        # Sprint summaries checked BEFORE project summaries below — "summary
        # of the last 3 sprints in project X" mentions both "sprint" and a
        # project name, and must not fall through to the project-summary
        # catch-all just because a project name is present too.
        if m := re.search(r"summar\w* of (?:tasks (?:from|in) )?(?:the )?"
                          r"last (\d+) sprints?(?: (?:in|for)(?: project)? (.+))?$", low):
            return self._call("sprint_summary", {
                "count": int(m.group(1)), "project": (m.group(2) or "").strip()})
        if m := re.search(r"summar\w* of (?:tasks (?:from|in) )?(?:the )?"
                          r"last sprint(?: (?:in|for)(?: project)? (.+))?$", low):
            return self._call("sprint_summary", {
                "count": 1, "project": (m.group(1) or "").strip()})
        if m := re.search(r"(?:summar\w* of (?:all |the )?sprints|sprint summar\w*)"
                          r"(?: (?:in|for)(?: project)? (.+))?$", low):
            return self._call("sprint_summary", {"project": (m.group(1) or "").strip()})
        # "summary of my (personal) tasks"/"...pending in ... sprint" must be
        # checked BEFORE the generic project_summary catch-all right below —
        # that one's `(.+)` would otherwise swallow "my personal tasks" or
        # "task pending in sprint" as a literal (nonexistent) project name.
        if re.search(r"summar\w* of my personal (?:tasks|to-?dos?)", low):
            return self._call("list_personal_tasks", {})
        if re.search(r"summar\w* of my tasks\b", low):
            return self._call("query_tasks", {"mine": True})
        if re.search(r"summar\w* of (?:tasks? )?pending in (?:the |a )?sprint", low):
            return self._call("sprint_summary", {"project": ""})
        if re.search(r"summar\w* of (?:the )?(?:persons?|people) who(?:'re| are)? "
                     r"over ?due|summar\w* of (?:persons?|people) overdue", low):
            return self._call("graph_query", {"question": "overdue_by_person"})
        if m := re.search(r"(?:summar\w* of (?:project )?(.+)|project summar\w* "
                          r"(?:for|of) (.+)|how'?s project (.+?) doing)$", low):
            project = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            # Bare "summary" with nothing else, or a generic "all/entire/every
            # project(s) [and timelines]"/"everything" phrase rather than an
            # actual project name -> the omit-project, across-everything form
            # (project_summary's own "" == no arg).
            if not project or re.fullmatch(
                r"everything|(?:the |all |entire |every )?projects?"
                r"(?: and timelines?)?", project):
                project = ""
            return self._call("project_summary", {"project": project})
        if "overdue" in low:
            scope = None
            if m := re.search(r"overdue (?:tasks? )?in (.+?)(?: assigned to (\w+))?$", low):
                scope = m.group(1).strip()
                if m.group(2):
                    return self._call("graph_query", {
                        "question": "overdue", "scope": scope, "assignee": m.group(2)})
            return self._call("graph_query", {"question": "overdue", "scope": scope})
        if re.search(r"who has the most tasks|workload|busiest|most loaded|"
                     r"task count", low):
            m = re.search(r"(?:in|this) (?:sprint |project |space )?(.+)$", low)
            return self._call("graph_query", {
                "question": "workload",
                "scope": m.group(1).strip() if m else None})
        if m := re.search(r"(?:what|which) tasks? (?:does|do|has) (\w+) have", low):
            return self._call("graph_query", {"question": "tasks_for",
                                              "assignee": m.group(1)})
        if m := re.search(r"(.+?)'s tasks\b", low):
            return self._call("graph_query", {"question": "tasks_for",
                                              "assignee": m.group(1).strip()})
        if m := re.search(r"tasks? assigned to (\w+)", low):
            return self._call("graph_query", {"question": "tasks_for",
                                              "assignee": m.group(1)})
        # Generic "who is X" catch-all is checked only after every more
        # specific graph_query/task_info phrasing above has had first
        # claim — "who is the busiest" etc. must not be swallowed here.
        if m := re.search(r"who is (.+?)\??$", low):
            return self._call("list_users", {"name": (q(user) or m.group(1)).strip()})
        if re.search(r"(?:list|show) (?:the |all )?(?:the )?"
                     r"(?:users|people|team members?)\b", low):
            return self._call("list_users", {})
        if m := re.search(r"who knows (.+?)\??$", low):
            return self._call("list_users", {"technology": m.group(1).strip()})
        if m := re.search(r"(?:people|users) (?:tagged|with the) (.+?)"
                          r"(?: tag)?\??$", low):
            return self._call("list_users", {"technology": m.group(1).strip()})
        if m := re.search(r"tasks? (?:tagged|with the) (.+?)(?: tag)?\??$", low):
            return self._call("query_tasks", {"technology": m.group(1).strip()})
        if m := re.search(r"(?:find|search)(?: for)? tasks? "
                          r"(?:like|about|similar to|related to) (.+)", low):
            return self._call("semantic_search", {"query": m.group(1).strip()})
        if m := re.search(r"(?:list|show|what(?:'s| is| are)?) (?:the |all )?"
                          r"(?:the )?tasks? in (?:project )?(.+)", low):
            return self._call("query_tasks", {"project": m.group(1).strip()})
        if re.search(r"\bmy personal (?:tasks|to-?dos?)\b", low):
            return self._call("list_personal_tasks", {})
        if re.search(r"\bmy tasks\b", low):
            return self._call("query_tasks", {"mine": True})
        if re.search(r"(?:list|show|what(?:'s| is| are)?) (?:the |all )?(?:the )?projects\b", low):
            return self._call("list_projects", {})
        if re.search(r"(?:list|show|what(?:'s| is| are)?) (?:the |all )?(?:the )?spaces\b", low):
            return self._call("list_spaces", {})
        if re.search(r"(?:list|show|what(?:'s| is| are)?) (?:the |all )?(?:the )?"
                     r"(?:custom )?roles\b", low):
            return self._call("list_roles", {})
        if re.search(r"(?:list|show|what(?:'s| is| are)?) (?:the |all )?(?:the )?tasks\b", low):
            return self._call("query_tasks", {})

        # JSON tool plugins (../plugins.py) get the last word before the
        # generic fallback below — never ahead of a hand-written pattern
        # above, so a plugin can't accidentally shadow a built-in phrasing.
        # Named capture groups map straight onto the tool's arguments; no
        # positional-group bookkeeping needed since every plugin author
        # names their own groups in trigger_patterns.
        for plugin in PLUGINS.values():
            for pattern in plugin.trigger_patterns:
                if m := pattern.search(user):
                    args = {k: v.strip() for k, v in m.groupdict().items() if v}
                    return self._call(plugin.name, args)

        return LLMResult(provider=self.name, text=(
            "I can create spaces/projects/tasks/sprints/subtasks, assign tasks, "
            "add project members, set due dates, change status, move tasks to "
            "sprints, start or complete sprints, delete tasks or archive "
            "projects, list projects/spaces/users, look up a task's or "
            "person's details, message a task's thread, manage personal "
            "tasks, answer 'what's overdue in X' or 'who has the most tasks', "
            "check whether a task is taking longer than planned ('why is X "
            "taking so long'), and find similar tasks. Try: create a task "
            "called 'Fix login' in the Mobile Revamp project."))

    def _call(self, name, args):
        return LLMResult(provider=self.name,
                         tool_call={"name": name, "arguments": args})
