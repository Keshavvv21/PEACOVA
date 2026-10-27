# Tool plugins — add a PM action with JSON, no Python

Drop a `.json` file here. On the next process start, it's loaded (`app/
plugins.py`'s `load_plugins()`) and becomes indistinguishable from a
hand-written tool in `agent/tools.py`: it appears in `perceive_intent`'s
tool schema, `output_guard` validates calls to it, the rule-based floor can
trigger it offline, and its target gets fuzzy-resolved to a real id exactly
like any built-in tool — never invented by an LLM.

**What a plugin can't do**: call arbitrary code. `service_call.method` must
already be one of the names in `plugins.py`'s `SERVICE_METHOD_REGISTRY` — a
fixed whitelist of already-reviewed, already-role-gated `pm_service`
functions. A JSON file wires a *new phrase* to an *existing, approved*
action; registering a genuinely new action still means writing (and
reviewing) one function in `services/pm_service.py`, the same bar every
hand-written tool already clears.

## File format

`rename_project.json`, the worked example in this directory:

```json
{
  "name": "rename_project",
  "description": "Rename a project",
  "required": ["project", "new_name"],
  "properties": {
    "project": {"type": "string"},
    "new_name": {"type": "string"}
  },
  "trigger_patterns": [
    "rename project (?P<project>.+?) to (?P<new_name>.+)",
    "rename (?P<project>.+?) to (?P<new_name>.+)"
  ],
  "resolve": {
    "project": {"entity": "project", "as": "project_id"}
  },
  "service_call": {
    "method": "rename_project",
    "args": {"project_id": "project_id", "new_name": "new_name"}
  },
  "response_template": "Renamed the project to “{new_name}”.",
  "destructive": false
}
```

| Field | Required | What it does |
|---|---|---|
| `name` | yes | The tool's id — must be unique, becomes the key in `TOOL_SCHEMAS`. |
| `description` | yes | Shown to a real LLM's `perceive_intent` call as the tool's purpose. |
| `required` / `properties` | no | Standard JSON-schema-shaped arg spec — `output_guard` enforces `required` and each property's `type` (`string`/`boolean` today) exactly like a hand-written tool. |
| `trigger_patterns` | no | Regexes with **named groups** (`(?P<x>...)`), tried by the rule-based floor (offline mode) after every hand-written pattern, in file-listing order. A capture becomes an argument named after its group. Omit this if the plugin should only be reachable via a real LLM's tool-calling. |
| `resolve` | no | `{arg_name: {"entity": "project\|space\|task\|user\|sprint", "as": resolved_key}}` — fuzzy-matches that argument against live data via the *same* `resolve_project`/`resolve_task`/... functions every built-in tool uses (never reimplemented, never guesses: ambiguous → asks which one; missing → asks to clarify). Any argument *not* listed here passes through unresolved (e.g. a plain new name). |
| `service_call.method` | yes | Must be a key in `SERVICE_METHOD_REGISTRY` (`plugins.py`) — the whitelist. |
| `service_call.args` | yes | `{kwarg_name: resolved_key}` — maps the service method's parameter names onto keys produced by `resolve` (or passed straight through). |
| `response_template` | yes | Plain `str.format(**fields)` — `fields` is every resolved/passthrough value plus whatever dict the service call returned. No `eval`, just named substitution. |
| `destructive` | no (default `false`) | `true` routes it through the same confirm-before-executing gate as `delete_task`/`archive_project`/`complete_sprint` — the agent asks "yes/no" and only calls the service method on "yes". |

## Registering a new service method

```python
# plugins.py
SERVICE_METHOD_REGISTRY: dict[str, Callable[..., dict]] = {
    "rename_project": pm_service.rename_project,
    "your_new_action": pm_service.your_new_action,   # add this line
}
```

That's the one Python-side step — a deliberate one, since it's the actual
security decision (is this function safe to expose to chat, does it call
`_need()` with the right role). Everything after that — which phrases
trigger it, which arguments need resolving, how the confirmation reads —
is JSON.

## A malformed or misconfigured plugin never breaks startup

Invalid JSON, a missing required key, or a `service_call.method` not in the
registry — each is logged as a warning and that one file is skipped; every
other plugin and every built-in tool still loads. Same "optional, additive
config" philosophy as `EXTRA_LLM_PROVIDERS` (see `llm/README.md`) applied to
the tool layer instead of the model layer. See `tests/test_plugins.py`'s
`test_malformed_plugin_json_is_skipped_not_a_crash` and its neighbors.
