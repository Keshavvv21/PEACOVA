"""The JSON tool-plugin framework (app/plugins.py + plugins.d/*.json) — a
new PM action wired up purely from a config file, exercised the same way
every hand-written tool is: through the real chat endpoint, on both agent
tracks, through the rule-based (offline) floor.
"""
import json

import pytest

from app import plugins
from tests.test_agent_flows import TRACKS, say


def test_rename_project_plugin_is_loaded_and_merged_into_tool_schemas():
    from app.agent import tools
    assert "rename_project" in plugins.PLUGINS
    assert "rename_project" in tools.TOOL_SCHEMAS
    assert tools.TOOL_SCHEMAS["rename_project"] == plugins.PLUGINS["rename_project"].schema


@pytest.mark.parametrize("track", TRACKS)
def test_rename_project_via_chat_both_tracks(client, exec_token, track):
    """No Python code for this capability — trigger_patterns (the
    rule-based floor), resolve (entity resolution), and service_call (which
    whitelisted pm_service method to call) are entirely in
    plugins.d/rename_project.json."""
    d = say(client, exec_token,
            f"rename project Compass OS to Renamed By Plugin {track}", track, "rename")
    assert d["reply"] == f"Renamed the project to “Renamed By Plugin {track}”."

    # rename it back so later tests that look up "Compass OS" by name still
    # find it (client/exec_token/seed data are session-scoped fixtures)
    say(client, exec_token, f"rename project Renamed By Plugin {track} to Compass OS",
        track, "rename")


@pytest.mark.parametrize("track", TRACKS)
def test_rename_project_unknown_project_asks_to_clarify(client, exec_token, track):
    d = say(client, exec_token,
            "rename project zzznonexistentzzz to Whatever", track, "rename-missing")
    assert d["pending_confirmation"] is False
    assert "couldn't find a project" in d["reply"].lower()


def test_rename_project_requires_manager_role(client, member_token):
    d = say(client, member_token, "rename project Compass OS to Nope", "native", "rename-member")
    assert "403" in d["reply"] or "requires" in d["reply"]


def test_malformed_plugin_json_is_skipped_not_a_crash(tmp_path):
    (tmp_path / "broken.json").write_text("{not valid json")
    loaded = _load_from(tmp_path)
    assert loaded == {}


def test_plugin_with_unregistered_service_method_is_skipped(tmp_path):
    (tmp_path / "sneaky.json").write_text(json.dumps({
        "name": "delete_everything", "description": "x",
        "response_template": "done",
        "service_call": {"method": "drop_database", "args": {}},
    }))
    loaded = _load_from(tmp_path)
    assert loaded == {}


def test_plugin_missing_required_keys_is_skipped(tmp_path):
    (tmp_path / "incomplete.json").write_text(json.dumps({"name": "no_service_call"}))
    loaded = _load_from(tmp_path)
    assert loaded == {}


def _load_from(directory) -> dict:
    """Loads plugins from an arbitrary directory without touching the
    module-level plugins.PLUGINS singleton (which every other test/import
    relies on already being loaded from the real plugins.d/)."""
    found = {}
    for path in sorted(directory.glob("*.json")):
        p = plugins._load_one(path)
        if p is not None:
            found[p.name] = p
    return found
