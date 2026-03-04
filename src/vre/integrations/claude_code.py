# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Claude Code PreToolUse hook integration for VRE.

Intercepts every Bash tool call before execution and gates it through
VRE grounding and policy evaluation. Commands whose concepts are not
grounded at D3 are blocked and knowledge gaps are surfaced. Commands
that are grounded but trigger a policy gate surface a confirmation
prompt via Claude Code's TUI approval dialog — human consent, not model consent.

Setup::

    from vre.integrations.claude_code import install
    install("neo4j://localhost:7687", "neo4j", "password")

Removal::

    from vre.integrations.claude_code import uninstall
    uninstall()

Hook protocol::

    Exit 0 + JSON stdout with permissionDecision "allow" → command proceeds.
    Exit 2 + stderr message → command blocked, stderr fed to Claude.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
_VRE_CONFIG_PATH = Path.home() / ".vre" / "config.json"


_MODULE = "vre.integrations.claude_code"


def _hook_command() -> str:
    """
    Build the hook command string using the current interpreter's absolute path.

    This ensures the hook runs in the same virtualenv where VRE is installed,
    regardless of what `python` resolves to in Claude Code's shell.
    """
    return f"{shlex.quote(sys.executable)} -m {_MODULE}"


def _is_vre_hook(hook_entry: dict) -> bool:
    """
    Check whether a hook entry belongs to VRE, regardless of interpreter path.
    """
    return _MODULE in json.dumps(hook_entry)

_EXIT_ALLOW = 0
_EXIT_BLOCK = 2


def _allow(reason: str | None = None) -> None:
    """
    Exit the hook allowing tool execution to proceed.

    Writes a JSON response with permissionDecision "allow" to stdout
    and exits with code 0.
    """
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }
    if reason:
        output["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(output))
    sys.exit(_EXIT_ALLOW)


def _block(message: str) -> None:
    """
    Exit the hook blocking tool execution.

    Writes the message to stderr (which Claude Code feeds to the model)
    and exits with code 2.
    """
    print(message, file=sys.stderr)
    sys.exit(_EXIT_BLOCK)


def install(uri: str, user: str, password: str, database: str = "neo4j") -> None:
    """
    Install the VRE PreToolUse hook into Claude Code's settings.

    Writes Neo4j connection details to ~/.vre/config.json and injects
    the hook entry into ~/.claude/settings.json. Safe to call multiple
    times — existing VRE hook entries are replaced, not duplicated.
    """
    _VRE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _VRE_CONFIG_PATH.write_text(json.dumps(
        {"uri": uri, "user": user, "password": password, "database": database},
        indent=2,
    ))
    _VRE_CONFIG_PATH.chmod(0o600)

    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings: dict = {}
    if _SETTINGS_PATH.exists():
        settings = json.loads(_SETTINGS_PATH.read_text())

    pre_tool_use: list = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
    pre_tool_use[:] = [h for h in pre_tool_use if not _is_vre_hook(h)]
    pre_tool_use.append({
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": _hook_command()}],
    })

    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    print(f"VRE hook installed. Config: {_VRE_CONFIG_PATH}")


def uninstall() -> None:
    """
    Remove the VRE PreToolUse hook from Claude Code's settings.
    """
    if not _SETTINGS_PATH.exists():
        return

    settings = json.loads(_SETTINGS_PATH.read_text())
    pre_tool_use: list = settings.get("hooks", {}).get("PreToolUse", [])
    pre_tool_use[:] = [h for h in pre_tool_use if not _is_vre_hook(h)]
    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    print("VRE hook removed.")


def _ask(reason: str) -> None:
    """
    Exit the hook deferring the decision to the user via Claude Code's TUI.

    Returns permissionDecision "ask", which causes Claude Code to show its
    normal approval prompt. The reason is displayed to the user alongside
    the prompt so they can make an informed decision.
    """
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(output))
    sys.exit(_EXIT_ALLOW)


def _run_hook() -> None:
    """
    Hook entry point invoked by Claude Code for every Bash tool call.

    Reads the tool call payload from stdin, extracts the command, maps
    it to VRE concepts via parse_bash_primitives, then runs grounding
    followed by policy evaluation.

    - Ungrounded: exit 2, grounding trace to stderr (fed to Claude).
    - Policy PENDING: defers to Claude Code's TUI approval prompt so
      the human can accept or reject the action directly.
    - Policy BLOCK: exit 2, policy result to stderr.
    - Grounded and no policy: exit 0 with permissionDecision "allow".

    Fails open (allows) when no concepts are recognised or when the
    VRE config is absent — unknown commands are never silently blocked.
    """
    try:
        payload = json.loads(sys.stdin.read())
        command: str = payload.get("tool_input", {}).get("command", "")

        if not command:
            _allow()

        from vre.builtins.shell import parse_bash_primitives
        concepts = parse_bash_primitives(command)

        if not concepts:
            _allow("No recognised VRE concepts in command")

        if not _VRE_CONFIG_PATH.exists():
            _allow("No VRE config found")

        config = json.loads(_VRE_CONFIG_PATH.read_text())

        from vre import VRE
        from vre.core.graph import PrimitiveRepository

        with PrimitiveRepository(
            config["uri"], config["user"], config["password"], config.get("database", "neo4j")
        ) as repo:
            vre = VRE(repo)
            grounding = vre.check(concepts)

            if not grounding.grounded:
                _block(str(grounding))

            policy = vre.check_policy(grounding)

        if policy.action == "PENDING":
            _ask(policy.confirmation_message or "This action requires confirmation.")

        if policy.action == "BLOCK":
            _block(str(policy))

        _allow()
    except SystemExit:
        raise
    except Exception:
        _allow("VRE hook error — failing open")


if __name__ == "__main__":
    _run_hook()
