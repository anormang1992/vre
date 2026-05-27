# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Claude Code PreToolUse hook integration for VRE.

Intercepts every Bash tool call before execution and gates it through
VRE grounding and policy evaluation using a two-pass protocol:

1. **First call** (no `# vre:` prefix) — the hook blocks and instructs
   Claude to identify the conceptual primitives the command touches and
   retry with a `# vre:concept1,concept2` prefix.
2. **Second call** (with prefix) — the hook extracts the concepts, checks
   grounding and policy, and if allowed returns `updatedInput` with the
   prefix stripped so the executed command is clean.

This lets Claude — the LLM — propose the concepts itself rather than
relying on a static command-to-concept map.

Setup (SQLite — default, zero config)::

    python examples/claude-code/claude_code.py install

Setup (Neo4j)::

    python examples/claude-code/claude_code.py install \
        --backend neo4j --uri neo4j://localhost:7687 --user neo4j --password password

Removal::

    python examples/claude-code/claude_code.py uninstall

Hook protocol::

    Exit 0 + JSON stdout with permissionDecision "allow" → command proceeds.
    Exit 2 + stderr message → command blocked, stderr fed to Claude.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

from vre import VRE, PolicyAction
from vre.core.backends import Neo4jRepository, SQLiteRepository


_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
_VRE_CONFIG_PATH = Path.home() / ".vre" / "config.json"


_HOOK_MARKER = "examples/claude-code/claude_code.py"

# Matches a leading `# vre:concept1,concept2` comment line.
_VRE_PREFIX_RE = re.compile(r"^#\s*vre:\s*(.+)")

_CONCEPT_REQUEST = """\
VRE epistemic check required. Before executing this command, identify the \
conceptual primitives it touches (actions like delete, create, read; targets \
like file, directory, permission) and retry with a `# vre:` prefix. Example:

  # vre:delete,file,directory
  rm -rf foo/

Do NOT include flag names (recursive, force, verbose) as primitives — map \
what the flag *does* to the concepts it affects. Retry the command with the \
prefix now."""


def _hook_command() -> str:
    """
    Build the hook command string using the current interpreter and this script's
    absolute path.

    This ensures the hook runs in the same virtualenv where VRE is installed,
    regardless of what `python` resolves to in Claude Code's shell.
    """
    script = Path(__file__).resolve()
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _is_vre_hook(hook_entry: dict) -> bool:
    """
    Check whether a hook entry belongs to VRE, regardless of interpreter path.
    """
    return _HOOK_MARKER in json.dumps(hook_entry)


_EXIT_ALLOW = 0
_EXIT_BLOCK = 2


def _allow(
    reason: str | None = None,
    updated_input: dict | None = None,
) -> None:
    """
    Exit the hook allowing tool execution to proceed.

    Writes a JSON response with permissionDecision "allow" to stdout
    and exits with code 0. If *updated_input* is provided it is included
    so Claude Code replaces the tool input before execution (e.g. to
    strip the `# vre:` prefix).
    """
    output: dict = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }
    if reason:
        output["hookSpecificOutput"]["permissionDecisionReason"] = reason
    if updated_input is not None:
        output["hookSpecificOutput"]["updatedInput"] = updated_input
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


def install(
    backend: str = "sqlite",
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
    path: str | None = None,
) -> None:
    """
    Install the VRE PreToolUse hook into Claude Code's settings.

    Writes backend configuration to ~/.vre/config.json and injects
    the hook entry into ~/.claude/settings.json. Safe to call multiple
    times — existing VRE hook entries are replaced, not duplicated.
    """
    _VRE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if backend == "sqlite":
        config: dict = {"backend": "sqlite"}
        if path is not None:
            config["path"] = path
    else:
        config = {
            "backend": "neo4j",
            "uri": uri,
            "user": user,
            "password": password,
            "database": database,
        }
    _VRE_CONFIG_PATH.write_text(json.dumps(config, indent=2))
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
    the prompt so they can make an informed decision. If the user approves,
    the hook re-runs and the prefix is stripped on the final allow path.
    """
    output: dict = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(output))
    sys.exit(_EXIT_ALLOW)


def _parse_vre_prefix(command: str) -> tuple[list[str], str] | None:
    """
    Extract concepts and the clean command from a `# vre:` prefixed string.

    Returns `(concepts, clean_command)` if the prefix is present, else None.
    """
    first_line, _, rest = command.partition("\n")
    match = _VRE_PREFIX_RE.match(first_line.strip())
    if not match:
        return None
    raw = match.group(1)
    concepts = [c.strip().lower() for c in raw.split(",") if c.strip()]
    clean = rest.strip() if rest else ""
    return concepts, clean


def _run_hook() -> None:
    """
    Hook entry point invoked by Claude Code for every Bash tool call.

    Uses a two-pass protocol:

    **Pass 1** — command has no `# vre:` prefix. The hook blocks and
    instructs Claude to identify the conceptual primitives and retry
    with the prefix.

    **Pass 2** — command has a `# vre:concept1,concept2` prefix. The
    hook extracts concepts, checks grounding and policy, and responds:

    - Grounded, no policy → allow with updatedInput (prefix stripped).
    - Confirmation-required violations → ask with updatedInput.
    - Hard policy block or ungrounded → block (exit 2).

    Fails open when the VRE config is absent.
    """
    try:
        payload = json.loads(sys.stdin.read())
        command: str = payload.get("tool_input", {}).get("command", "")

        if not command:
            _allow()

        parsed = _parse_vre_prefix(command)

        if parsed is None:
            _block(_CONCEPT_REQUEST)

        concepts, clean_command = parsed  # type: ignore[misc]
        updated_input = {"command": clean_command}

        if not concepts:
            _block(_CONCEPT_REQUEST)

        if not _VRE_CONFIG_PATH.exists():
            _allow("No VRE config found", updated_input)

        config = json.loads(_VRE_CONFIG_PATH.read_text())

        backend = config.get("backend", "sqlite")
        if backend == "sqlite":
            repo_ctx = SQLiteRepository(config.get("path"))
        else:
            repo_ctx = Neo4jRepository(
                config["uri"], config["user"], config["password"],
                config.get("database", "neo4j"),
            )
        with repo_ctx as repo:
            vre = VRE(repo)
            grounding = vre.check(concepts)

            if not grounding.grounded:
                _block(str(grounding))

            policy = vre.check_policy(grounding)

        if policy.action == PolicyAction.BLOCK:
            confirmation_violations = [
                v for v in policy.violations if v.requires_confirmation
            ]
            if confirmation_violations:
                reason = "\n".join(
                    f"- {v.message}" for v in confirmation_violations
                )
                _ask(reason)
            else:
                _block(str(policy))

        _allow(updated_input=updated_input)
    except SystemExit:
        raise
    except Exception:
        _allow("VRE hook error — failing open")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VRE Claude Code hook")
    sub = parser.add_subparsers(dest="command")

    inst = sub.add_parser("install", help="Install the VRE PreToolUse hook")
    inst.add_argument("--backend", choices=["neo4j", "sqlite"], default="sqlite",
                      help="Persistence backend (default: sqlite)")

    neo4j = inst.add_argument_group("neo4j", "Options for --backend neo4j")
    neo4j.add_argument("--uri", default=None)
    neo4j.add_argument("--user", default=None)
    neo4j.add_argument("--password", default=None)
    neo4j.add_argument("--database", default="neo4j")

    sqlite = inst.add_argument_group("sqlite", "Options for --backend sqlite")
    sqlite.add_argument("--path", default=None,
                        help="Database path (default: ~/.vre/graph.db)")

    sub.add_parser("uninstall", help="Remove the VRE PreToolUse hook")

    args = parser.parse_args()

    if args.command == "install":
        if args.backend == "neo4j" and not all([args.uri, args.user, args.password]):
            parser.error("--backend neo4j requires --uri, --user, and --password")
        install(
            backend=args.backend,
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            path=args.path,
        )
    elif args.command == "uninstall":
        uninstall()
    else:
        # No subcommand — invoked as hook by Claude Code
        _run_hook()
