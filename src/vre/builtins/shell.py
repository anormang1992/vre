# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Shell command → concept extraction for VRE enforcement.

Maps common shell command names to the conceptual primitives they invoke.
"""

from __future__ import annotations

import shlex

# Maps shell command names to the VRE concept names they invoke.
# Keys are lowercase command names; values are lists of concept names.
SHELL_ALIASES: dict[str, list[str]] = {
    # Filesystem — read
    "ls": ["list", "directory"],
    "cat": ["read", "file"],
    "head": ["read", "file"],
    "tail": ["read", "file"],
    "less": ["read", "file"],
    "more": ["read", "file"],
    "find": ["list", "file"],
    "stat": ["read", "file"],
    # Filesystem — write / create
    "touch": ["create", "file"],
    "mkdir": ["create", "directory"],
    "cp": ["copy", "file"],
    "mv": ["move", "file"],
    "echo": ["write", "file"],
    "tee": ["write", "file"],
    # Filesystem — delete
    "rm": ["delete", "file"],
    "rmdir": ["delete", "directory"],
    # Network
    "curl": ["network", "request"],
    "wget": ["network", "request"],
    "ping": ["network"],
    "ssh": ["network", "connection"],
    "scp": ["copy", "file", "network"],
    # Processes
    "kill": ["terminate", "process"],
    "pkill": ["terminate", "process"],
    "ps": ["list", "process"],
    "lsof": ["list", "process", "network"],
    # Archives
    "tar": ["archive", "file"],
    "zip": ["archive", "file"],
    "unzip": ["extract", "file"],
    # Text processing
    "grep": ["read", "file"],
    "sed": ["write", "file"],
    "awk": ["read", "file"],
    # Package management
    "pip": ["install", "package"],
    "poetry": ["install", "package"],
    "npm": ["install", "package"],
    "brew": ["install", "package"],
    # Permissions
    "chmod": ["permission", "file"],
    "chown": ["permission", "file"],
    "sudo": ["permission"],
    "whoami": ["user"],
}


def parse_bash_primitives(command: str) -> list[str]:
    """
    Extract VRE primitive names from a shell command string.

    Splits the command with shlex, looks up each token against SHELL_ALIASES,
    and returns the deduplicated union of all matched primitive names. Returns an
    empty list if no tokens are recognised.

    Parameters
    ----------
    command:
        A shell command string, e.g. "rm -rf /tmp/foo".

    Returns
    -------
    list of canonical primitive names, e.g. ["delete", "file"].
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    if not tokens:
        return []

    primitives: set[str] = set()
    for token in tokens:
        key = token.lower().rsplit("/", 1)[-1]
        primitives.update(SHELL_ALIASES.get(key, []))

    return list(primitives)
