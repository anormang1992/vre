"""
Demo PolicyCallback — blocks deletion of files matching `protected*`.

Intended to be attached to the Delete → File APPLIES_TO relatum by the
integrator (via the policy wizard or a direct attachment), not by the
domain seeder. Demonstrates a callback that inspects the shell command
to make a domain-specific policy decision, including filesystem
inspection when a wildcard or directory deletion could affect protected
files.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
from pathlib import Path

from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.models import PolicyCallbackResult

_PROTECTED_PATTERN = "protected*"


def _extract_command(context: PolicyCallContext) -> str:
    """
    Pull the command string from call_args or call_kwargs.
    """
    if context.call_args:
        command = str(context.call_args[0])
    else:
        command = str(context.call_kwargs.get("command", ""))
    return command


def _matches_protected(name: str) -> bool:
    """
    Check if a path's basename literally matches `protected*`.
    """
    basename = Path(name).name
    return fnmatch.fnmatch(basename, _PROTECTED_PATTERN)


def _is_glob(pattern: str) -> bool:
    """
    Check if a string contains glob metacharacters.
    """
    return "*" in pattern or "?" in pattern or "[" in pattern


def _glob_matches_protected(pattern: str, cwd: str) -> list[str]:
    """
    Expand a glob pattern against the filesystem and return any matches
    that are protected files.

    Resolves relative paths against *cwd* (the sandbox directory).
    Returns the list of matching protected filenames (empty if none).
    If the directory doesn't exist or can't be listed, returns empty
    (fail open — the literal match check already covers explicit names).
    """
    if not _is_glob(pattern):
        matches = []
    else:
        parent = str(Path(pattern).parent)
        basename_pattern = Path(pattern).name
        directory = str(Path(cwd) / parent) if not Path(parent).is_absolute() else parent
        try:
            if not os.path.isdir(directory):
                matches = []
            else:
                entries = os.listdir(directory)
                matches = [
                    e for e in entries
                    if fnmatch.fnmatch(e, basename_pattern)
                    and fnmatch.fnmatch(e, _PROTECTED_PATTERN)
                ]
        except OSError:
            matches = []

    return matches


def _directory_contains_protected(target: str, cwd: str) -> bool:
    """
    List a directory and check for entries matching `protected*`.

    Resolves relative paths against *cwd* (the sandbox directory).
    Returns True if the path is a directory containing protected files.
    Returns False if the path doesn't exist, isn't a directory, or
    contains no protected files.
    """
    resolved = str(Path(cwd) / target) if not Path(target).is_absolute() else target
    try:
        if not os.path.isdir(resolved):
            return False
        entries = os.listdir(resolved)
        return any(fnmatch.fnmatch(e, _PROTECTED_PATTERN) for e in entries)
    except OSError:
        return False

def protected_file_delete(context: PolicyCallContext) -> PolicyCallbackResult:
    """
    Inspect an `rm` command and block if it would affect protected files.

    Three detection modes:

    1. **Literal match** — a target argument starts with `protected`
       (e.g. `rm protected_secret.txt`).
    2. **Glob expansion** — a target contains a wildcard. Lists the
       target directory and checks whether any files matching the glob
       are also protected (e.g. `rm *.txt` in a dir with `protected_config.txt`).
    3. **Directory inspection** — a recursive delete targets a directory.
       Lists the directory and checks whether any entry matches `protected*`.

    Returns `passed=True` (no violation) when the command is safe,
    `passed=False` (violation fires) when protected files are at risk.
    """
    command: str = _extract_command(context)
    cwd: str = context.call_kwargs.get("cwd", ".")

    if not command:
        callback_result = PolicyCallbackResult(passed=True)
    else:
        try:
            tokens = shlex.split(command)
        except ValueError:
            # Malformed command — fail closed
            return PolicyCallbackResult(
                passed=False,
                message="Could not parse command; assuming protected files at risk.",
            )

        if not tokens or tokens[0] != "rm":
            callback_result = PolicyCallbackResult(passed=True)
        else:
            is_recursive = any(t in ("-r", "-R", "-rf", "-fr", "--recursive") for t in tokens)
            targets = [t for t in tokens[1:] if not t.startswith("-")]
            callback_result = PolicyCallbackResult(passed=True)
            for target in targets:
                # 1. Literal match
                if _matches_protected(target):
                    callback_result = PolicyCallbackResult(
                        passed=False,
                        message=f"'{target}' matches a protected file pattern.",
                    )
                    break
                # 2. Glob expansion — check the actual directory contents
                matched = _glob_matches_protected(target, cwd)
                if matched:
                    callback_result = PolicyCallbackResult(
                        passed=False,
                        message=f"Glob '{target}' would delete protected files: {', '.join(matched)}",
                    )
                    break
                # 3. Directory inspection (recursive delete or bare directory path)
                elif is_recursive and _directory_contains_protected(target, cwd):
                    callback_result = PolicyCallbackResult(
                        passed=False,
                        message=f"Directory '{target}' contains protected files.",
                    )
                    break

    return callback_result
