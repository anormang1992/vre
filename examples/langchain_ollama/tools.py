"""
VRE-guarded shell tool for the demo agent.
"""

from __future__ import annotations

import subprocess
from typing import Callable

from vre.guard import vre_guard


def init_tools(
    vre,
    sandbox: str,
    concepts: Callable,
    cardinality: Callable,
    on_trace: Callable,
    on_policy: Callable,
    on_learn: Callable | None = None,
):
    @vre_guard(
        vre,
        concepts=concepts,
        cardinality=cardinality,
        on_trace=on_trace,
        on_policy=on_policy,
        on_learn=on_learn,
    )
    def shell_tool(command: str, cwd: str = sandbox) -> str:
        """
        Execute a shell command inside the workspace directory. The
        workspace is fully writable — files can be created, modified,
        deleted, and executed here. Use relative paths to stay within
        the workspace.
        """
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.stdout + result.stderr

    return shell_tool
