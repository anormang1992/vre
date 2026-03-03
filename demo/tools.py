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
):
    @vre_guard(
        vre,
        concepts=concepts,
        cardinality=cardinality,
        on_trace=on_trace,
        on_policy=on_policy,
    )
    def shell_tool(command: str) -> str:
        """Execute a shell command inside the sandbox directory."""
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=sandbox
        )
        return result.stdout + result.stderr

    return shell_tool
