"""
VRE Demo Agent — entry point.

Usage:
    python -m demo.main [--neo4j-uri ...] [--model ...] [--sandbox ...]
"""

from __future__ import annotations

import argparse
import os

from langchain_core.tools import StructuredTool

from vre import VRE
from vre.core.graph import PrimitiveRepository

from demo.agent import make_agent
from demo.callbacks import get_concepts, get_cardinality, on_policy, on_trace
from demo.repl import run
from demo.tools import init_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="VRE Demo Agent")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--sandbox", default="demo/workspace")
    args = parser.parse_args()

    os.makedirs(args.sandbox, exist_ok=True)

    repo = PrimitiveRepository(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    vre = VRE(repo)

    shell_fn = init_tools(
        vre,
        args.sandbox,
        get_concepts,
        get_cardinality,
        on_trace,
        on_policy
    )
    shell_tool = StructuredTool.from_function(
        shell_fn,
        name="shell_tool",
        description="Run a shell command in the sandbox.",
    )

    agent = make_agent([shell_tool], model=args.model)
    run(agent)


if __name__ == "__main__":
    main()
