"""
VRE Demo Agent — entry point.

Usage:
    python -m examples.langchain_ollama.main [--neo4j-uri ...] [--model ...] [--sandbox ...]
"""

from __future__ import annotations

import argparse
import os

from langchain_core.tools import StructuredTool

from vre import VRE
from vre.core.graph import PrimitiveRepository

from examples.langchain_ollama.agent import make_agent
from examples.langchain_ollama.callbacks import ConceptExtractor, get_cardinality, make_on_learn, on_policy, on_trace
from examples.langchain_ollama.repl import run
from examples.langchain_ollama.tools import init_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="VRE Demo Agent")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--model", default="qwen3.5:latest")
    parser.add_argument("--sandbox", default="examples/langchain_ollama/workspace")
    parser.add_argument("--concepts-model", default="qwen2.5-coder:7b")
    args = parser.parse_args()

    os.makedirs(args.sandbox, exist_ok=True)

    repo = PrimitiveRepository(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    vre = VRE(repo)

    concepts = ConceptExtractor(model=args.concepts_model)
    on_learn = make_on_learn(model=args.model)
    shell_fn = init_tools(
        vre,
        args.sandbox,
        concepts,
        get_cardinality,
        on_trace,
        on_policy,
        on_learn,
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
