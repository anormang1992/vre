"""
Demo callbacks for vre_guard: trace renderer, policy confirmation, and
auto-learning via meta-epistemic dialogue with the agent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel
from rich.prompt import Confirm
from rich.tree import Tree

from examples.langchain_ollama.repl import console
from vre.core.policy.models import PolicyViolation

if TYPE_CHECKING:
    from vre.core.grounding.models import GroundingResult


_SYSTEM_PROMPT = """
You are a shell command expert. Your task is to identify the conceptual primitives
touched by the given shell command.
"""

# Splits a command string on unquoted |, &&, or ; into individual segments.
_COMPOUND_PATTERN = r"""((?:[^"'|;&]|"[^"]*"|'[^']*'|&(?!&))*)(?:\||\&\&|;|$)"""


class CandidateConcepts(BaseModel):
    """Structured output containing the list of concept names."""

    concepts: list[str]


class ConceptExtractor:
    """
    LLM-based concept extraction from shell commands.

    Mirrors the DemoLearner pattern: ChatOllama chain constructed once in
    __init__ and reused across calls. Callable via __call__ so it can be
    passed directly as vre_guard's `concepts` parameter.
    """

    def __init__(self, model: str = "qwen2.5-coder:7b") -> None:
        self._llm = ChatOllama(
            model=model, temperature=0.0
        ).with_structured_output(CandidateConcepts)

    @staticmethod
    def _format_prompt(command: str) -> str:
        return f"""
            Shell command: {command}

            Identify the conceptual primitives this command touches.

            Primitives are the conceptual entities required to reason about the effects
            of the command. This includes ACTIONS, TARGETS, and concepts implied by flags.

            - Actions: read, write, delete, create, move, copy, list, execute, modify, etc.
            - Targets: file, directory, process, network, permission, etc.

            Flag-to-concept examples:
            - `rm -rf dir/` → delete + directory + file
            - `cp -a src/ dst/` → copy + file + directory + permission
            - `chmod +x script.sh` → modify + permission + file + execute
            - `find . -name "*.py" -delete` → list + delete + file + directory

            Flags themselves are NOT primitives, but they change semantic intent or
            introduce additional concepts as shown above.

            Do NOT return flag names (recursive, force, verbose, interactive, etc.)
            as primitives. Map what the flag *does* to the concepts it affects.

            Return only the list of required conceptual primitives.
        """

    @staticmethod
    def _normalize_primitives(concepts: set[str]) -> list[str]:
        normalized = set()
        for c in concepts:
            clean = c.lower().strip().replace(" ", "_")
            normalized.add(clean)
        return list(normalized)

    def __call__(self, command: str, **kwargs) -> list[str]:
        segments = [
            m.group(1).strip()
            for m in re.finditer(_COMPOUND_PATTERN, command)
        ]
        commands = [s for s in segments if s]

        primitives: set[str] = set()
        for cmd in commands:
            prompt = self._format_prompt(cmd)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            try:
                result = self._llm.invoke(messages)
            except Exception as exc:
                raise RuntimeError(
                    f"LLM invocation failed — is Ollama running with the "
                    f"configured model? (original: {exc})"
                ) from exc
            primitives.update(result.concepts)

        return self._normalize_primitives(primitives)


def get_cardinality(command: str, **kwargs) -> str:
    flags = {"-r", "-R", "-rf", "--recursive", "*"}
    tokens = set(command.split())
    has_glob = any("*" in token for token in tokens)
    has_recursive_flag = flags & tokens
    return "multiple" if has_glob or has_recursive_flag else "single"


def _gap_description(gap) -> str:
    kind = gap.kind
    if kind == "EXISTENCE":
        return f"'{gap.primitive.name}' is not in the knowledge graph"
    if kind == "DEPTH":
        curr = (
            f"D{gap.current_depth.value} {gap.current_depth.name}"
            if gap.current_depth is not None
            else "none"
        )
        req = f"D{gap.required_depth.value} {gap.required_depth.name}"
        return f"'{gap.primitive.name}' known to {curr}, requires {req}"
    if kind == "RELATIONAL":
        req = f"D{gap.required_depth.value} {gap.required_depth.name}"
        return f"{gap.source.name} → {gap.target.name} requires {req} on target"
    if kind == "REACHABILITY":
        return f"'{gap.primitive.name}' is not connected to other concepts"
    return str(gap)


def _dots(primitive, gap_level: int | None) -> str:
    present = {d.level.value for d in primitive.depths}
    max_present = max(present, default=-1)
    num = max(4, max_present + 1, (gap_level + 1) if gap_level is not None else 0)
    parts = []
    for level in range(num):
        if level in present:
            parts.append("[green]●[/]")
        elif gap_level is not None and level == gap_level:
            parts.append("[bold red]✗[/]")
        else:
            parts.append("[dim]○[/]")
    return " ".join(parts)


def on_trace(grounding: "GroundingResult") -> None:
    tree = Tree("[bold]VRE Epistemic Check[/]")

    if grounding.trace is None:
        for name in grounding.resolved:
            tree.add(f"[bold cyan]◈ {name}[/]")
        for gap in grounding.gaps:
            tree.add(f"[yellow]⚠  {_gap_description(gap)}[/]")
        tree.add(
            "[bold green]✓ Grounded — epistemic permission granted[/]"
            if grounding.grounded
            else "[bold red]✗ Not grounded — action blocked[/]"
        )
        console.print(tree)
        return

    primitives = grounding.get_primitives()
    id_to_name = {p.id: p.name for p in primitives}

    depth_gap_map: dict = {
        gap.primitive.id: gap.required_depth.value
        for gap in grounding.gaps
        if gap.kind == "DEPTH"
    }
    relational_gaps: set = {
        (gap.source.id, gap.target.id)
        for gap in grounding.gaps
        if gap.kind == "RELATIONAL"
    }

    for primitive in primitives:
        dot_str = _dots(primitive, depth_gap_map.get(primitive.id))
        branch = tree.add(f"[bold cyan]◈ {primitive.name}[/]   {dot_str}")

        for depth in sorted(primitive.depths, key=lambda d: d.level):
            for relatum in depth.relata:
                target_name = id_to_name.get(relatum.target_id, str(relatum.target_id))
                tgt_d = relatum.target_depth.value
                gap_marker = (
                    "  [bold red]✗[/]"
                    if (primitive.id, relatum.target_id) in relational_gaps
                    else ""
                )
                branch.add(
                    f"[dim]{relatum.relation_type.value}[/]  →  [cyan]{target_name}[/]"
                    f"  [dim](target D{tgt_d})[/]{gap_marker}"
                )

    for gap in grounding.gaps:
        tree.add(f"[yellow]⚠  {_gap_description(gap)}[/]")

    tree.add(
        "[bold green]✓ Grounded — EPISTEMIC PERMISSION GRANTED[/]"
        if grounding.grounded
        else "[bold red]✗ Not grounded — COMMAND EXECUTION IS BLOCKED[/]"
    )

    console.print(tree)


def on_policy(violations: list[PolicyViolation]) -> bool:
    """
    Prompt the user interactively for each confirmation-required policy violation.

    Returns True if the user confirms all violations, False if any is declined.
    The first declined violation short-circuits the loop.
    """
    for v in violations:
        if not Confirm.ask(f"[yellow]⚠  Policy gate:[/] {v.message}"):
            return False
    return True
