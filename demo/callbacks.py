"""
Demo callbacks for vre_guard: trace renderer and policy confirmation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.prompt import Confirm
from rich.tree import Tree

from demo.repl import console
from vre.builtins.shell import parse_bash_primitives

if TYPE_CHECKING:
    from vre.core.grounding.models import GroundingResult


def get_concepts(command: str) -> list[str]:
    return parse_bash_primitives(command)


def get_cardinality(command: str) -> str:
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
            "[bold green]✓ Grounded at D3 — epistemic permission granted[/]"
            if grounding.grounded
            else "[bold red]✗ Not grounded — action blocked[/]"
        )
        console.print(tree)
        return

    primitives = grounding.trace.result.primitives
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
        "[bold green]✓ Grounded at D3 — EPISTEMIC PERMISSION GRANTED[/]"
        if grounding.grounded
        else "[bold red]✗ Not grounded — COMMAND EXECUTION IS BLOCKED[/]"
    )

    console.print(tree)


def on_policy(message: str) -> bool:
    return Confirm.ask(f"[yellow]⚠  Policy gate:[/] {message}")
