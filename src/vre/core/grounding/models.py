# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
GroundingResult — the public result type returned by VRE grounding checks.
"""

from typing import Any

from pydantic import BaseModel

from vre.core.models import EpistemicResponse


# ── Private formatting helpers ────────────────────────────────────────────────

def _fmt_gap(gap: Any) -> str:
    """
    Format a KnowledgeGap as a human-readable string.
    """
    kind = gap.kind
    if kind == "EXISTENCE":
        return f"EXISTENCE: '{gap.primitive.name}' is not in the knowledge graph"
    if kind == "DEPTH":
        curr = (
            f"D{gap.current_depth.value} {gap.current_depth.name}"
            if gap.current_depth is not None
            else "none"
        )
        req = f"D{gap.required_depth.value} {gap.required_depth.name}"
        return f"DEPTH: '{gap.primitive.name}' known to {curr}, requires {req}"
    if kind == "RELATIONAL":
        curr = (
            f"D{gap.current_depth.value} {gap.current_depth.name}"
            if gap.current_depth is not None
            else "none"
        )
        req = f"D{gap.required_depth.value} {gap.required_depth.name}"
        return (
            f"RELATIONAL: '{gap.source.name}' → '{gap.target.name}' "
            f"requires {req} on target, found {curr}"
        )
    if kind == "REACHABILITY":
        return f"REACHABILITY: '{gap.primitive.name}' is not connected to other concepts"
    return f"UNKNOWN: {gap}"


def _fmt_relatum(r: Any, id_to_name: dict) -> list[str]:
    """
    Format a single Relatum as display lines, including metadata and policy count if present.
    """
    target_name = id_to_name.get(r.target_id, str(r.target_id))
    lines = [f"      → {target_name}  [{r.relation_type.value}, target@D{r.target_depth.value}]"]
    if r.metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in r.metadata.items())
        lines.append(f"        metadata: {meta_str}")
    if r.policies:
        n = len(r.policies)
        word = "policy" if n == 1 else "policies"
        lines.append(f"        policies: {n} {word}")
    return lines


def _fmt_depth(depth: Any, id_to_name: dict) -> list[str]:
    """
    Format a single Depth level as display lines, including its relata.
    """
    lines = [f"  D{depth.level.value} {depth.level.name}"]
    if depth.properties:
        lines.append("    properties:")
        for k, v in depth.properties.items():
            lines.append(f"      {k}: {v}")
    if depth.relata:
        lines.append("    relata:")
        for r in depth.relata:
            lines.extend(_fmt_relatum(r, id_to_name))
    return lines


def _fmt_primitive(primitive: Any, id_to_name: dict) -> list[str]:
    """
    Format a Primitive and all its depths as display lines.
    """
    name = primitive.name
    header = f"═══ {name} {'═' * max(0, 50 - len(name))}"
    lines = [header]

    if not primitive.depths:
        return lines

    # Compact single-line format when all depths have no properties or relata
    all_empty = all(not d.properties and not d.relata for d in primitive.depths)
    if all_empty:
        labels = [
            f"D{d.level.value} {d.level.name}"
            for d in sorted(primitive.depths, key=lambda d: d.level)
        ]
        lines.append("  " + " → ".join(labels))
    else:
        for depth in sorted(primitive.depths, key=lambda d: d.level):
            lines.extend(_fmt_depth(depth, id_to_name))

    return lines


# ── Public result type ────────────────────────────────────────────────────────

class GroundingResult(BaseModel):
    """
    Result of a VRE grounding check.

    `grounded` is True only when all concepts are grounded at D3 with no gaps.
    Unknown concepts pass through as their original names and produce
    ExistenceGaps, causing `grounded` to be False.

    """

    grounded: bool
    resolved: list[str]
    gaps: list[Any]
    trace: EpistemicResponse | None = None

    def __str__(self) -> str:
        """
        Render the full epistemic trace including primitives, pathway, and any gaps.
        """
        lines: list[str] = []
        resolved_str = ", ".join(self.resolved)
        prefix = "Grounded" if self.grounded else "Not grounded"
        lines.append(f"[VRE] {prefix} — {resolved_str}")

        if self.grounded:
            lines.append("")
            lines.append(
                "This is your epistemic trace. The concepts below have been verified at D3 (CONSTRAINTS)."
            )

        if self.trace:
            id_to_name = {p.id: p.name for p in self.trace.result.primitives}
            for primitive in self.trace.result.primitives:
                lines.append("")
                lines.extend(_fmt_primitive(primitive, id_to_name))

            if self.trace.result.pathway:
                seen: set = set()
                deduped = []
                for step in self.trace.result.pathway:
                    key = (step.source_id, step.target_id, step.relation_type)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(step)
                lines.append("")
                lines.append("Pathway:")
                for step in deduped:
                    src = id_to_name.get(step.source_id, str(step.source_id))
                    tgt = id_to_name.get(step.target_id, str(step.target_id))
                    lines.append(
                        f"  {src} —[{step.relation_type.value}@D{step.target_depth.value}]→ {tgt}"
                    )

        if self.gaps:
            lines.append("")
            lines.append("Gaps:")
            for gap in self.gaps:
                lines.append(f"  {_fmt_gap(gap)}")

        if not self.grounded:
            lines.append("")
            lines.append("Cannot execute until knowledge gaps are resolved.")

        return "\n".join(lines)
