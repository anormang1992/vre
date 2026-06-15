# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
GroundingResult — the public result type returned by VRE grounding checks.
"""

from uuid import UUID

from pydantic import BaseModel

from vre.core.models import (
    Depth,
    EpistemicResponse,
    EpistemicStep,
    KnowledgeGap,
    Primitive,
    Relatum,
    format_depth_label,
)


# ── Private formatting helpers ────────────────────────────────────────────────

def _fmt_gap(gap: KnowledgeGap) -> str:
    """
    Format a KnowledgeGap as a human-readable string.
    """
    kind = gap.kind
    if kind == "EXISTENCE":
        out = f"EXISTENCE: '{gap.primitive.name}' is not in the knowledge graph"
    elif kind == "DEPTH":
        curr = format_depth_label(gap.current_depth)
        req = format_depth_label(gap.required_depth)
        out = f"DEPTH: '{gap.primitive.name}' known to {curr}, requires {req}"
    elif kind == "RELATIONAL":
        curr = format_depth_label(gap.current_depth)
        req = format_depth_label(gap.required_depth)
        out = (
            f"RELATIONAL: '{gap.source.name}' → '{gap.target.name}' "
            f"requires {req} on target, found {curr}"
        )
    elif kind == "REACHABILITY":
        out = f"REACHABILITY: '{gap.primitive.name}' is not connected to other concepts"
    else:
        out = f"UNKNOWN: {gap}"
    return out


def _fmt_relatum(r: Relatum, id_to_name: dict[UUID, str]) -> list[str]:
    """
    Format a single Relatum as display lines, including metadata and provenance.
    """
    target_name = id_to_name.get(r.target_id, str(r.target_id))
    lines = [f"      → {target_name}  [{r.relation_type.value}, target@D{r.target_depth.value}]"]
    if r.metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in r.metadata.items())
        lines.append(f"        metadata: {meta_str}")
    if r.provenance:
        date_str = r.provenance.created_at.strftime("%Y-%m-%d")
        lines.append(f"        provenance: {r.provenance.source.value} ({date_str})")
    return lines


def _fmt_depth(depth: Depth, id_to_name: dict[UUID, str]) -> list[str]:
    """
    Format a single Depth level as display lines, including its relata.
    """
    prov_tag = f"  [{depth.provenance.source.value}]" if depth.provenance else ""
    lines = [f"  {format_depth_label(depth.level)}{prov_tag}"]
    if depth.properties:
        lines.append("    properties:")
        for k, v in depth.properties.items():
            lines.append(f"      {k}: {v}")
    if depth.relata:
        lines.append("    relata:")
        for r in depth.relata:
            lines.extend(_fmt_relatum(r, id_to_name))
    return lines


def _fmt_primitive(primitive: Primitive, id_to_name: dict[UUID, str]) -> list[str]:
    """
    Format a Primitive and all its depths as display lines.
    """
    name = primitive.name
    header = f"═══ {name} {'═' * max(0, 50 - len(name))}"
    lines = [header]

    if primitive.provenance:
        date_str = primitive.provenance.created_at.strftime("%Y-%m-%d")
        lines.append(f"  provenance: {primitive.provenance.source.value} ({date_str})")

    if primitive.depths:
        # Compact single-line format when all depths have no properties or relata
        all_empty = all(d.is_empty for d in primitive.depths)
        if all_empty:
            labels = [
                format_depth_label(d.level)
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

    `grounded` is True only when all concepts are grounded with no gaps.
    Unknown concepts pass through as their original names and produce
    ExistenceGaps, causing `grounded` to be False.

    """

    grounded: bool
    resolved: list[str]
    gaps: list[KnowledgeGap]
    trace: EpistemicResponse | None = None
    agent_id: UUID | None = None

    def get_primitives(self) -> list[Primitive]:
        """
        Return the primitives from the underlying trace, or [] when absent.
        """
        return self.trace.result.primitives if self.trace is not None else []

    def get_pathway_steps(self) -> list[EpistemicStep]:
        """
        Return the pathway steps from the underlying trace, or [] when absent.
        """
        return self.trace.result.pathway if self.trace is not None else []

    def __str__(self) -> str:
        """
        Render the full epistemic trace including primitives, pathway, and any gaps.
        """
        lines: list[str] = []
        resolved_str = ", ".join(self.resolved)
        prefix = "Grounded" if self.grounded else "Not grounded"
        agent_tag = f"  (agent: {self.agent_id})" if self.agent_id else ""
        lines.append(f"[VRE] {prefix} — {resolved_str}{agent_tag}")

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
