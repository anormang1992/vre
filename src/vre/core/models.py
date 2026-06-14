# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Core epistemic models for the Volute Reasoning Engine.
"""

from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Annotated, Any, Literal, NamedTuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DepthLevel(IntEnum):
    """
    Canonical depth levels for epistemic grounding.
    """

    EXISTENCE = 0
    IDENTITY = 1
    CAPABILITIES = 2
    CONSTRAINTS = 3
    IMPLICATIONS = 4


def format_depth_label(level: "DepthLevel | None") -> str:
    """
    Render a DepthLevel as "D{value} {NAME}", or "none" when level is None.
    """
    return "none" if level is None else f"D{level.value} {level.name}"


class RelationType(str, Enum):
    """
    Constrained relationship types between primitives.
    """

    APPLIES_TO = "APPLIES_TO"
    REQUIRES = "REQUIRES"
    CONSTRAINED_BY = "CONSTRAINED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    INCLUDES = "INCLUDES"


TRANSITIVE_RELATION_TYPES: frozenset[RelationType] = frozenset(
    {
        RelationType.REQUIRES,
        RelationType.CONSTRAINED_BY,
        RelationType.DEPENDS_ON,
    }
)


class ProvenanceSource(str, Enum):
    """
    Origin category for knowledge in the epistemic graph.

    Provenance is genealogy -- who drafted the content -- not a trust
    gradient. As a knowledge linter, VRE only ever persists knowledge that
    a human has attested at the persistence boundary, so the two categories
    differ solely in who drafted the content:

    - AUTHORED: a human drafted the content from scratch.
    - LEARNED: an agent proposed the content and a human approved it at
      the point of persistence.

    Both are human-attested by construction.
    """

    AUTHORED = "authored"
    LEARNED = "learned"


class Provenance(BaseModel):
    """
    Structured provenance record for epistemic knowledge.
    """

    source: ProvenanceSource
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: str | None = None


class PrimitiveMetrics(BaseModel):
    """
    Aggregate usage counters for a primitive node.
    """

    last_grounded: datetime | None = None
    last_failed: datetime | None = None
    grounding_count: int = 0
    failure_count: int = 0

    @property
    def last_exercised(self) -> datetime | None:
        """
        The most recent time this primitive was exercised in any way.
        """
        if self.last_grounded and self.last_failed:
            result = max(self.last_grounded, self.last_failed)
        else:
            result = self.last_grounded or self.last_failed
        return result


class Relatum(BaseModel):
    """
    Directional, typed, depth-aware relationship.
    """

    relation_type: RelationType
    target_id: UUID
    target_depth: DepthLevel
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance | None = None

    def validate_provenance(self, context: str = "") -> None:
        """
        Raise ValueError if provenance is missing.
        """
        if self.provenance is None:
            raise ValueError(
                f"{context}relatum {self.relation_type.value} → "
                f"{self.target_id} is missing provenance"
            )


class Depth(BaseModel):
    """
    Knowledge at a specific depth level.
    """

    level: DepthLevel
    properties: dict[str, Any] = Field(default_factory=dict)
    relata: list[Relatum] = Field(default_factory=list)
    provenance: Provenance | None = None

    def validate_provenance(self, context: str = "") -> None:
        """
        Raise ValueError if provenance is missing on this depth or any of its relata.
        """
        if self.provenance is None:
            raise ValueError(
                f"{context}depth D{self.level.value} ({self.level.name}) "
                f"is missing provenance"
            )
        for relatum in self.relata:
            relatum.validate_provenance(
                context=f"{context}depth D{self.level.value} "
            )


class Primitive(BaseModel):
    """
    A conceptual primitive in the epistemic graph.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    depths: list[Depth] = Field(default_factory=list)
    provenance: Provenance | None = None
    metrics: PrimitiveMetrics | None = None

    @property
    def contiguous_max_depth(self) -> DepthLevel | None:
        """
        Highest DepthLevel forming a contiguous chain from D0, or None if no depths.
        """
        levels = {d.level for d in self.depths}
        result: DepthLevel | None = None
        for level in sorted(DepthLevel):
            if level not in levels:
                break
            result = level
        return result

    def validate_provenance(self) -> None:
        """
        Raise ValueError if provenance is missing on this primitive, any depth, or any relatum.
        """
        if self.provenance is None:
            raise ValueError(
                f"Primitive '{self.name}' is missing provenance"
            )
        for depth in self.depths:
            depth.validate_provenance(context=f"Primitive '{self.name}' ")


class EpistemicQuery(BaseModel):
    """
    Structured query submitted to VRE.
    """

    concept_ids: list[UUID]


def _missing_depth_levels(
    present: set[DepthLevel], current: DepthLevel | None, required: DepthLevel
) -> list[DepthLevel]:
    """
    The levels in (current, required] that are not already present — the exact
    holes a fill must author to close the gap.

    Excludes levels that already exist, including ones detached above the
    contiguous max (e.g. a dormant authored D4 over a D1 chain): those are
    reactivated by contiguity, never re-authored, so a fill is never invited to
    overwrite them.
    """
    floor = -1 if current is None else int(current)
    return [
        level
        for level in sorted(DepthLevel)
        if floor < int(level) <= int(required) and level not in present
    ]


class DepthGap(BaseModel):
    """
    Surfaced when a primitive lacks the depth required for execution.
    """

    kind: Literal["DEPTH"] = "DEPTH"
    primitive: Primitive
    required_depth: DepthLevel
    current_depth: DepthLevel | None

    @property
    def missing_levels(self) -> list[DepthLevel]:
        """The specific levels a fill must author to close this gap (the holes)."""
        present = {d.level for d in self.primitive.depths}
        return _missing_depth_levels(present, self.current_depth, self.required_depth)


class ExistenceGap(BaseModel):
    """
    Surfaced when a concept is not found in the graph at all.
    """

    kind: Literal["EXISTENCE"] = "EXISTENCE"
    primitive: Primitive


class RelationalGap(BaseModel):
    """
    Surfaced when an edge's target is not grounded deeply enough to satisfy
    the edge's declared target_depth requirement (Phase 3 only).
    """

    kind: Literal["RELATIONAL"] = "RELATIONAL"
    source: Primitive
    target: Primitive
    required_depth: DepthLevel
    current_depth: DepthLevel | None

    @property
    def missing_levels(self) -> list[DepthLevel]:
        """The levels a fill must author on the target to close this gap (the holes)."""
        present = {d.level for d in self.target.depths}
        return _missing_depth_levels(present, self.current_depth, self.required_depth)


class ReachabilityGap(BaseModel):
    """
    Surfaced when a concept is not connected to the other submitted concepts
    via any edge path in the collected subgraph.
    """

    kind: Literal["REACHABILITY"] = "REACHABILITY"
    primitive: Primitive


KnowledgeGap = Annotated[
    DepthGap | ExistenceGap | RelationalGap | ReachabilityGap,
    Field(discriminator="kind"),
]


def gap_primitive_ids(gaps: Iterable[KnowledgeGap]) -> set[UUID]:
    """
    Collect the IDs of the primitives each gap is "about".

    RelationalGaps point at the target — the visible edge means the source
    is epistemically sound; the failure belongs to the under-grounded target.
    All other gap kinds point at .primitive.
    """
    ids: set[UUID] = set()
    for gap in gaps:
        if gap.kind == "RELATIONAL":
            ids.add(gap.target.id)
        else:
            ids.add(gap.primitive.id)
    return ids


class EpistemicStep(BaseModel):
    """
    A single traversal step in the epistemic pathway.
    """

    source_id: UUID
    target_id: UUID
    relation_type: RelationType
    source_depth: DepthLevel
    target_depth: DepthLevel


class ResolvedSubgraph(NamedTuple):
    """
    Raw subgraph returned by repository traversal.
    """

    roots: list[Primitive]
    nodes: list[Primitive]
    edges: list[EpistemicStep]


class EpistemicResult(BaseModel):
    """
    The epistemic envelope. A self-contained subgraph of resolved knowledge.
    """

    primitives: list[Primitive]
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    pathway: list[EpistemicStep] = Field(default_factory=list)


class EpistemicResponse(BaseModel):
    """
    Structured result returned by VRE.
    """

    query: EpistemicQuery
    result: EpistemicResult
