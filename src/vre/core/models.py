# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Core epistemic models for the Volute Reasoning Engine.
"""

from enum import Enum, IntEnum
from typing import Annotated, Any, Literal, NamedTuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from vre.core.policy.models import Policy


class DepthLevel(IntEnum):
    """
    Canonical depth levels for epistemic grounding.
    """

    EXISTENCE = 0
    IDENTITY = 1
    CAPABILITIES = 2
    CONSTRAINTS = 3
    IMPLICATIONS = 4


class RelationType(str, Enum):
    """
    Constrained relationship types between primitives.
    """

    APPLIES_TO = "APPLIES_TO"
    REQUIRES = "REQUIRES"
    CONSTRAINED_BY = "CONSTRAINED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    INCLUDES = "INCLUDES"


class Relatum(BaseModel):
    """
    Directional, typed, depth-aware relationship.
    """

    relation_type: RelationType
    target_id: UUID
    target_depth: DepthLevel
    metadata: dict[str, Any] = Field(default_factory=dict)
    policies: list[Policy] = Field(default_factory=list)


class Depth(BaseModel):
    """
    Knowledge at a specific depth level.
    """

    level: DepthLevel
    properties: dict[str, Any] = Field(default_factory=dict)
    relata: list[Relatum] = Field(default_factory=list)


class Primitive(BaseModel):
    """
    A conceptual primitive in the epistemic graph.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    depths: list[Depth] = Field(default_factory=list)


class EpistemicQuery(BaseModel):
    """
    Structured query submitted to VRE.
    """

    concept_ids: list[UUID]


class DepthGap(BaseModel):
    """
    Surfaced when a primitive lacks the depth required for execution.
    """

    kind: Literal["DEPTH"] = "DEPTH"
    primitive: Primitive
    required_depth: DepthLevel
    current_depth: DepthLevel | None


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
