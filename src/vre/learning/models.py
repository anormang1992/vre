# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Candidate models for VRE learning.

Candidates carry only what's *new* — the integrator's proposed knowledge.
All context (primitive IDs, existing depths, required depths) lives on
the gap itself, which the engine already has access to.

This keeps the models lightweight and compatible with LLM structured output.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from vre.core.errors import CandidateValidationError
from vre.core.models import DepthLevel, KnowledgeGap, RelationType


class ProposedDepth(BaseModel):
    """
    A depth level proposed by the agent during learning.

    Unlike the full Depth model, this carries only what the agent should
    fill: the level and descriptive properties. Relata and provenance are
    structural concerns handled by the engine during persistence.
    """

    level: DepthLevel
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Descriptive attributes intrinsic to this concept at this depth level. "
            "Each key names an attribute, each value describes it in natural language."
        ),
    )


class ExistenceCandidate(BaseModel):
    """
    Proposal for an ExistenceGap — concept not found in the graph.

    The agent fills in D1 (identity). On acceptance, D0 is generated
    automatically — the act of accepting *is* the D0 confirmation.
    """

    kind: Literal["EXISTENCE"] = "EXISTENCE"
    name: str
    d1: ProposedDepth | None = None

    def validate_for_gap(self, gap: KnowledgeGap) -> None:
        if self.d1 is None:
            raise CandidateValidationError(
                f"ExistenceCandidate '{self.name}' is missing D1 (identity)"
            )


class DepthCandidate(BaseModel):
    """
    Proposal for a DepthGap — concept exists but lacks required depth.

    The agent fills in the missing depth levels. The engine pulls
    primitive ID and existing depths from the gap itself.
    """

    kind: Literal["DEPTH"] = "DEPTH"
    new_depths: list[ProposedDepth] = Field(default_factory=list)

    def validate_for_gap(self, gap: KnowledgeGap) -> None:
        if not self.new_depths:
            raise CandidateValidationError(
                f"DepthCandidate for '{gap.primitive.name}' has no new depths"
            )


class RelationalCandidate(BaseModel):
    """
    Proposal for a RelationalGap — edge target not grounded deeply enough.

    The agent fills in the missing depth levels on the target. The engine
    pulls target ID and existing depths from the gap itself.
    """

    kind: Literal["RELATIONAL"] = "RELATIONAL"
    new_depths: list[ProposedDepth] = Field(default_factory=list)

    def validate_for_gap(self, gap: KnowledgeGap) -> None:
        if not self.new_depths:
            raise CandidateValidationError(
                f"RelationalCandidate for '{gap.target.name}' has no new depths"
            )


class ReachabilityCandidate(BaseModel):
    """
    Proposal for a ReachabilityGap — concept not connected to other concepts.

    Focused solely on edge placement. The integrator proposes both the
    source and target of the edge, the relation type, and the depth
    levels. At least one of ``source_name`` or ``target_name`` must
    match the gap's primitive — the edge must fix *this* disconnection.
    The direction is up to the integrator.

    Source and target must already have the required depth levels before
    edge placement; if they don't, ``learn_gap`` raises
    ``CandidateValidationError`` telling the integrator which DepthGaps
    to fill first.
    """

    kind: Literal["REACHABILITY"] = "REACHABILITY"
    source_name: str | None = Field(
        default=None,
        description="Name of the source concept (edge originates here).",
    )
    target_name: str | None = Field(
        default=None,
        description="Name of the target concept (edge points here).",
    )
    relation_type: RelationType | None = Field(
        default=None,
        description="The type of relationship between source and target.",
    )
    source_depth_level: DepthLevel | None = Field(
        default=None,
        description="Depth level on the source where the edge is placed.",
    )
    target_depth_level: DepthLevel | None = Field(
        default=None,
        description="Minimum depth required on the target for the edge to resolve.",
    )

    def validate_for_gap(self, gap: KnowledgeGap) -> None:
        if self.source_name is None or self.target_name is None:
            raise CandidateValidationError(
                f"ReachabilityCandidate for '{gap.primitive.name}' is missing "
                f"source_name or target_name"
            )
        if self.relation_type is None:
            raise CandidateValidationError(
                f"ReachabilityCandidate for '{gap.primitive.name}' is missing "
                f"relation_type"
            )
        if self.source_depth_level is None or self.target_depth_level is None:
            raise CandidateValidationError(
                f"ReachabilityCandidate for '{gap.primitive.name}' is missing "
                f"source_depth_level or target_depth_level"
            )
        gap_name = gap.primitive.name.lower()
        if (
            self.source_name.lower() != gap_name
            and self.target_name.lower() != gap_name
        ):
            raise CandidateValidationError(
                f"ReachabilityCandidate must reference the gapped primitive "
                f"'{gap.primitive.name}' as either source or target, "
                f"got source='{self.source_name}', target='{self.target_name}'"
            )


LearningCandidate = Annotated[
    ExistenceCandidate | DepthCandidate | RelationalCandidate | ReachabilityCandidate,
    Field(discriminator="kind"),
]
