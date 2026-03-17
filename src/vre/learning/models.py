# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Candidate models for the VRE auto-learning loop.

Candidates carry only what's *new* — the agent's proposed knowledge. All
context (primitive IDs, existing depths, required depths) lives on the gap
itself, which the engine already has access to.

This keeps the models lightweight and compatible with LLM structured output.
"""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from vre.core.models import DepthLevel, RelationType


class CandidateDecision(str, Enum):
    """
    Outcome of a learning candidate review.

    ACCEPTED — agent proposal persisted as-is (provenance: learned).
    MODIFIED — user refined the proposal before persistence (provenance: conversational).
    REJECTED — candidate discarded, nothing persisted. Stops the learning loop entirely.
    SKIPPED — candidate intentionally dismissed (e.g. edge absence is deliberate
              enforcement). The loop continues to the next gap.
    """

    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"
    SKIPPED = "skipped"


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


class DepthCandidate(BaseModel):
    """
    Proposal for a DepthGap — concept exists but lacks required depth.

    The agent fills in the missing depth levels. The engine pulls
    primitive ID and existing depths from the gap itself.
    """

    kind: Literal["DEPTH"] = "DEPTH"
    new_depths: list[ProposedDepth] = Field(default_factory=list)


class RelationalCandidate(BaseModel):
    """
    Proposal for a RelationalGap — edge target not grounded deeply enough.

    The agent fills in the missing depth levels on the target. The engine
    pulls target ID and existing depths from the gap itself.
    """

    kind: Literal["RELATIONAL"] = "RELATIONAL"
    new_depths: list[ProposedDepth] = Field(default_factory=list)


class ReachabilityCandidate(BaseModel):
    """
    Proposal for a ReachabilityGap — concept not connected to other concepts.

    Focused solely on edge placement. The agent proposes which target to
    connect to, the relation type, and the depth levels for the edge.
    The engine resolves target_name to an ID from the grounding trace and
    scaffolds any missing depths as stubs. If the scaffolded depths lack
    required knowledge, re-grounding will surface them as depth or
    relational gaps for the loop to handle naturally.

    If the absence is intentional enforcement, the user skips instead of
    filling this in.
    """

    kind: Literal["REACHABILITY"] = "REACHABILITY"
    target_name: str | None = Field(
        default=None,
        description="Name of the target concept to connect to.",
    )
    relation_type: RelationType | None = Field(
        default=None,
        description="The type of relationship between source and target.",
    )
    source_depth_level: DepthLevel | None = Field(
        default=None,
        description="Depth level on the source where the edge is placed. Determines when the agent can reason about this relationship.",
    )
    target_depth_level: DepthLevel | None = Field(
        default=None,
        description="Minimum depth required on the target for the edge to resolve.",
    )


LearningCandidate = Annotated[
    ExistenceCandidate | DepthCandidate | RelationalCandidate | ReachabilityCandidate,
    Field(discriminator="kind"),
]


class LearningResult(BaseModel):
    """
    Outcome of a single learning round.
    """

    decision: CandidateDecision
    candidate: LearningCandidate
