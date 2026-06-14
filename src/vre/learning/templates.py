# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
template_for_gap — converts knowledge gaps into structured candidate templates.

VRE presents the shape of what is needed; the agent fills in the content.
Context (primitive IDs, existing depths) lives on the gap; the candidate
carries only the new knowledge to be proposed. ExistenceGap pre-fills the
concept name; all other candidates start empty since context lives on the
gap itself.
"""

from vre.core.models import (
    DepthGap,
    ExistenceGap,
    KnowledgeGap,
    ReachabilityGap,
    RelationalGap,
)
from vre.learning.models import (
    DepthCandidate,
    ExistenceCandidate,
    LearningCandidate,
    ProposedDepth,
    ReachabilityCandidate,
    RelationalCandidate,
)


def template_for_gap(gap: KnowledgeGap) -> LearningCandidate:
    """
    Build the candidate template that matches the given gap kind.

    For depth and relational gaps the template is pre-seeded with one empty
    `ProposedDepth` per `gap.missing_levels` — the exact holes to author. The
    integrator fills in only the `properties`; VRE has already resolved *which*
    levels are missing, so a fill is never invited to re-author (and clobber) a
    level that is already grounded.
    """
    candidate: LearningCandidate
    match gap:
        case ExistenceGap():
            candidate = ExistenceCandidate(name=gap.primitive.name)
        case DepthGap():
            candidate = DepthCandidate(
                new_depths=[ProposedDepth(level=level) for level in gap.missing_levels]
            )
        case RelationalGap():
            candidate = RelationalCandidate(
                new_depths=[ProposedDepth(level=level) for level in gap.missing_levels]
            )
        case ReachabilityGap():
            candidate = ReachabilityCandidate()
        case _:
            raise ValueError(f"Unknown gap type: {type(gap)}")
    return candidate
