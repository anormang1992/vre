# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
template_for_gap — converts knowledge gaps into structured candidate templates.

VRE presents the shape of what is needed; the agent fills in the content. VRE
resolves *which* levels are missing (structure); the integrator supplies *what
they contain* (properties). Context (primitive IDs, existing depths) lives on the
gap; the candidate carries only the new knowledge to be proposed.
"""

from vre.core.models import (
    DepthGap,
    DepthLevel,
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

    Each template is pre-seeded with the structural slots VRE can resolve, leaving
    only `properties` for the integrator to fill:

    - ExistenceGap → the concept name and a D1 (IDENTITY) slot (the only level an
      existence fill ever authors; D0 is auto-generated on persist).
    - DepthGap / RelationalGap → one empty `ProposedDepth` per `gap.missing_levels`,
      the exact holes to author — so a fill is never invited to re-author (and
      clobber) a level that is already grounded.
    - ReachabilityGap → empty; edge placement carries no depth content.
    """
    candidate: LearningCandidate
    match gap:
        case ExistenceGap():
            candidate = ExistenceCandidate(
                name=gap.primitive.name,
                d1=ProposedDepth(level=DepthLevel.IDENTITY),
            )
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
