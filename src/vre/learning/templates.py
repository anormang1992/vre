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
    ReachabilityCandidate,
    RelationalCandidate,
)


def template_for_gap(gap: KnowledgeGap) -> LearningCandidate:
    """
    Build the candidate template that matches the given gap kind.
    """
    candidate: LearningCandidate
    match gap:
        case ExistenceGap():
            candidate = ExistenceCandidate(name=gap.primitive.name)
        case DepthGap():
            candidate = DepthCandidate()
        case RelationalGap():
            candidate = RelationalCandidate()
        case ReachabilityGap():
            candidate = ReachabilityCandidate()
        case _:
            raise ValueError(f"Unknown gap type: {type(gap)}")
    return candidate
