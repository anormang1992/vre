# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
TemplateFactory — converts knowledge gaps into structured candidate templates.

VRE presents the shape of what is needed; the agent fills in the content.
Context (primitive IDs, existing depths) lives on the gap; the candidate
carries only the new knowledge to be proposed.
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


class TemplateFactory:
    """
    Converts typed knowledge gaps into candidate templates.

    ExistenceGap pre-fills the concept name; all other candidates start
    empty since context lives on the gap itself.
    """

    @staticmethod
    def from_gap(gap: KnowledgeGap) -> LearningCandidate:
        candidate: LearningCandidate
        if isinstance(gap, ExistenceGap):
            candidate = ExistenceCandidate(name=gap.primitive.name)
        elif isinstance(gap, DepthGap):
            candidate = DepthCandidate()
        elif isinstance(gap, RelationalGap):
            candidate = RelationalCandidate()
        elif isinstance(gap, ReachabilityGap):
            candidate = ReachabilityCandidate()
        else:
            raise ValueError(f"Unknown gap type: {type(gap)}")
        return candidate
