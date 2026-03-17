# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Volute Reasoning Engine — decorator-based epistemic enforcement.

Usage::

    from vre import VRE
    from vre.core.graph import PrimitiveRepository

    repo = PrimitiveRepository("neo4j://localhost:7687", "neo4j", "password")
    vre = VRE(repo)
    result = vre.check(["file", "write"])
    print(result.grounded, result.resolved)
"""

from vre.core.graph import PrimitiveRepository
from vre.core.grounding import ConceptResolver, GroundingEngine, GroundingResult
from vre.core.models import DepthLevel, Provenance, ProvenanceSource
from vre.core.policy import Cardinality, PolicyResult
from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.gate import PolicyGate
from vre.learning import (
    CandidateDecision,
    LearningCallback,
    LearningCandidate,
    LearningEngine,
    LearningResult,
)

__all__ = [
    "VRE",
    "PrimitiveRepository",
    "ConceptResolver",
    "GroundingEngine",
    "GroundingResult",
    "DepthLevel",
    "Provenance",
    "ProvenanceSource",
    "Cardinality",
    "PolicyResult",
    "PolicyCallContext",
    "PolicyGate",
    "CandidateDecision",
    "LearningCallback",
    "LearningCandidate",
    "LearningEngine",
    "LearningResult",
]


class VRE:
    """
    Volute Reasoning Engine — public interface.

    Wraps ConceptResolver and GroundingEngine. Depth requirements are
    derived from graph structure; an optional min_depth override lets
    integrators enforce a stricter floor.
    """

    def __init__(self, repository: PrimitiveRepository) -> None:
        """
        Initialize VRE with the given primitive repository.
        """
        self._repo = repository
        self._resolver = ConceptResolver(repository)
        self._engine = GroundingEngine(repository)
        self._learning_engine = LearningEngine(repository)

    def resolve(self, concepts: list[str]) -> list[str]:
        """
        Resolve free-form concept names to canonical primitive names.
        """
        return self._resolver.resolve(concepts)

    def check(
        self,
        concepts: list[str],
        min_depth: DepthLevel | None = None,
    ) -> GroundingResult:
        """
        Ground concepts with graph-derived depth gating.

        Returns a GroundingResult with grounded=True only when all resolved
        concepts are fully grounded with no gaps.

        Parameters
        ----------
        min_depth:
            Optional integrator override — enforces a minimum depth floor
            on all root primitives. Can only raise the floor, never lower it.
        """
        return self._engine.ground(concepts, self._resolver, min_depth=min_depth)

    def learn_all(
        self,
        grounding: GroundingResult,
        callback: LearningCallback,
        concepts: list[str],
        min_depth: DepthLevel | None = None,
    ) -> GroundingResult:
        """
        Iteratively resolve all gaps via the learning loop.

        Processes one gap at a time, re-grounding after each accepted/modified
        candidate. Skipped gaps are excluded from subsequent rounds (the user
        has acknowledged them). Rejected gaps stop the loop entirely.
        Returns the final GroundingResult.
        """
        skipped: set[int] = set()
        with callback:
            while not grounding.grounded and grounding.gaps:
                gap_index = next(
                    (i for i, g in enumerate(grounding.gaps) if i not in skipped),
                    None,
                )
                if gap_index is None:
                    break
                result = self._learning_engine.learn_at(grounding, gap_index, callback)
                if result.decision == CandidateDecision.REJECTED:
                    break
                if result.decision == CandidateDecision.SKIPPED:
                    skipped.add(gap_index)
                    continue
                grounding = self.check(concepts, min_depth=min_depth)
                skipped.clear()
        return grounding

    def check_policy(
        self,
        concepts: list[str] | GroundingResult,
        cardinality: str | None = None,
        call_context: PolicyCallContext | None = None,
    ) -> PolicyResult:
        """
        Evaluate policies for the given concepts.

        `concepts` may be a list of concept names (grounding is run) or a
        pre-computed `GroundingResult` (grounding is skipped).

        `call_context` carries the tool name, grounding result, and the args/
        kwargs of the decorated function so that policy callbacks can make
        domain-specific decisions. Omit when calling outside a guarded context.

        Returns PolicyResult with action PASS, PENDING, or BLOCK.
        """
        if isinstance(concepts, GroundingResult):
            grounding = concepts
        else:
            grounding = self._engine.ground(concepts, self._resolver)

        if grounding.trace is None:
            return PolicyResult(action="PASS")

        card_enum = Cardinality.SINGLE
        if cardinality is not None:
            try:
                card_enum = Cardinality(cardinality)
            except ValueError:
                pass  # unknown string → fall back to SINGLE

        gate = PolicyGate()
        return gate.evaluate(grounding.trace, card_enum, call_context)

