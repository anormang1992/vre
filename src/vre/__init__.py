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

import logging
from pathlib import Path
from typing import Callable

from vre.core.graph import PrimitiveRepository
from vre.core.grounding import ConceptResolver, GroundingEngine, GroundingResult
from vre.core.errors import (
    CandidateValidationError,
    CyclicRelationshipError,
    GraphError,
    GraphIntegrityError,
    HydrationError,
    PersistenceError,
    RegistryError,
    ResolutionError,
    VREError,
)
from vre.core.models import DepthLevel, Provenance, ProvenanceSource
from vre.core.policy import Cardinality, PolicyAction, PolicyCallbackResult, PolicyResult, PolicyViolation
from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.gate import PolicyGate
from vre.identity import AgentIdentity, AgentRegistry
from vre.learning import (
    CandidateDecision,
    LearningCallback,
    LearningCandidate,
    LearningEngine,
    LearningResult,
)

logging.getLogger("vre").addHandler(logging.NullHandler())

__all__ = [
    "VRE",
    "AgentIdentity",
    "AgentRegistry",
    "CandidateValidationError",
    "CyclicRelationshipError",
    "GraphError",
    "GraphIntegrityError",
    "HydrationError",
    "PersistenceError",
    "RegistryError",
    "ResolutionError",
    "VREError",
    "PrimitiveRepository",
    "ConceptResolver",
    "GroundingEngine",
    "GroundingResult",
    "DepthLevel",
    "Provenance",
    "ProvenanceSource",
    "Cardinality",
    "PolicyAction",
    "PolicyCallbackResult",
    "PolicyResult",
    "PolicyViolation",
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

    def __init__(
        self,
        repository: PrimitiveRepository,
        agent_key: str | None = None,
        agent_name: str | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        """
        Initialize VRE with the given primitive repository.

        Parameters
        ----------
        repository:
            The Neo4j primitive repository for graph operations.
        agent_key:
            Optional registration key for agent identity. When provided,
            the key is resolved via the persisted registry to a stable
            AgentIdentity whose `agent_id` is stamped on every
            GroundingResult produced by this instance.
        agent_name:
            Optional human-readable name for the agent. Only used on
            first registration; ignored on subsequent calls with the
            same `agent_key`.
        registry_path:
            Path to the agent registry JSON file. Defaults to
            `AgentRegistry`'s built-in default when None.
        """
        self._repo = repository
        self._resolver = ConceptResolver(repository)
        self._engine = GroundingEngine(repository)
        self._learning_engine = LearningEngine(repository)

        if agent_key is not None:
            self._identity: AgentIdentity | None = AgentRegistry(registry_path).get_or_create(agent_key, name=agent_name)
        else:
            self._identity = None

    @property
    def identity(self) -> AgentIdentity | None:
        """
        The agent identity associated with this VRE instance, if any.
        """
        return self._identity

    def _stamp_identity(self, result: GroundingResult) -> GroundingResult:
        """
        Set `agent_id` on the result if this instance has an identity and the result doesn't already have one.
        """
        if self._identity is not None and result.agent_id is None:
            result.agent_id = self._identity.agent_id
        return result

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
        concepts:
            List of free-form concept names to ground.
        min_depth:
            Optional integrator override — enforces a minimum depth floor
            on all root primitives. Can only raise the floor, never lower it.
        """
        return self._stamp_identity(self._engine.ground(concepts, self._resolver, min_depth=min_depth))

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
        on_policy: Callable[[list[PolicyViolation]], bool] | None = None,
    ) -> PolicyResult:
        """
        Evaluate policies for the given concepts.

        `concepts` may be a list of concept names (grounding is run) or a
        pre-computed `GroundingResult` (grounding is skipped).

        `call_context` carries the tool name, grounding result, and the args/
        kwargs of the decorated function so that policy callbacks can make
        domain-specific decisions. Omit when calling outside a guarded context.

        `on_policy` is an optional handler consulted when any violation has
        `requires_confirmation=True`. It receives all violations and returns
        a single bool — True to proceed, False to block. When absent, the
        fail-safe is BLOCK.

        Returns PolicyResult with action PASS or BLOCK (never PENDING).
        """
        if isinstance(concepts, GroundingResult):
            grounding = concepts
        else:
            grounding = self._stamp_identity(self._engine.ground(concepts, self._resolver))

        if grounding.trace is None:
            return PolicyResult(action=PolicyAction.PASS)

        card_enum: Cardinality | None = None
        if cardinality is not None:
            try:
                card_enum = Cardinality(cardinality)
            except ValueError:
                card_enum = None  # unknown → fire all policies

        gate = PolicyGate()
        violations = gate.evaluate(grounding.trace, card_enum, call_context)

        if not violations:
            policy_result = PolicyResult(action=PolicyAction.PASS)
        else:
            hard_blocks = [v for v in violations if not v.requires_confirmation]
            pending = [v for v in violations if v.requires_confirmation]

            # Hard blocks do not consult on_policy — they are immediate BLOCKs with their own messages
            if hard_blocks:
                messages = "; ".join(v.message for v in hard_blocks)
                policy_result = PolicyResult(
                    action=PolicyAction.BLOCK,
                    reason=messages,
                    violations=violations,
                )
            else:
                # Only confirmation-required violations remain — consult on_policy
                if on_policy is not None:
                    if on_policy(pending):
                        policy_result = PolicyResult(
                            action=PolicyAction.PASS,
                            violations=pending,
                        )
                    else:
                        policy_result = PolicyResult(
                            action=PolicyAction.BLOCK,
                            reason="User declined",
                            violations=pending,
                        )
                else:
                    policy_result = PolicyResult(
                        action=PolicyAction.BLOCK,
                        reason="Confirmation required, no handler",
                        violations=pending,
                    )

        return policy_result
