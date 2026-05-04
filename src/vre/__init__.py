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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import UUID

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
from vre.core.models import (
    DepthGap,
    DepthLevel,
    ExistenceGap,
    PrimitiveMetrics,
    Provenance,
    ProvenanceSource,
    ReachabilityGap,
    RelationalGap,
)
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
from vre.tracing import TraceWriter, build_trace_entry

logging.getLogger("vre").addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

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
    "PrimitiveMetrics",
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
        persist_traces: bool = True,
    ) -> None:
        """
        Initialize VRE with the given primitive repository.

        When `agent_key` is provided, it resolves via the persisted registry
        to a stable AgentIdentity stamped on every GroundingResult; `agent_name`
        is used only on first registration. Traces are persisted to daily JSONL
        files under `~/.vre/traces/` when `persist_traces` is True.
        """
        self._repo = repository
        self._resolver = ConceptResolver(repository)
        self._engine = GroundingEngine(repository)
        self._learning_engine = LearningEngine(repository)

        if agent_key is not None:
            self._identity: AgentIdentity | None = AgentRegistry(registry_path).get_or_create(agent_key, name=agent_name)
        else:
            self._identity = None

        self._suppress_trace = False
        self._trace_writer: TraceWriter | None = TraceWriter() if persist_traces else None

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

    @staticmethod
    def _gap_primitive_ids(gaps: list) -> set[UUID]:
        """
        Collect the set of primitive UUIDs that are epistemically deficient.

        RelationalGaps contribute only the target ID — the relationship
        is a pure structural declaration and the source is epistemically
        sound if the edge is visible. The failure belongs to the target
        that lacks the required depth.
        """
        ids: set[UUID] = set()
        for gap in gaps:
            if gap.kind == "RELATIONAL":
                ids.add(gap.target.id)
            else:
                ids.add(gap.primitive.id)
        return ids

    def _update_grounding_metrics(self, result: GroundingResult) -> None:
        """
        Update per-primitive grounding metrics after a `check` call.

        Batch-reads current metrics for all resolved root concepts, computes
        increments in-process, and batch-writes the results. Updates are
        best-effort — failures are logged but never block the caller.
        """
        resolved_lower = {r.lower() for r in result.resolved}
        target_prims = [
            prim for prim in result.get_primitives()
            if prim.name.lower() in resolved_lower
        ]

        current_metrics: dict[UUID, PrimitiveMetrics | None] | None = None
        if target_prims:
            try:
                current_metrics = self._repo.batch_read_metrics([p.id for p in target_prims])
            except Exception:
                logger.warning("Failed to batch-read metrics", exc_info=True)

        updates: dict[UUID, PrimitiveMetrics] = {}
        if current_metrics is not None:
            now = datetime.now(timezone.utc)
            gap_ids = self._gap_primitive_ids(result.gaps)
            for prim in target_prims:
                if prim.id not in current_metrics:
                    continue
                metrics = current_metrics[prim.id] or PrimitiveMetrics()
                if prim.id in gap_ids:
                    metrics.failure_count += 1
                    metrics.last_failed = now
                else:
                    metrics.grounding_count += 1
                    metrics.last_grounded = now
                updates[prim.id] = metrics

        if updates:
            try:
                self._repo.batch_update_metrics(updates)
            except Exception:
                logger.warning("Failed to batch-update metrics for %d primitives", len(updates), exc_info=True)

    def _update_learning_metric(
        self,
        gap: DepthGap | ExistenceGap | RelationalGap | ReachabilityGap,
        decision: CandidateDecision,
    ) -> None:
        """
        Update learning metrics on the primitive targeted by a gap.

        Increments `learning_count` for accepted/modified decisions and
        `rejection_count` for rejected decisions. Skipped decisions are
        ignored. Looks up the primitive by ID first, falling back to name
        for ExistenceGaps where the gap carries a transient ID.
        """
        prim_id: UUID | None = None
        prim_name: str | None = None
        if decision != CandidateDecision.SKIPPED:
            if isinstance(gap, RelationalGap):
                prim_id, prim_name = gap.target.id, gap.target.name
            elif isinstance(gap, (DepthGap, ExistenceGap, ReachabilityGap)):
                prim_id, prim_name = gap.primitive.id, gap.primitive.name

        found = None
        if prim_id is not None:
            found = self._repo.find_by_id(prim_id)
            if found is None and prim_name is not None:
                found = self._repo.find_by_name(prim_name)

        if found is not None:
            metrics = found.metrics or PrimitiveMetrics()
            if decision in (CandidateDecision.ACCEPTED, CandidateDecision.MODIFIED):
                metrics.learning_count += 1
            elif decision == CandidateDecision.REJECTED:
                metrics.rejection_count += 1
            try:
                self._repo.update_metrics(found.id, metrics)
            except Exception:
                logger.warning("Failed to update learning metrics for %r", prim_name, exc_info=True)

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
        concepts are fully grounded with no gaps. `min_depth` is an optional
        integrator override that can only raise the floor, never lower it.
        """
        result = self._stamp_identity(self._engine.ground(concepts, self._resolver, min_depth=min_depth))
        self._update_grounding_metrics(result)
        if self._trace_writer is not None and not self._suppress_trace:
            try:
                self._trace_writer.write(build_trace_entry("check", concepts, result))
            except Exception:
                logger.warning("Failed to persist trace for check()", exc_info=True)
        return result

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
        learning_outcomes: list[LearningResult] = []
        self._suppress_trace = True
        try:
            with callback:
                while not grounding.grounded and grounding.gaps:
                    gap_index = next(
                        (i for i, g in enumerate(grounding.gaps) if i not in skipped),
                        None,
                    )
                    if gap_index is None:
                        break
                    result = self._learning_engine.learn_at(grounding, gap_index, callback)
                    learning_outcomes.append(result)
                    self._update_learning_metric(grounding.gaps[gap_index], result.decision)
                    if result.decision == CandidateDecision.REJECTED:
                        break
                    if result.decision == CandidateDecision.SKIPPED:
                        skipped.add(gap_index)
                        continue
                    self._resolver.invalidate()
                    grounding = self.check(concepts, min_depth=min_depth)
                    skipped.clear()
        finally:
            self._suppress_trace = False

        if self._trace_writer is not None and learning_outcomes:
            try:
                self._trace_writer.write(
                    build_trace_entry("learn", concepts, grounding, learning_outcomes)
                )
            except Exception:
                logger.warning("Failed to persist trace for learn_all()", exc_info=True)

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
            policy_result = PolicyResult(action=PolicyAction.PASS)
        else:
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
                elif on_policy is None:
                    policy_result = PolicyResult(
                        action=PolicyAction.BLOCK,
                        reason="Confirmation required, no handler",
                        violations=pending,
                    )
                elif on_policy(pending):
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

        return policy_result
