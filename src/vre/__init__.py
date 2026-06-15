# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Volute Reasoning Engine — decorator-based epistemic enforcement.

Usage::

    from vre import VRE, SQLiteRepository

    repo = SQLiteRepository()  # defaults to ~/.vre/graph.db; pass ":memory:" for testing
    vre = VRE(repo)
    result = vre.check(["file", "write"])
    print(result.grounded, result.resolved)
"""

import logging
from pathlib import Path
from typing import Callable
from uuid import UUID

from vre.core.errors import (
    CandidateValidationError,
    CyclicRelationshipError,
    GapResolvedError,
    GraphError,
    GraphIntegrityError,
    HydrationError,
    PersistenceError,
    PolicyPlacementError,
    ProvenanceError,
    RegistryError,
    VREError,
)
from vre.core.backends import Repository, SQLiteRepository

try:
    from vre.core.backends import Neo4jRepository
except ImportError:
    pass
from vre.core.grounding import GroundingEngine, GroundingResult
from vre.core.models import (
    DepthLevel,
    PrimitiveMetrics,
    Provenance,
    ProvenanceSource,
    RelationType,
    format_depth_label,
)
from vre.core.policy import Cardinality, PolicyAction, PolicyCallbackResult, PolicyResult, PolicyViolation
from vre.core.policy.callback import (
    GroundingContext,
    PolicyCallContext,
    ToolCallContext,
    TriggeringEdge,
)
from vre.core.policy.gate import PolicyGate
from vre.core.policy.registry import (
    _DEFAULT_REGISTRY,
    OrphanedPlacement,
    PolicyPlacement,
    PolicyRegistry,
    policy_callback,
    register_policy,
)
from vre.guard import GuardBlock, vre_guard
from vre.identity import AgentIdentity, AgentRegistry
from vre.learning import LearningEngine, template_for_gap
from vre.metrics import MetricsManager
from vre.tracing import TraceManager, TraceWriter

logging.getLogger("vre").addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

__all__ = [
    "VRE",
    "vre_guard",
    "GuardBlock",
    "AgentIdentity",
    "AgentRegistry",
    "CandidateValidationError",
    "CyclicRelationshipError",
    "GapResolvedError",
    "GraphError",
    "GraphIntegrityError",
    "HydrationError",
    "PersistenceError",
    "PolicyPlacementError",
    "ProvenanceError",
    "RegistryError",
    "VREError",
    "Neo4jRepository",
    "Repository",
    "SQLiteRepository",
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
    "ToolCallContext",
    "GroundingContext",
    "TriggeringEdge",
    "PolicyGate",
    "PolicyRegistry",
    "OrphanedPlacement",
    "policy_callback",
    "register_policy",
    "LearningEngine",
    "template_for_gap",
]


class VRE:
    """
    Volute Reasoning Engine — public interface.

    Wraps GroundingEngine. Depth requirements are
    derived from graph structure; an optional min_depth override lets
    integrators enforce a stricter floor.
    """

    def __init__(
        self,
        repository: Repository,
        agent_key: str | None = None,
        agent_name: str | None = None,
        registry_path: str | Path | None = None,
        persist_traces: bool = True,
        policy_registry: PolicyRegistry | None = None,
        validate_policies: bool = True,
        expect_policies: int | None = None,
    ) -> None:
        """
        Initialize VRE with the given primitive repository.

        When `agent_key` is provided, it resolves via the persisted registry
        to a stable AgentIdentity stamped on every GroundingResult; `agent_name`
        is used only on first registration. Traces are persisted to daily JSONL
        files under `~/.vre/traces/` when `persist_traces` is True.

        Code-resident policies are read from `policy_registry` (defaults to the
        module-global registry that `policy_callback` / `register_policy` write to at
        import time). At construction VRE logs the registered policy keys, optionally
        asserts the count equals `expect_policies`, validates that every declared
        placement resolves to an APPLIES_TO edge in the graph — raising
        `PolicyPlacementError` unless `validate_policies=False` — and then freezes the
        registry (no policy may be registered after construction). With
        `validate_policies=True` (the default) the freeze additionally guarantees the
        invariant "everything enforced was validated"; with it disabled the registry is
        frozen-but-unvalidated and you own calling `validate_policy_placements()`.
        Import policy modules before constructing VRE.
        """
        self._repo = repository
        self._engine = GroundingEngine(repository)
        self._learning_engine = LearningEngine(repository)
        self._metrics = MetricsManager(repository)
        self._traces = TraceManager(TraceWriter() if persist_traces else None)
        self._policy_registry = policy_registry if policy_registry is not None else _DEFAULT_REGISTRY

        if agent_key is not None:
            self._identity: AgentIdentity | None = AgentRegistry(registry_path).get_or_create(agent_key, name=agent_name)
        else:
            self._identity = None

        self._configure_policies(validate_policies, expect_policies)

    @property
    def identity(self) -> AgentIdentity | None:
        """
        The agent identity associated with this VRE instance, if any.
        """
        return self._identity

    @property
    def learning_engine(self) -> LearningEngine:
        """
        The learning engine for validating and persisting candidate fills.
        """
        return self._learning_engine

    def _stamp_identity(self, result: GroundingResult) -> GroundingResult:
        """
        Set `agent_id` on the result if this instance has an identity and the result doesn't already have one.
        """
        if self._identity is not None and result.agent_id is None:
            result.agent_id = self._identity.agent_id
        return result

    def _has_applies_to_edge(self, source, target_id: UUID, source_depth: DepthLevel) -> bool:
        """
        Whether `source` has an APPLIES_TO relatum to `target_id` at `source_depth`.
        """
        return any(
            relatum.relation_type == RelationType.APPLIES_TO and relatum.target_id == target_id
            for depth in source.depths if depth.level == source_depth
            for relatum in depth.relata
        )

    def _placement_gap(self, placement: PolicyPlacement) -> str | None:
        """
        Return why a placement's edge is missing, or None if a matching APPLIES_TO
        edge exists. Resolves source and target once via the repository (the canonical
        case-insensitive name match) and compares by id — no per-relatum lookup.
        """
        source = self._repo.find_by_name(placement.source)
        target = self._repo.find_by_name(placement.target)
        if source is None:
            gap: str | None = "source primitive not found"
        elif target is None:
            gap = "target primitive not found"
        elif self._has_applies_to_edge(source, target.id, placement.source_depth):
            gap = None
        else:
            gap = "no APPLIES_TO edge at this source depth"
        return gap

    def validate_policy_placements(self) -> list[OrphanedPlacement]:
        """
        Return the declared placements whose APPLIES_TO edge is absent from the graph.

        A non-empty result means a declared gate would protect nothing — the
        dangerous, otherwise-silent case (a typo, missing knowledge, or a wrong
        depth). Each placement's (source, target, source_depth) is resolved against
        the repository.
        """
        orphans: list[OrphanedPlacement] = []
        for placement in self._policy_registry.iter_placements():
            reason = self._placement_gap(placement)
            if reason is not None:
                orphans.append(OrphanedPlacement(
                    key=placement.key,
                    name=placement.policy.name,
                    source=placement.source,
                    target=placement.target,
                    source_depth=placement.source_depth,
                    reason=reason,
                ))
        return orphans

    def _configure_policies(self, validate_policies: bool, expect_policies: int | None) -> None:
        """
        Log the registered policies, assert the expected count, validate placements
        against the graph (fail loud unless opted out), then freeze the registry and
        cache the frozen gate + keys (the registry is immutable hereafter).
        """
        keys = self._policy_registry.keys()
        logger.info("VRE policy registry: %d registered %s", len(keys), keys)
        if expect_policies is not None and len(keys) != expect_policies:
            raise VREError(
                f"expected {expect_policies} registered policy/policies, found {len(keys)}: {keys}"
            )
        if validate_policies:
            orphans = self.validate_policy_placements()
            if orphans:
                lines = "\n".join(
                    f"  - {o.name!r} ({o.key}): {o.source}->{o.target} "
                    f"@ {format_depth_label(o.source_depth)} [{o.reason}]"
                    for o in orphans
                )
                raise PolicyPlacementError(
                    f"{len(orphans)} declared policy placement(s) reference edges absent "
                    f"from the graph:\n{lines}"
                )
        self._policy_registry.freeze()
        # Registry is immutable now — cache the gate and key list rather than rebuild per call.
        self._policy_keys = self._policy_registry.keys()
        self._policy_gate = PolicyGate(self._policy_registry)

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
        result = self._stamp_identity(self._engine.ground(concepts, min_depth=min_depth))
        self._metrics.update_grounding(result)
        self._traces.write_check(concepts, result, self._policy_keys)
        return result

    def check_policy(
        self,
        concepts: list[str] | GroundingResult,
        cardinality: str | None = None,
        tool_call: ToolCallContext | None = None,
        on_policy: Callable[[list[PolicyViolation]], bool] | None = None,
    ) -> PolicyResult:
        """
        Evaluate policies for the given concepts.

        `concepts` may be a list of concept names (grounding is run, via `check`, so
        metrics and traces are recorded) or a pre-computed `GroundingResult` (grounding
        is skipped). Either way, if the result is **not grounded**, the call fails
        closed with BLOCK — policy enforcement is only meaningful over a grounded
        closure, and confused epistemic state must never yield a green PASS.

        `tool_call` carries the tool name and the args/kwargs of the decorated
        function so that policy callbacks can make domain-specific decisions.
        Omit when calling outside a guarded context.

        `on_policy` is an optional handler consulted when any violation has
        `requires_confirmation=True`. It receives all violations and returns
        a single bool — True to proceed, False to block. When absent, the
        fail-safe is BLOCK. An exception raised by `on_policy` is captured and
        fails closed (BLOCK) rather than propagating to the caller.

        Returns PolicyResult with action PASS or BLOCK (never PENDING).
        """
        if isinstance(concepts, GroundingResult):
            grounding = concepts
        else:
            grounding = self.check(concepts)  # routes through metrics + tracing, like check()

        if not grounding.grounded or grounding.trace is None:
            # Fail closed: enforcement is only meaningful over a grounded closure. An
            # ungrounded result (empty or partial closure) is confused epistemic state.
            # The engine always pairs grounded=True with a trace, so a "grounded" result
            # lacking one is malformed — a hand-built attempt to bypass grounding — and is
            # rejected for the same reason: never a green PASS on input the engine could
            # not have produced.
            policy_result = PolicyResult(
                action=PolicyAction.BLOCK,
                reason="policy evaluation requires a grounded result",
            )
        else:
            card_enum: Cardinality | None = None
            if cardinality is not None:
                try:
                    card_enum = Cardinality(cardinality)
                except ValueError:
                    card_enum = None  # unknown -> fire all policies

            violations = self._policy_gate.evaluate(
                grounding.trace, card_enum, tool_call=tool_call,
                grounding=GroundingContext.from_grounding(grounding),
            )

            if not violations:
                policy_result = PolicyResult(action=PolicyAction.PASS)
            else:
                hard_blocks = [v for v in violations if not v.requires_confirmation]
                pending = [v for v in violations if v.requires_confirmation]

                # Hard blocks do not consult on_policy — they are immediate BLOCKs with their own messages
                if hard_blocks:
                    # Violation messages are multi-line (callback reason + confirmation);
                    # separate distinct hard blocks with a blank line to keep them readable.
                    messages = "\n\n".join(v.message for v in hard_blocks)
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
                else:
                    try:
                        proceed = on_policy(pending)
                    except Exception as exc:  # noqa: BLE001 — a raising handler fails closed
                        logger.exception("Policy confirmation handler raised")
                        policy_result = PolicyResult(
                            action=PolicyAction.BLOCK,
                            reason=f"confirmation handler raised {exc!r}",
                            violations=pending,
                        )
                    else:
                        policy_result = PolicyResult(
                            action=PolicyAction.PASS if proceed else PolicyAction.BLOCK,
                            reason=None if proceed else "User declined",
                            violations=pending,
                        )

        return policy_result
