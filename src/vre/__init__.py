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

from vre.core.errors import (
    CandidateValidationError,
    CyclicRelationshipError,
    GraphError,
    GraphIntegrityError,
    HydrationError,
    PersistenceError,
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
)
from vre.core.policy import Cardinality, PolicyAction, PolicyCallbackResult, PolicyResult, PolicyViolation
from vre.core.policy.callback import (
    GroundingContext,
    PolicyCallContext,
    ToolCallContext,
    TriggeringEdge,
)
from vre.core.policy.gate import PolicyGate
from vre.identity import AgentIdentity, AgentRegistry
from vre.learning import LearningEngine, template_for_gap
from vre.metrics import MetricsManager
from vre.tracing import TraceManager, TraceWriter

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
    ) -> None:
        """
        Initialize VRE with the given primitive repository.

        When `agent_key` is provided, it resolves via the persisted registry
        to a stable AgentIdentity stamped on every GroundingResult; `agent_name`
        is used only on first registration. Traces are persisted to daily JSONL
        files under `~/.vre/traces/` when `persist_traces` is True.
        """
        self._repo = repository
        self._engine = GroundingEngine(repository)
        self._learning_engine = LearningEngine(repository)
        self._metrics = MetricsManager(repository)
        self._traces = TraceManager(TraceWriter() if persist_traces else None)

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
        self._traces.write_check(concepts, result)
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

        `concepts` may be a list of concept names (grounding is run) or a
        pre-computed `GroundingResult` (grounding is skipped).

        `tool_call` carries the tool name and the args/kwargs of the decorated
        function so that policy callbacks can make domain-specific decisions.
        Omit when calling outside a guarded context.

        `on_policy` is an optional handler consulted when any violation has
        `requires_confirmation=True`. It receives all violations and returns
        a single bool — True to proceed, False to block. When absent, the
        fail-safe is BLOCK.

        Returns PolicyResult with action PASS or BLOCK (never PENDING).
        """
        if isinstance(concepts, GroundingResult):
            grounding = concepts
        else:
            grounding = self._stamp_identity(self._engine.ground(concepts))

        if grounding.trace is None:
            policy_result = PolicyResult(action=PolicyAction.PASS)
        else:
            card_enum: Cardinality | None = None
            if cardinality is not None:
                try:
                    card_enum = Cardinality(cardinality)
                except ValueError:
                    card_enum = None  # unknown -> fire all policies

            gate = PolicyGate()
            grounding_ctx = GroundingContext(
                agent_id=grounding.agent_id,
                resolved_concepts=grounding.resolved,
            )
            violations = gate.evaluate(
                grounding.trace, card_enum, tool_call=tool_call, grounding=grounding_ctx,
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
