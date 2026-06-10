# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyCallback Protocol and PolicyCallContext — the user-facing callback contract.

Users implementing custom policy logic should type-annotate against
PolicyCallback and accept a PolicyCallContext argument. PolicyCallContext is a
composition of four orthogonal pieces — the tool call (`tool_call`), a bounded
grounding facade (`grounding`), the edge that fired (`triggering_edge`), and the
policy itself (`policy`).

Example::

    from vre.core.policy.callback import PolicyCallback, PolicyCallContext
    from vre.core.policy.models import PolicyCallbackResult

    class AllowTempWrites:
        def __call__(self, context: PolicyCallContext) -> PolicyCallbackResult:
            path = context.tool_call.call_kwargs.get("path", "")
            if path.startswith("/tmp"):
                return PolicyCallbackResult(passed=True, message="Temp writes allowed")
            return PolicyCallbackResult(passed=False)
"""

from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from vre.core.models import DepthLevel
from vre.core.policy.models import Policy, PolicyCallbackResult

if TYPE_CHECKING:
    from vre.core.grounding import GroundingResult


class ToolCallContext(BaseModel):
    """
    The tool invocation being gated — only what the caller knows.

    Built by `vre_guard` (or an integrator calling `check_policy` directly).
    The grounding facade and the per-edge identity are not here: VRE derives
    the former and the gate builds the latter.
    """

    model_config = {"arbitrary_types_allowed": True}  # call args may be arbitrary objects

    tool_name: str
    call_args: tuple[Any, ...] = ()
    call_kwargs: dict[str, Any] = Field(default_factory=dict)


class GroundingContext(BaseModel):
    """
    Minimal, VRE-authored view of the grounding that occurred in this call.

    Replaces the leaky full `GroundingResult`. Callbacks may branch on which
    other root concepts were grounded alongside the triggering edge. `agent_id`
    is callback pass-through — policy firing is agent-agnostic.
    """

    agent_id: UUID | None = None
    resolved_concepts: list[str] = Field(default_factory=list)

    @classmethod
    def from_grounding(cls, grounding: "GroundingResult") -> "GroundingContext":
        """
        Project the bounded, callback-facing facade from a full GroundingResult —
        only `agent_id` and the resolved concept names cross this boundary.
        """
        return cls(agent_id=grounding.agent_id, resolved_concepts=grounding.resolved)


class TriggeringEdge(BaseModel):
    """
    The specific APPLIES_TO edge whose policy invoked this callback.

    Lets one callback registered on several edges tell them apart — by
    source/target concept and by the depth the edge lives at (D2 vs D3).
    """

    source_name: str
    target_name: str
    source_depth: DepthLevel
    target_depth: DepthLevel


class PolicyCallContext(BaseModel):
    """
    Complete, per-edge context passed to a policy callback.

    A composition of four orthogonal pieces: the tool call, the grounding
    facade, the edge that fired, and the policy that fired. All are populated
    by the gate before the callback runs.
    """

    tool_call: ToolCallContext
    grounding: GroundingContext
    triggering_edge: TriggeringEdge
    policy: Policy


class PolicyCallback(Protocol):
    """
    Protocol for policy callback callables.

    A callback receives the full call context and returns a PolicyCallbackResult:

    - `passed=True`  — the action passes the policy (no violation)
    - `passed=False` — the action fails the policy (violation fires)

    Implement this Protocol to write custom, domain-specific policy logic.
    """

    def __call__(self, context: PolicyCallContext) -> PolicyCallbackResult:
        """
        Evaluate the policy against the given call context.

        Return PolicyCallbackResult(passed=True) if the action passes,
        PolicyCallbackResult(passed=False) if the action fails the policy.
        """
        ...
