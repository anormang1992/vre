# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyGate — overlays code-resident policy placements onto an epistemic trace.
"""

import logging

from vre.core.models import EpistemicResponse, RelationType, format_depth_label
from vre.core.policy.callback import (
    GroundingContext,
    PolicyCallback,
    PolicyCallContext,
    ToolCallContext,
    TriggeringEdge,
)
from vre.core.policy.models import Cardinality, Policy, PolicyCallbackResult, PolicyViolation
from vre.core.policy.registry import PolicyRegistry

logger = logging.getLogger(__name__)


class PolicyGate:
    """
    Overlays the registry's policy placements onto a trace's APPLIES_TO edges and
    returns the triggered violations.
    """

    def __init__(self, registry: PolicyRegistry) -> None:
        """
        Bind the gate to the registry whose placements it overlays.
        """
        self._registry = registry

    @staticmethod
    def _evaluate_callback(
        callback: PolicyCallback,
        policy: Policy,
        edge: TriggeringEdge,
        tool_call: ToolCallContext | None,
        grounding: GroundingContext,
    ) -> PolicyCallbackResult:
        """
        Run a placement's callback and return its result, failing closed with a
        detailed reason rather than ever propagating a raw exception.

        Two fail-closed cases (both fire the policy): no tool_call in this context, and
        a callback that raises. Otherwise the callback's own result decides.
        """
        detail = (
            f"Policy {policy.name!r} on {edge.source_name}->{edge.target_name} "
            f"@ {format_depth_label(edge.source_depth)}"
        )
        if tool_call is None:
            result = PolicyCallbackResult.unevaluable(
                f"{detail}: callback could not be evaluated (no tool call in this "
                f"context); firing conservatively."
            )
        else:
            context = PolicyCallContext(
                tool_call=tool_call, grounding=grounding, triggering_edge=edge, policy=policy
            )
            try:
                result = callback(context)
                if not isinstance(result, PolicyCallbackResult):
                    raise TypeError(
                        f"callback returned {type(result).__name__}, expected PolicyCallbackResult"
                    )
            except Exception as exc:  # noqa: BLE001 — fail closed on any callback failure
                logger.exception("Policy %r callback failed", policy.name)
                result = PolicyCallbackResult.unevaluable(
                    f"{detail}: callback raised {exc!r}; firing conservatively."
                )
        return result

    def _collect_violations(
        self,
        response: EpistemicResponse,
        cardinality: Cardinality | None = None,
        tool_call: ToolCallContext | None = None,
        grounding: GroundingContext | None = None,
    ) -> list[PolicyViolation]:
        """
        Walk every APPLIES_TO relatum in the trace and collect the violations from the
        placements registered on that edge.

        When cardinality is None (unknown), all policies fire — we cannot justify
        skipping any policy without knowing the cardinality.
        """
        violations: list[PolicyViolation] = []
        id_to_name = {p.id: p.name for p in response.result.primitives}
        grounding = grounding or GroundingContext()
        for primitive in response.result.primitives:
            for depth in primitive.depths:
                for relatum in depth.relata:
                    if relatum.relation_type != RelationType.APPLIES_TO:
                        continue
                    target_name = id_to_name.get(relatum.target_id, str(relatum.target_id))
                    for placement in self._registry.placements_for(
                        primitive.name, target_name, depth.level
                    ):
                        policy = placement.policy
                        if (cardinality is not None
                                and policy.trigger_cardinality is not None
                                and policy.trigger_cardinality != cardinality):
                            logger.debug(
                                "Policy %r skipped: trigger_cardinality=%s does not match %s",
                                policy.name, policy.trigger_cardinality, cardinality,
                            )
                            continue

                        edge = TriggeringEdge(
                            source_name=primitive.name,
                            target_name=target_name,
                            source_depth=depth.level,
                            target_depth=relatum.target_depth,
                        )
                        cb_result = self._evaluate_callback(
                            placement.callback, policy, edge, tool_call, grounding
                        )
                        if cb_result.passed:
                            logger.debug("Policy %r passed by callback", policy.name)
                            continue  # action passed the policy — no violation

                        violation = PolicyViolation(policy=policy, callback_result=cb_result)
                        logger.info(
                            "Policy violation: %r — %s (requires_confirmation=%s)",
                            policy.name, violation.message, policy.requires_confirmation,
                        )
                        violations.append(violation)

        return violations

    def evaluate(
        self,
        response: EpistemicResponse,
        cardinality: Cardinality | None = None,
        tool_call: ToolCallContext | None = None,
        grounding: GroundingContext | None = None,
    ) -> list[PolicyViolation]:
        """
        Evaluate all placements against the trace and return triggered violations.
        """
        logger.debug(
            "Evaluating policies (cardinality=%s, has_tool_call=%s)",
            cardinality, tool_call is not None,
        )
        if len(self._registry) == 0:
            violations: list[PolicyViolation] = []  # nothing registered — skip the trace walk
        else:
            violations = self._collect_violations(response, cardinality, tool_call, grounding)
        if violations:
            logger.info("Policy evaluation: %d violation(s)", len(violations))
        else:
            logger.debug("Policy evaluation: no violations")
        return violations
