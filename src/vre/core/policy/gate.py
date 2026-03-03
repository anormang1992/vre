# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyGate — evaluates policy violations against an epistemic trace.
"""

from vre.core.models import EpistemicResponse, RelationType
from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.models import Cardinality, Policy, PolicyResult, PolicyViolation


class PolicyGate:
    """
    Evaluates policies attached to an epistemic trace and returns a PolicyResult.
    """

    def evaluate(
        self,
        response: EpistemicResponse,
        cardinality: Cardinality = Cardinality.SINGLE,
        call_context: PolicyCallContext | None = None,
    ) -> PolicyResult:
        """
        Evaluate all policies in the trace and return PASS, PENDING, or BLOCK.
        """
        violations = self._collect_violations(response, cardinality, call_context)
        if not violations:
            return PolicyResult(action="PASS")
        pending = [v for v in violations if v.requires_confirmation]
        if pending:
            return PolicyResult(
                action="PENDING",
                confirmation_message=pending[0].message,
            )
        return PolicyResult(action="PASS")  # informational-only violations don't block

    def _collect_violations(
        self,
        response: EpistemicResponse,
        cardinality: Cardinality,
        call_context: PolicyCallContext | None = None,
    ) -> list[PolicyViolation]:
        """
        Walk all APPLIES_TO relata in the trace and collect triggered policy violations.
        """
        violations: list[PolicyViolation] = []
        for primitive in response.result.primitives:
            for depth in primitive.depths:
                for relatum in depth.relata:
                    if relatum.relation_type != RelationType.APPLIES_TO:
                        continue
                    for policy in relatum.policies:
                        if not self._triggers(policy, cardinality):
                            continue
                        cb = policy.resolve_callback()
                        if cb is not None and call_context is not None and not cb(call_context):
                            continue
                        try:
                            message = policy.confirmation_message.format(
                                action=primitive.name
                            )
                        except (KeyError, ValueError):
                            message = policy.confirmation_message
                        violations.append(PolicyViolation(
                            policy=policy,
                            requires_confirmation=policy.requires_confirmation,
                            message=message,
                        ))
        return violations

    @staticmethod
    def _triggers(policy: Policy, cardinality: Cardinality) -> bool:
        """
        Return True if the policy should fire for the given cardinality.
        """
        if not policy.requires_confirmation:
            return False
        if policy.trigger_cardinality is None:
            return True
        return policy.trigger_cardinality == cardinality
