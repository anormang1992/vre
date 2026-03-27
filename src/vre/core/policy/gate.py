# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyGate — evaluates policy violations against an epistemic trace.
"""

from vre.core.models import EpistemicResponse, RelationType
from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.models import Cardinality, PolicyCallbackResult, PolicyViolation


class PolicyGate:
    """
    Evaluates policies attached to an epistemic trace and returns violations.
    """
    @staticmethod
    def _collect_violations(
        response: EpistemicResponse,
        cardinality: Cardinality | None = None,
        call_context: PolicyCallContext | None = None,
    ) -> list[PolicyViolation]:
        """
        Walk all APPLIES_TO relata in the trace and collect triggered policy violations.

        When cardinality is None (unknown), all policies fire — we cannot justify
        skipping any policy without knowing the cardinality.
        """
        violations: list[PolicyViolation] = []
        for primitive in response.result.primitives:
            for depth in primitive.depths:
                for relatum in depth.relata:
                    if relatum.relation_type != RelationType.APPLIES_TO:
                        continue
                    for policy in relatum.policies:
                        if (cardinality is not None
                                and policy.trigger_cardinality is not None
                                and policy.trigger_cardinality != cardinality):
                            continue
                        cb = policy.resolve_callback()
                        cb_result: PolicyCallbackResult | None = None
                        if cb is not None and call_context is not None:
                            cb_result = cb(call_context)
                            if cb_result.passed:
                                continue  # action passed the policy — no violation

                        try:
                            message = policy.confirmation_message.format(
                                action=primitive.name
                            )
                        except (KeyError, ValueError):
                            message = policy.confirmation_message

                        violations.append(PolicyViolation(
                            policy=policy,
                            message=message,
                            callback_result=cb_result,
                        ))

        return violations

    def evaluate(
        self,
        response: EpistemicResponse,
        cardinality: Cardinality | None = None,
        call_context: PolicyCallContext | None = None,
    ) -> list[PolicyViolation]:
        """
        Evaluate all policies in the trace and return triggered violations.
        """
        return self._collect_violations(response, cardinality, call_context)