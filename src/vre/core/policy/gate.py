# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyGate — evaluates policy violations against an epistemic trace.
"""

import logging

from vre.core.models import EpistemicResponse, RelationType
from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.models import Cardinality, PolicyCallbackResult, PolicyViolation

logger = logging.getLogger(__name__)


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
                            logger.debug(
                                "Policy %r skipped: trigger_cardinality=%s does not match %s",
                                policy.name, policy.trigger_cardinality, cardinality,
                            )
                            continue
                        cb = policy.resolve_callback()
                        cb_result: PolicyCallbackResult | None = None
                        if cb is not None and call_context is not None:
                            cb_result = cb(call_context)
                            if cb_result.passed:
                                logger.debug("Policy %r passed by callback", policy.name)
                                continue  # action passed the policy — no violation

                        message = policy.confirmation_message

                        logger.info(
                            "Policy violation: %r — %s (requires_confirmation=%s)",
                            policy.name, message, policy.requires_confirmation,
                        )
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
        logger.debug("Evaluating policies (cardinality=%s, has_context=%s)", cardinality, call_context is not None)
        violations = self._collect_violations(response, cardinality, call_context)
        if violations:
            logger.info("Policy evaluation: %d violation(s)", len(violations))
        else:
            logger.debug("Policy evaluation: no violations")
        return violations