"""
Unit tests for Policy models and PolicyGate.
"""

import json
from uuid import uuid4

from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicQuery,
    EpistemicResponse,
    EpistemicResult,
    Primitive,
    Relatum,
    RelationType,
)
from vre.core.policy import Cardinality, Policy, PolicyCallbackResult, parse_policy
from vre.core.policy.gate import PolicyGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_primitive_with_applies_to(name: str, policies: list[Policy]) -> Primitive:
    target_id = uuid4()
    relatum = Relatum(
        relation_type=RelationType.APPLIES_TO,
        target_id=target_id,
        target_depth=DepthLevel.CONSTRAINTS,
        policies=policies,
    )
    depth = Depth(level=DepthLevel.CAPABILITIES, relata=[relatum])
    return Primitive(name=name, depths=[depth])


def _make_step_result(primitive: Primitive) -> EpistemicResponse:
    query = EpistemicQuery(concept_ids=[primitive.id])
    result = EpistemicResult(primitives=[primitive])
    return EpistemicResponse(query=query, result=result)


def test_policy_metadata_preserved():
    """Metadata dict round-trips through JSON serialization unchanged."""
    original_meta = {"owner": "ops-team", "level": "critical", "ticket": "VLI-99"}
    policy = Policy(
        name="SafeWrite",
        metadata=original_meta,
    )

    serialized = json.dumps(policy.model_dump(), default=str)
    restored = parse_policy(json.loads(serialized))
    assert restored.metadata == original_meta


def test_policy_cardinality_field():
    """trigger_cardinality accepts None and both Cardinality variants."""
    p_none = Policy(name="AlwaysFires", trigger_cardinality=None)
    p_single = Policy(name="SingleOnly", trigger_cardinality=Cardinality.SINGLE)
    p_multi = Policy(name="MultiOnly", trigger_cardinality=Cardinality.MULTIPLE)

    assert p_none.trigger_cardinality is None
    assert p_single.trigger_cardinality == Cardinality.SINGLE
    assert p_multi.trigger_cardinality == Cardinality.MULTIPLE


def test_policy_defaults():
    """Default values are sensible."""
    policy = Policy(name="Minimal")
    assert policy.requires_confirmation is True
    assert policy.trigger_cardinality is None
    assert policy.callback is None
    assert policy.metadata == {}


# ---------------------------------------------------------------------------
# PolicyGate tests
# ---------------------------------------------------------------------------


def test_no_policies_proceed():
    """Relatum with no policies → no violations."""
    primitive = _make_primitive_with_applies_to("create", [])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 0


def test_step_cardinality_single_no_trigger():
    """SINGLE cardinality does not trigger a policy that requires MULTIPLE → no violations."""
    policy = Policy(
        name="BulkDelete",
        trigger_cardinality=Cardinality.MULTIPLE,
        confirmation_message="Bulk {action} requires confirmation.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 0


def test_step_cardinality_multiple_triggers():
    """MULTIPLE cardinality triggers a policy scoped to MULTIPLE → violation produced."""
    policy = Policy(
        name="BulkDelete",
        trigger_cardinality=Cardinality.MULTIPLE,
        confirmation_message="Bulk {action} requires confirmation.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.MULTIPLE)
    assert len(violations) == 1
    assert "delete" in violations[0].message


def test_policy_trigger_cardinality_none_always_fires():
    """trigger_cardinality=None fires for both SINGLE and MULTIPLE → violations."""
    policy = Policy(
        name="AlwaysConfirm",
        trigger_cardinality=None,
        confirmation_message="Always confirm {action}.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)

    assert len(PolicyGate().evaluate(response, Cardinality.SINGLE)) == 1
    assert len(PolicyGate().evaluate(response, Cardinality.MULTIPLE)) == 1


def test_no_callback_fires_violation():
    """A policy with no callback fires the violation."""
    policy = Policy(
        name="NoCallback",
        callback=None,
        confirmation_message="Confirm {action}.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 1
    assert violations[0].message is not None


def test_confirmation_message_formatted():
    """{action} in confirmation_message is interpolated from primitive.name."""
    policy = Policy(
        name="Confirm",
        confirmation_message="About to {action} — proceed?",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert violations[0].message == "About to delete — proceed?"


def test_non_applies_to_relata_ignored():
    """CONSTRAINED_BY relata with policies are not evaluated by the gate → no violations."""
    policy = Policy(name="ShouldBeIgnored")
    target_id = uuid4()
    relatum = Relatum(
        relation_type=RelationType.CONSTRAINED_BY,
        target_id=target_id,
        target_depth=DepthLevel.CONSTRAINTS,
        policies=[policy],
    )
    depth = Depth(level=DepthLevel.CAPABILITIES, relata=[relatum])
    primitive = Primitive(name="create", depths=[depth])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 0


def test_requires_confirmation_false_produces_violations():
    """A policy with requires_confirmation=False still produces violations (no longer filtered)."""
    policy = Policy(
        name="Informational",
        requires_confirmation=False,
        trigger_cardinality=None,
        confirmation_message="This would normally show.",
    )
    primitive = _make_primitive_with_applies_to("list", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 1
    assert violations[0].requires_confirmation is False


def test_callback_passed_true_suppresses_violation():
    """Callback returning passed=True — action passes the policy, no violation."""
    policy = Policy(
        name="WithCallback",
        callback="tests.vre.test_policies._cb_pass",
        confirmation_message="Confirm {action}.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)

    from vre.core.policy.callback import PolicyCallContext
    from vre.core.grounding import GroundingResult

    ctx = PolicyCallContext(
        tool_name="test_fn",
        grounding=GroundingResult(grounded=True, resolved=["write"], gaps=[]),
        call_args=(),
        call_kwargs={},
    )
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE, ctx)
    assert len(violations) == 0


def test_callback_passed_false_fires_violation():
    """Callback returning passed=False — action fails the policy, violation carries result."""
    policy = Policy(
        name="WithCallback",
        callback="tests.vre.test_policies._cb_fail",
        confirmation_message="Confirm {action}.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)

    from vre.core.policy.callback import PolicyCallContext
    from vre.core.grounding import GroundingResult

    ctx = PolicyCallContext(
        tool_name="test_fn",
        grounding=GroundingResult(grounded=True, resolved=["write"], gaps=[]),
        call_args=(),
        call_kwargs={},
    )
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE, ctx)
    assert len(violations) == 1
    assert violations[0].callback_result is not None
    assert violations[0].callback_result.passed is False


def test_multiple_policies_on_same_relatum_all_collected():
    """Multiple policies on the same relatum are all collected."""
    p1 = Policy(name="Policy1", confirmation_message="First.")
    p2 = Policy(name="Policy2", confirmation_message="Second.")
    primitive = _make_primitive_with_applies_to("delete", [p1, p2])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 2
    names = {v.policy.name for v in violations}
    assert names == {"Policy1", "Policy2"}


# ---------------------------------------------------------------------------
# Callback fixtures (importable by dotted path)
# ---------------------------------------------------------------------------


def _cb_pass(context) -> PolicyCallbackResult:
    return PolicyCallbackResult(passed=True, message="Allowed by callback")


def _cb_fail(context) -> PolicyCallbackResult:
    return PolicyCallbackResult(passed=False, message="Failed by callback")
