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
from vre.core.policy import Cardinality, Policy, parse_policy
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
    """Relatum with no policies → PASS."""
    primitive = _make_primitive_with_applies_to("create", [])
    response = _make_step_result(primitive)
    result = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert result.action == "PASS"


def test_step_cardinality_single_no_trigger():
    """SINGLE cardinality does not trigger a policy that requires MULTIPLE → PASS."""
    policy = Policy(
        name="BulkDelete",
        trigger_cardinality=Cardinality.MULTIPLE,
        confirmation_message="Bulk {action} requires confirmation.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    result = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert result.action == "PASS"


def test_step_cardinality_multiple_triggers():
    """MULTIPLE cardinality triggers a policy scoped to MULTIPLE → PENDING."""
    policy = Policy(
        name="BulkDelete",
        trigger_cardinality=Cardinality.MULTIPLE,
        confirmation_message="Bulk {action} requires confirmation.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    result = PolicyGate().evaluate(response, Cardinality.MULTIPLE)
    assert result.action == "PENDING"
    assert result.confirmation_message is not None
    assert "BulkDelete" in result.confirmation_message or "delete" in result.confirmation_message


def test_policy_trigger_cardinality_none_always_fires():
    """trigger_cardinality=None fires for both SINGLE and MULTIPLE → PENDING."""
    policy = Policy(
        name="AlwaysConfirm",
        trigger_cardinality=None,
        confirmation_message="Always confirm {action}.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)

    assert PolicyGate().evaluate(response, Cardinality.SINGLE).action == "PENDING"
    assert PolicyGate().evaluate(response, Cardinality.MULTIPLE).action == "PENDING"


def test_no_callback_fires_violation():
    """A policy with no callback fires the violation → PENDING."""
    policy = Policy(
        name="NoCallback",
        callback=None,
        confirmation_message="Confirm {action}.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    result = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert result.action == "PENDING"
    assert result.confirmation_message is not None


def test_confirmation_message_formatted():
    """{action} in confirmation_message is interpolated from primitive.name."""
    policy = Policy(
        name="Confirm",
        confirmation_message="About to {action} — proceed?",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    result = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert result.confirmation_message == "About to delete — proceed?"


def test_non_applies_to_relata_ignored():
    """CONSTRAINED_BY relata with policies are not evaluated by the gate → PASS."""
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
    result = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert result.action == "PASS"


def test_requires_confirmation_false_never_triggers():
    """A policy with requires_confirmation=False never produces a violation → PASS."""
    policy = Policy(
        name="Informational",
        requires_confirmation=False,
        trigger_cardinality=None,
        confirmation_message="This would normally show.",
    )
    primitive = _make_primitive_with_applies_to("list", [policy])
    response = _make_step_result(primitive)
    result = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert result.action == "PASS"
