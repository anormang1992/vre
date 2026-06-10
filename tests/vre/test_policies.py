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
        confirmation_message="Bulk delete requires confirmation.",
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
        confirmation_message="Bulk delete requires confirmation.",
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
        confirmation_message="Always confirm write.",
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
        confirmation_message="Confirm delete.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert len(violations) == 1
    assert violations[0].message is not None


def test_confirmation_message_verbatim():
    """confirmation_message is used verbatim — {action} is NOT interpolated."""
    policy = Policy(
        name="Confirm",
        confirmation_message="About to {action} — proceed?",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)
    assert violations[0].message == "About to {action} — proceed?"


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
    from vre.core.policy.callback import ToolCallContext
    policy = Policy(
        name="WithCallback",
        callback="tests.vre.test_policies._cb_pass",
        confirmation_message="Confirm write.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)
    tool_call = ToolCallContext(tool_name="test_fn")
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE, tool_call=tool_call)
    assert len(violations) == 0


def test_callback_passed_false_fires_violation():
    """Callback returning passed=False — action fails the policy, violation carries result."""
    from vre.core.policy.callback import ToolCallContext
    policy = Policy(
        name="WithCallback",
        callback="tests.vre.test_policies._cb_fail",
        confirmation_message="Confirm write.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)
    tool_call = ToolCallContext(tool_name="test_fn")
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE, tool_call=tool_call)
    assert len(violations) == 1
    assert violations[0].callback_result is not None
    assert violations[0].callback_result.passed is False


def test_callback_receives_triggering_edge():
    """The callback sees the source/target names and depths of the edge that fired it."""
    from vre.core.policy.callback import ToolCallContext
    policy = Policy(
        name="EdgeAware",
        callback="tests.vre.test_policies._cb_record_edge",
        confirmation_message="Confirm.",
    )
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)
    _RECORDED_EDGES.clear()
    PolicyGate().evaluate(response, Cardinality.SINGLE, tool_call=ToolCallContext(tool_name="rm"))
    assert len(_RECORDED_EDGES) == 1
    edge = _RECORDED_EDGES[0]
    assert edge.source_name == "delete"
    assert edge.source_depth == DepthLevel.CAPABILITIES   # _make_primitive_with_applies_to: D2 source
    assert edge.target_depth == DepthLevel.CONSTRAINTS    # ...and D3 target


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


_RECORDED_EDGES = []


def _cb_record_edge(context) -> PolicyCallbackResult:
    _RECORDED_EDGES.append(context.triggering_edge)
    return PolicyCallbackResult(passed=True)


_RECORDED_META = []


def _cb_record_metadata(context) -> PolicyCallbackResult:
    _RECORDED_META.append(dict(context.policy.metadata))
    return PolicyCallbackResult(passed=True)


def _cb_block_if_protected(context) -> PolicyCallbackResult:
    if "protected" in context.grounding.resolved_concepts:
        return PolicyCallbackResult(passed=False, message="protected concept co-grounded")
    return PolicyCallbackResult(passed=True)


# ---------------------------------------------------------------------------
# New capability-coverage tests (Task 4)
# ---------------------------------------------------------------------------


def _make_two_edge_response(source_name, policies, target_a="file", target_b="dir"):
    """A source primitive with two APPLIES_TO edges (to target_a, target_b), all targets in the trace."""
    a = Primitive(name=target_a)
    b = Primitive(name=target_b)
    depth = Depth(
        level=DepthLevel.CAPABILITIES,
        relata=[
            Relatum(relation_type=RelationType.APPLIES_TO, target_id=a.id,
                    target_depth=DepthLevel.CONSTRAINTS, policies=policies),
            Relatum(relation_type=RelationType.APPLIES_TO, target_id=b.id,
                    target_depth=DepthLevel.CONSTRAINTS, policies=policies),
        ],
    )
    source = Primitive(name=source_name, depths=[depth])
    query = EpistemicQuery(concept_ids=[source.id])
    result = EpistemicResult(primitives=[source, a, b])
    return EpistemicResponse(query=query, result=result)


def test_callback_on_two_edges_receives_distinct_targets():
    """One callback on two edges is told which target each invocation is for."""
    from vre.core.policy.callback import ToolCallContext
    policy = Policy(name="EdgeAware", callback="tests.vre.test_policies._cb_record_edge")
    response = _make_two_edge_response("delete", [policy])
    _RECORDED_EDGES.clear()
    PolicyGate().evaluate(response, Cardinality.SINGLE, tool_call=ToolCallContext(tool_name="rm"))
    targets = sorted(e.target_name for e in _RECORDED_EDGES)
    assert targets == ["dir", "file"]


def test_callback_reads_policy_metadata():
    """The callback can read its own parameters from the triggering policy's metadata."""
    from vre.core.policy.callback import ToolCallContext
    policy = Policy(
        name="RateLimited",
        callback="tests.vre.test_policies._cb_record_metadata",
        metadata={"limit": 5},
    )
    primitive = _make_primitive_with_applies_to("send", [policy])
    response = _make_step_result(primitive)
    _RECORDED_META.clear()
    PolicyGate().evaluate(response, Cardinality.SINGLE, tool_call=ToolCallContext(tool_name="send"))
    assert _RECORDED_META == [{"limit": 5}]


def test_callback_branches_on_resolved_concepts():
    """A callback can fail based on a co-occurring concept in the grounding facade."""
    from vre.core.policy.callback import GroundingContext, ToolCallContext
    policy = Policy(name="ProtectedAware", callback="tests.vre.test_policies._cb_block_if_protected")
    primitive = _make_primitive_with_applies_to("delete", [policy])
    response = _make_step_result(primitive)

    safe = GroundingContext(resolved_concepts=["delete", "file"])
    violations = PolicyGate().evaluate(
        response, Cardinality.SINGLE,
        tool_call=ToolCallContext(tool_name="rm"), grounding=safe,
    )
    assert len(violations) == 0  # callback passed

    risky = GroundingContext(resolved_concepts=["delete", "file", "protected"])
    violations = PolicyGate().evaluate(
        response, Cardinality.SINGLE,
        tool_call=ToolCallContext(tool_name="rm"), grounding=risky,
    )
    assert len(violations) == 1  # callback fired


def test_no_tool_call_skips_callback_and_fires():
    """Without a tool_call, the callback is not consulted and the policy fires (behavior preserved)."""
    policy = Policy(
        name="WouldPass",
        callback="tests.vre.test_policies._cb_pass",  # would suppress IF consulted
        confirmation_message="Confirm.",
    )
    primitive = _make_primitive_with_applies_to("write", [policy])
    response = _make_step_result(primitive)
    violations = PolicyGate().evaluate(response, Cardinality.SINGLE)  # no tool_call
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# New context type tests (Task 2)
# ---------------------------------------------------------------------------


def test_tool_call_context_defaults():
    """ToolCallContext carries the invocation; args/kwargs default to empty."""
    from vre.core.policy.callback import ToolCallContext
    tc = ToolCallContext(tool_name="write_file")
    assert tc.tool_name == "write_file"
    assert tc.call_args == ()
    assert tc.call_kwargs == {}


def test_grounding_context_defaults():
    """GroundingContext defaults to no agent and no resolved concepts."""
    from vre.core.policy.callback import GroundingContext
    gc = GroundingContext()
    assert gc.agent_id is None
    assert gc.resolved_concepts == []


def test_triggering_edge_fields():
    """TriggeringEdge captures source/target name and the two depths."""
    from vre.core.policy.callback import TriggeringEdge
    edge = TriggeringEdge(
        source_name="delete",
        target_name="file",
        source_depth=DepthLevel.CONSTRAINTS,
        target_depth=DepthLevel.CAPABILITIES,
    )
    assert edge.source_name == "delete"
    assert edge.target_name == "file"
    assert edge.source_depth == DepthLevel.CONSTRAINTS
    assert edge.target_depth == DepthLevel.CAPABILITIES
