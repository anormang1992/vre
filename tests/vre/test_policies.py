"""
Unit tests for Policy models and PolicyGate (registry-based, code-resident policies).
"""

from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicQuery,
    EpistemicResponse,
    EpistemicResult,
    Primitive,
    Provenance,
    ProvenanceSource,
    Relatum,
    RelationType,
)
from vre.core.policy import Cardinality, Policy, PolicyCallbackResult, PolicyViolation
from vre.core.policy.callback import GroundingContext, ToolCallContext, TriggeringEdge
from vre.core.policy.gate import PolicyGate
from vre.core.policy.registry import PolicyRegistry

_PROV = Provenance(source=ProvenanceSource.AUTHORED)

# Edge geometry shared by the helpers: the APPLIES_TO edge lives on the source at D2,
# and requires the target grounded to D3.
SRC_DEPTH = DepthLevel.CAPABILITIES
TGT_DEPTH = DepthLevel.CONSTRAINTS


# ---------------------------------------------------------------------------
# Callback fixtures (registered into a per-test registry, never imported by path)
# ---------------------------------------------------------------------------


def _cb_pass(context) -> PolicyCallbackResult:
    return PolicyCallbackResult(passed=True, message="Allowed by callback")


def _cb_fail(context) -> PolicyCallbackResult:
    return PolicyCallbackResult(passed=False, message="Failed by callback")


def _cb_raise(context) -> PolicyCallbackResult:
    raise RuntimeError("boom")


def _cb_wrong_type(context):
    return None  # not a PolicyCallbackResult — a buggy callback


_RECORDED_EDGES: list[TriggeringEdge] = []


def _cb_record_edge(context) -> PolicyCallbackResult:
    _RECORDED_EDGES.append(context.triggering_edge)
    return PolicyCallbackResult(passed=True)


_RECORDED_META: list[dict] = []


def _cb_record_metadata(context) -> PolicyCallbackResult:
    _RECORDED_META.append(dict(context.policy.metadata))
    return PolicyCallbackResult(passed=True)


def _cb_block_if_protected(context) -> PolicyCallbackResult:
    if "protected" in context.grounding.resolved_concepts:
        return PolicyCallbackResult(passed=False, message="protected concept co-grounded")
    return PolicyCallbackResult(passed=True)


_RECORDED_TOOL_CALLS: list[ToolCallContext] = []


def _cb_record_tool_call(context) -> PolicyCallbackResult:
    _RECORDED_TOOL_CALLS.append(context.tool_call)
    return PolicyCallbackResult(passed=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace(source_name, target_name="file", relation=RelationType.APPLIES_TO):
    """A source primitive with one `relation` edge to a target, both in the trace."""
    target = Primitive(name=target_name, provenance=_PROV)
    relatum = Relatum(relation_type=relation, target_id=target.id, target_depth=TGT_DEPTH, provenance=_PROV)
    source = Primitive(name=source_name, depths=[Depth(level=SRC_DEPTH, relata=[relatum], provenance=_PROV)], provenance=_PROV)
    query = EpistemicQuery(concept_ids=[source.id])
    return EpistemicResponse(query=query, result=EpistemicResult(primitives=[source, target]))


def _trace_two_edges(source_name, target_a="file", target_b="dir"):
    """A source primitive with two APPLIES_TO edges (to target_a, target_b), all in the trace."""
    a, b = Primitive(name=target_a, provenance=_PROV), Primitive(name=target_b, provenance=_PROV)
    depth = Depth(
        level=SRC_DEPTH,
        relata=[
            Relatum(relation_type=RelationType.APPLIES_TO, target_id=a.id, target_depth=TGT_DEPTH, provenance=_PROV),
            Relatum(relation_type=RelationType.APPLIES_TO, target_id=b.id, target_depth=TGT_DEPTH, provenance=_PROV),
        ],
        provenance=_PROV,
    )
    source = Primitive(name=source_name, depths=[depth], provenance=_PROV)
    query = EpistemicQuery(concept_ids=[source.id])
    return EpistemicResponse(query=query, result=EpistemicResult(primitives=[source, a, b]))


def _registry(callback, *, source, target="file", key="k", **policy_kwargs) -> PolicyRegistry:
    """A registry with a single placement on the source -> target edge at SRC_DEPTH."""
    reg = PolicyRegistry()
    reg.register(callback, key=key, source_primitive=source, target_primitive=target,
                 source_depth=SRC_DEPTH, **policy_kwargs)
    return reg


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


def test_policy_cardinality_field():
    """trigger_cardinality accepts None and both Cardinality variants."""
    assert Policy(name="A", trigger_cardinality=None).trigger_cardinality is None
    assert Policy(name="S", trigger_cardinality=Cardinality.SINGLE).trigger_cardinality == Cardinality.SINGLE
    assert Policy(name="M", trigger_cardinality=Cardinality.MULTIPLE).trigger_cardinality == Cardinality.MULTIPLE


def test_policy_defaults():
    """Default values are sensible."""
    policy = Policy(name="Minimal")
    assert policy.requires_confirmation is True
    assert policy.trigger_cardinality is None
    assert policy.callback is None
    assert policy.metadata == {}


# ---------------------------------------------------------------------------
# PolicyGate — placement matching & cardinality
# ---------------------------------------------------------------------------


def test_no_placements_proceed():
    """An APPLIES_TO edge with no registered placement → no violations."""
    violations = PolicyGate(PolicyRegistry()).evaluate(_trace("create"), Cardinality.SINGLE)
    assert len(violations) == 0


def test_non_applies_to_relata_ignored():
    """A CONSTRAINED_BY edge is never matched, even with a placement at the same edge."""
    reg = _registry(_cb_fail, source="create", name="ShouldBeIgnored")
    response = _trace("create", relation=RelationType.CONSTRAINED_BY)
    violations = PolicyGate(reg).evaluate(response, Cardinality.SINGLE, tool_call=ToolCallContext(tool_name="t"))
    assert len(violations) == 0


def test_cardinality_single_does_not_trigger_multiple_policy():
    """SINGLE cardinality does not trigger a policy scoped to MULTIPLE → no violations."""
    reg = _registry(_cb_fail, source="delete", name="BulkDelete",
                    trigger_cardinality=Cardinality.MULTIPLE)
    violations = PolicyGate(reg).evaluate(_trace("delete"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert len(violations) == 0


def test_cardinality_multiple_triggers():
    """MULTIPLE cardinality triggers a policy scoped to MULTIPLE → violation produced."""
    reg = _registry(_cb_fail, source="delete", name="BulkDelete",
                    trigger_cardinality=Cardinality.MULTIPLE, confirmation_message="Bulk delete.")
    violations = PolicyGate(reg).evaluate(_trace("delete"), Cardinality.MULTIPLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert len(violations) == 1


def test_cardinality_none_always_fires():
    """trigger_cardinality=None fires for both SINGLE and MULTIPLE."""
    reg = _registry(_cb_fail, source="write", name="AlwaysConfirm", trigger_cardinality=None)
    gate, tc = PolicyGate(reg), ToolCallContext(tool_name="t")
    assert len(gate.evaluate(_trace("write"), Cardinality.SINGLE, tool_call=tc)) == 1
    assert len(gate.evaluate(_trace("write"), Cardinality.MULTIPLE, tool_call=tc)) == 1


def test_multiple_placements_on_same_edge_all_collected():
    """Two placements on the same edge (distinct keys) are both collected."""
    reg = PolicyRegistry()
    reg.register(_cb_fail, key="p1", source_primitive="delete", target_primitive="file",
                 source_depth=SRC_DEPTH, name="Policy1", confirmation_message="First.")
    reg.register(_cb_fail, key="p2", source_primitive="delete", target_primitive="file",
                 source_depth=SRC_DEPTH, name="Policy2", confirmation_message="Second.")
    violations = PolicyGate(reg).evaluate(_trace("delete"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert {v.policy.name for v in violations} == {"Policy1", "Policy2"}


# ---------------------------------------------------------------------------
# PolicyGate — callback evaluation & fail-closed
# ---------------------------------------------------------------------------


def test_callback_passed_true_suppresses_violation():
    """Callback returning passed=True — action passes the policy, no violation."""
    reg = _registry(_cb_pass, source="write", name="WithCallback", confirmation_message="Confirm write.")
    violations = PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="test_fn"))
    assert len(violations) == 0


def test_callback_passed_false_fires_violation():
    """Callback returning passed=False — action fails the policy, violation carries result."""
    reg = _registry(_cb_fail, source="write", name="WithCallback", confirmation_message="Confirm write.")
    violations = PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="test_fn"))
    assert len(violations) == 1
    assert violations[0].callback_result.passed is False


def test_no_tool_call_fires_with_unevaluated_callback():
    """Without a tool_call the callback can't be evaluated → conservative fire with an explicit reason."""
    reg = _registry(_cb_pass, source="write", name="WouldPass", confirmation_message="Confirm.")
    violations = PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE)  # no tool_call
    assert len(violations) == 1
    v = violations[0]
    assert v.callback_result is not None and v.callback_result.passed is False
    assert "no tool call" in v.message
    assert "Confirm." in v.message


def test_callback_raising_fires_with_captured_reason():
    """A callback that raises fails closed (BLOCK) with the exception captured — never propagates (#97)."""
    reg = _registry(_cb_raise, source="write", name="Boom", confirmation_message="Confirm.")
    violations = PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert len(violations) == 1
    v = violations[0]
    assert v.callback_result is not None and v.callback_result.passed is False
    assert "raised" in v.message and "boom" in v.message


def test_callback_wrong_return_type_fires_closed():
    """A callback returning a non-PolicyCallbackResult fails closed, not as a raw exception."""
    reg = _registry(_cb_wrong_type, source="write", name="WrongType", confirmation_message="Confirm.")
    violations = PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert len(violations) == 1
    v = violations[0]
    assert v.callback_result is not None and v.callback_result.passed is False
    assert "PolicyCallbackResult" in v.message  # the TypeError reason is captured


def test_failing_callback_message_leads_violation_message():
    """A callback that fires with a message → that message leads, then the confirmation_message."""
    reg = _registry(_cb_fail, source="write", name="WithReason", confirmation_message="Confirm write.")
    violations = PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="w"))
    assert violations[0].message == "Failed by callback\nConfirm write."


def test_confirmation_message_verbatim():
    """confirmation_message is used verbatim — {action} is NOT interpolated."""
    reg = _registry(_cb_fail, source="delete", name="Confirm",
                    confirmation_message="About to {action} — proceed?")
    violations = PolicyGate(reg).evaluate(_trace("delete"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert violations[0].message == "Failed by callback\nAbout to {action} — proceed?"


def test_requires_confirmation_false_produces_violations():
    """A policy with requires_confirmation=False still produces violations."""
    reg = _registry(_cb_fail, source="list", name="Informational",
                    requires_confirmation=False, confirmation_message="This would normally show.")
    violations = PolicyGate(reg).evaluate(_trace("list"), Cardinality.SINGLE,
                                          tool_call=ToolCallContext(tool_name="t"))
    assert len(violations) == 1
    assert violations[0].requires_confirmation is False


# ---------------------------------------------------------------------------
# PolicyGate — context passed to callbacks
# ---------------------------------------------------------------------------


def test_callback_receives_triggering_edge():
    """The callback sees the source/target names and depths of the edge that fired it."""
    reg = _registry(_cb_record_edge, source="delete", name="EdgeAware")
    _RECORDED_EDGES.clear()
    PolicyGate(reg).evaluate(_trace("delete"), Cardinality.SINGLE, tool_call=ToolCallContext(tool_name="rm"))
    assert len(_RECORDED_EDGES) == 1
    edge = _RECORDED_EDGES[0]
    assert edge.source_name == "delete"
    assert edge.source_depth == SRC_DEPTH
    assert edge.target_depth == TGT_DEPTH


def test_callback_on_two_edges_receives_distinct_targets():
    """One callback registered on two edges is told which target each invocation is for."""
    reg = PolicyRegistry()
    reg.register(_cb_record_edge, key="a", source_primitive="delete", target_primitive="file",
                 source_depth=SRC_DEPTH, name="EdgeAware")
    reg.register(_cb_record_edge, key="b", source_primitive="delete", target_primitive="dir",
                 source_depth=SRC_DEPTH, name="EdgeAware2")
    _RECORDED_EDGES.clear()
    PolicyGate(reg).evaluate(_trace_two_edges("delete"), Cardinality.SINGLE,
                             tool_call=ToolCallContext(tool_name="rm"))
    assert sorted(e.target_name for e in _RECORDED_EDGES) == ["dir", "file"]


def test_callback_reads_policy_metadata():
    """The callback can read its own parameters from the triggering policy's metadata."""
    reg = _registry(_cb_record_metadata, source="send", name="RateLimited", metadata={"limit": 5})
    _RECORDED_META.clear()
    PolicyGate(reg).evaluate(_trace("send"), Cardinality.SINGLE, tool_call=ToolCallContext(tool_name="send"))
    assert _RECORDED_META == [{"limit": 5}]


def test_callback_branches_on_resolved_concepts():
    """A callback can fail based on a co-occurring concept in the grounding facade."""
    reg = _registry(_cb_block_if_protected, source="delete", name="ProtectedAware")
    gate, tc = PolicyGate(reg), ToolCallContext(tool_name="rm")

    safe = GroundingContext(resolved_concepts=["delete", "file"])
    assert len(gate.evaluate(_trace("delete"), Cardinality.SINGLE, tool_call=tc, grounding=safe)) == 0

    risky = GroundingContext(resolved_concepts=["delete", "file", "protected"])
    assert len(gate.evaluate(_trace("delete"), Cardinality.SINGLE, tool_call=tc, grounding=risky)) == 1


def test_callback_receives_tool_call():
    """The caller-supplied tool_call (name + args + kwargs) is threaded to the callback."""
    reg = _registry(_cb_record_tool_call, source="write", name="ToolAware")
    _RECORDED_TOOL_CALLS.clear()
    tc = ToolCallContext(tool_name="write_file", call_args=("a.txt",), call_kwargs={"text": "hi"})
    PolicyGate(reg).evaluate(_trace("write"), Cardinality.SINGLE, tool_call=tc)
    assert len(_RECORDED_TOOL_CALLS) == 1
    recorded = _RECORDED_TOOL_CALLS[0]
    assert recorded.tool_name == "write_file"
    assert recorded.call_args == ("a.txt",)
    assert recorded.call_kwargs == {"text": "hi"}


# ---------------------------------------------------------------------------
# Context types
# ---------------------------------------------------------------------------


def test_tool_call_context_defaults():
    """ToolCallContext carries the invocation; args/kwargs default to empty."""
    tc = ToolCallContext(tool_name="write_file")
    assert tc.tool_name == "write_file"
    assert tc.call_args == ()
    assert tc.call_kwargs == {}


def test_grounding_context_defaults():
    """GroundingContext defaults to no agent and no resolved concepts."""
    gc = GroundingContext()
    assert gc.agent_id is None
    assert gc.resolved_concepts == []


def test_triggering_edge_fields():
    """TriggeringEdge captures source/target name and the two depths."""
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


def test_grounding_context_from_grounding():
    """from_grounding projects only agent_id + resolved from a GroundingResult."""
    from vre.core.grounding import GroundingResult
    gr = GroundingResult(grounded=True, resolved=["a", "b"], gaps=[])
    gc = GroundingContext.from_grounding(gr)
    assert gc.resolved_concepts == ["a", "b"]
    assert gc.agent_id is None


# ---------------------------------------------------------------------------
# Model-owned composition
# ---------------------------------------------------------------------------


def test_policy_callback_result_unevaluable():
    """unevaluable(message) is a fail-closed result carrying the caller's reason."""
    r = PolicyCallbackResult.unevaluable("could not evaluate: no tool call")
    assert r.passed is False
    assert r.message == "could not evaluate: no tool call"


def test_policy_violation_message_property():
    """PolicyViolation.message composes the callback reason + confirmation, or falls back."""
    p = Policy(name="P", confirmation_message="Confirm.")
    assert PolicyViolation(policy=p).message == "Confirm."
    assert PolicyViolation(
        policy=p, callback_result=PolicyCallbackResult(passed=False, message="why")
    ).message == "why\nConfirm."
    assert PolicyViolation(
        policy=p, callback_result=PolicyCallbackResult(passed=False)
    ).message == "Confirm."
