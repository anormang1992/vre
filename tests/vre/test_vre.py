"""
Unit tests for the VRE public API class.

Uses a stub repository to avoid Neo4j dependency.
"""

from collections import deque
from uuid import UUID

from vre import VRE
from vre.core.backends import Repository
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    PrimitiveMetrics,
    Provenance,
    ProvenanceSource,
    Relatum,
    RelationType,
    ResolvedSubgraph,
)
import pytest

from vre.core.errors import PolicyPlacementError, VREError
from vre.core.policy import Cardinality, PolicyAction, PolicyCallbackResult, PolicyResult
from vre.core.policy.callback import ToolCallContext
from vre.core.policy.registry import OrphanedPlacement, PolicyRegistry
from vre.core.grounding import GroundingResult
from vre.learning import LearningEngine


_PROV = Provenance(source=ProvenanceSource.AUTHORED)


def _cb_always_fail(context) -> PolicyCallbackResult:
    return PolicyCallbackResult(passed=False, message="blocked")


# ---------------------------------------------------------------------------
# Stub repository
# ---------------------------------------------------------------------------

_TRANSITIVE_RELS = {RelationType.REQUIRES, RelationType.DEPENDS_ON, RelationType.CONSTRAINED_BY}


class StubRepository(Repository):
    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self._by_id: dict[UUID, Primitive] = {}
        self._by_name: dict[str, Primitive] = {}
        for p in primitives or []:
            self._by_id[p.id] = p
            self._by_name[p.name.lower()] = p

    def list_names(self) -> list[str]:
        return list(self._by_name.keys())

    def find_by_id(self, id: UUID) -> Primitive | None:
        return self._by_id.get(id)

    def find_by_name(self, name: str) -> Primitive | None:
        return self._by_name.get(name.lower())

    def save_primitive(self, primitive: Primitive) -> None:
        self._by_id[primitive.id] = primitive
        self._by_name[primitive.name.lower()] = primitive

    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        prim = self._by_id.get(primitive_id)
        if prim is not None:
            prim.metrics = metrics

    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]:
        return {pid: self._by_id[pid].metrics for pid in primitive_ids if pid in self._by_id}

    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None:
        for pid, metrics in updates.items():
            prim = self._by_id.get(pid)
            if prim is not None:
                prim.metrics = metrics

    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        roots = [self._by_name[n.lower()] for n in names if n.lower() in self._by_name]

        visited: set[UUID] = {r.id for r in roots}
        queue: deque[UUID] = deque(r.id for r in roots)
        while queue:
            uid = queue.popleft()
            prim = self._by_id.get(uid)
            if not prim:
                continue
            for depth in prim.depths:
                for rel in depth.relata:
                    if rel.relation_type not in _TRANSITIVE_RELS:
                        continue
                    if rel.target_id not in visited:
                        visited.add(rel.target_id)
                        queue.append(rel.target_id)

        nodes = [self._by_id[uid] for uid in visited if uid in self._by_id]
        node_ids = {n.id for n in nodes}
        edges: list[EpistemicStep] = []
        for n in nodes:
            for depth in n.depths:
                for rel in depth.relata:
                    if rel.target_id not in node_ids:
                        continue
                    edges.append(EpistemicStep(
                        source_id=n.id,
                        target_id=rel.target_id,
                        relation_type=rel.relation_type,
                        source_depth=depth.level,
                        target_depth=rel.target_depth,
                    ))
        return ResolvedSubgraph(roots=roots, nodes=nodes, edges=edges)

    def delete_primitive(self, id: UUID) -> bool: raise NotImplementedError
    def clear(self) -> int: raise NotImplementedError


def _make_fully_grounded(name: str) -> Primitive:
    return Primitive(name=name, provenance=_PROV, depths=[
        Depth(level=DepthLevel.EXISTENCE, provenance=_PROV),
        Depth(level=DepthLevel.IDENTITY, properties={"_": "identity"}, provenance=_PROV),
        Depth(level=DepthLevel.CAPABILITIES, properties={"_": "capabilities"}, provenance=_PROV),
        Depth(level=DepthLevel.CONSTRAINTS, properties={"_": "constraints"}, provenance=_PROV),
    ])


def _make_vre_with_stub(primitives: list[Primitive], registry: PolicyRegistry | None = None) -> VRE:
    """Create a VRE instance with a stub repository and an isolated (empty) policy registry."""
    repo = StubRepository(primitives)
    return VRE(repo, persist_traces=False, policy_registry=registry or PolicyRegistry())


def _make_primitive_with_edge(
    name: str,
    target: Primitive,
    source_depth: DepthLevel = DepthLevel.CAPABILITIES,
    target_depth: DepthLevel = DepthLevel.CONSTRAINTS,
) -> Primitive:
    """Return a fully-grounded primitive with an APPLIES_TO edge to `target` (no policy data)."""
    relatum = Relatum(
        relation_type=RelationType.APPLIES_TO,
        target_id=target.id,
        target_depth=target_depth,
        provenance=_PROV,
    )
    levels = [DepthLevel.EXISTENCE, DepthLevel.IDENTITY, DepthLevel.CAPABILITIES, DepthLevel.CONSTRAINTS]
    depths = [
        Depth(
            level=lvl,
            properties={} if lvl == DepthLevel.EXISTENCE else {"_": lvl.name.lower()},
            relata=[relatum] if lvl == source_depth else [],
            provenance=_PROV,
        )
        for lvl in levels
    ]
    return Primitive(name=name, provenance=_PROV, depths=depths)


def _registry_with(
    callback,
    *,
    source: str,
    target: str = "file",
    source_depth: DepthLevel = DepthLevel.CAPABILITIES,
    key: str = "k",
    name: str = "TestPolicy",
    **policy_kwargs,
) -> PolicyRegistry:
    """A registry with a single placement on the source -> target edge at source_depth."""
    reg = PolicyRegistry()
    reg.register(callback, key=key, source_primitive=source, target_primitive=target,
                 source_depth=source_depth, name=name, **policy_kwargs)
    return reg


class TestCheckPolicyCardinality:
    """Integration tests: cardinality string wires through to PolicyGate."""

    _TC = ToolCallContext(tool_name="t")

    def _setup(self, trigger_cardinality: Cardinality | None):
        """Return a VRE wired to a write -> file policy with the given trigger."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_always_fail, source="write", target="file",
                              trigger_cardinality=trigger_cardinality, confirmation_message="Confirm?")
        return _make_vre_with_stub([src, target], registry=reg)

    def test_cardinality_multiple_triggers_multiple_scoped_policy(self):
        """Passing cardinality="multiple" triggers a MULTIPLE-scoped policy → BLOCK (no handler)."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="multiple", tool_call=self._TC)
        assert result.action == PolicyAction.BLOCK
        assert len(result.violations) == 1

    def test_cardinality_single_does_not_trigger_multiple_scoped_policy(self):
        """Passing cardinality="single" skips a MULTIPLE-scoped policy → PASS."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="single", tool_call=self._TC)
        assert result.action == PolicyAction.PASS

    def test_cardinality_none_triggers_always_on_policy(self):
        """trigger_cardinality=None means the policy always fires regardless of cardinality → BLOCK."""
        vre = self._setup(trigger_cardinality=None)
        assert vre.check_policy(["write", "file"], cardinality="single", tool_call=self._TC).action == PolicyAction.BLOCK
        assert vre.check_policy(["write", "file"], cardinality="multiple", tool_call=self._TC).action == PolicyAction.BLOCK

    def test_unknown_cardinality_string_fires_all_policies(self):
        """Unrecognised cardinality string → None → all policies fire (safe default)."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="bulk_delete_everything", tool_call=self._TC)
        assert result.action == PolicyAction.BLOCK  # unknown cardinality cannot skip any policy


class TestVRECheck:
    def test_check_grounded_returns_true(self):
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        result = vre.check(["file"])
        assert isinstance(result, GroundingResult)
        assert result.grounded is True

    def test_check_returns_grounding_result(self):
        vre = _make_vre_with_stub([])
        result = vre.check(["unknown_concept"])
        assert isinstance(result, GroundingResult)
        assert result.grounded is False

    def test_check_min_depth_passthrough(self):
        """
        min_depth is forwarded through VRE.check to the engine.
        """
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        # file is at D3 — min_depth=D3 should still pass
        result = vre.check(["file"], min_depth=DepthLevel.CONSTRAINTS)
        assert result.grounded is True
        # min_depth=D4 should produce DepthGap
        result = vre.check(["file"], min_depth=DepthLevel.IMPLICATIONS)
        assert result.grounded is False
        depth_gaps = [g for g in result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].required_depth == DepthLevel.IMPLICATIONS

    def test_check_policy_returns_policy_result(self):
        """VRE.check_policy returns a PolicyResult with action PASS or BLOCK."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        result = vre.check_policy(["file"])
        assert isinstance(result, PolicyResult)
        assert result.action in (PolicyAction.PASS, PolicyAction.BLOCK)

    def test_check_empty_concepts_returns_not_grounded(self):
        """check([]) returns grounded=False with no gaps."""
        vre = _make_vre_with_stub([])
        result = vre.check([])
        assert isinstance(result, GroundingResult)
        assert result.grounded is False
        assert result.gaps == []

    def test_check_policy_with_grounding_result(self):
        """check_policy accepts a pre-computed GroundingResult."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        grounding = vre.check(["file"])
        result = vre.check_policy(grounding)
        assert isinstance(result, PolicyResult)
        assert result.action == PolicyAction.PASS


class TestCheckPolicyOrchestration:
    """Tests for the on_policy orchestration in VRE.check_policy()."""

    _TC = ToolCallContext(tool_name="t")

    def _setup_with_policy(self, requires_confirmation=True):
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_always_fail, source="write", target="file",
                             requires_confirmation=requires_confirmation, confirmation_message="Confirm?")
        return _make_vre_with_stub([src, target], registry=reg)

    def test_on_policy_true_returns_pass(self):
        """on_policy returning True → PASS with violations for observability."""
        vre = self._setup_with_policy(requires_confirmation=True)
        result = vre.check_policy(["write", "file"], tool_call=self._TC, on_policy=lambda violations: True)
        assert result.action == PolicyAction.PASS
        assert len(result.violations) == 1

    def test_on_policy_false_returns_block(self):
        """on_policy returning False → BLOCK."""
        vre = self._setup_with_policy(requires_confirmation=True)
        result = vre.check_policy(["write", "file"], tool_call=self._TC, on_policy=lambda violations: False)
        assert result.action == PolicyAction.BLOCK
        assert result.reason == "User declined"
        assert len(result.violations) == 1

    def test_no_on_policy_confirmation_required_returns_block(self):
        """No on_policy handler + confirmation required → BLOCK (fail-safe)."""
        vre = self._setup_with_policy(requires_confirmation=True)
        result = vre.check_policy(["write", "file"], tool_call=self._TC)
        assert result.action == PolicyAction.BLOCK
        assert "no handler" in result.reason.lower()

    def test_requires_confirmation_false_hard_block(self):
        """requires_confirmation=False violations → BLOCK without consulting on_policy."""
        vre = self._setup_with_policy(requires_confirmation=False)
        called = []
        result = vre.check_policy(["write", "file"], tool_call=self._TC,
                                  on_policy=lambda v: called.append(True) or True)
        assert result.action == PolicyAction.BLOCK
        assert len(result.violations) == 1
        assert called == []  # on_policy should NOT have been called

    def test_on_policy_raising_blocks(self):
        """A raising on_policy handler fails closed → BLOCK with the exception captured (#97)."""
        vre = self._setup_with_policy(requires_confirmation=True)

        def boom(violations):
            raise RuntimeError("handler boom")

        result = vre.check_policy(["write", "file"], tool_call=self._TC, on_policy=boom)
        assert result.action == PolicyAction.BLOCK
        assert "boom" in result.reason

    def test_on_policy_receives_all_violations(self):
        """on_policy receives all violations at once (two placements on the same edge)."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = PolicyRegistry()
        reg.register(_cb_always_fail, key="p1", source_primitive="write", target_primitive="file",
                     source_depth=DepthLevel.CAPABILITIES, name="P1", confirmation_message="First.")
        reg.register(_cb_always_fail, key="p2", source_primitive="write", target_primitive="file",
                     source_depth=DepthLevel.CAPABILITIES, name="P2", confirmation_message="Second.")
        vre = _make_vre_with_stub([src, target], registry=reg)

        received = []

        def handler(violations):
            received.extend(violations)
            return True

        result = vre.check_policy(["write", "file"], tool_call=self._TC, on_policy=handler)
        assert result.action == PolicyAction.PASS
        assert {v.policy.name for v in received} == {"P1", "P2"}


# ---------------------------------------------------------------------------
# VRE property tests
# ---------------------------------------------------------------------------


class TestVREProperties:
    def test_learning_engine_property(self):
        vre = _make_vre_with_stub([])
        assert isinstance(vre.learning_engine, LearningEngine)


# ---------------------------------------------------------------------------
# Agent identity integration
# ---------------------------------------------------------------------------


class TestAgentIdentityIntegration:
    """Integration tests: agent_key stamps agent_id on GroundingResult."""

    def test_check_stamps_agent_id(self, tmp_path):
        """VRE with agent_key stamps agent_id on check() result."""
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, agent_key="test-agent", registry_path=tmp_path / "agents.json", persist_traces=False)
        result = vre.check(["file"])
        assert result.agent_id is not None
        assert result.agent_id == vre.identity.agent_id

    def test_no_identity_leaves_agent_id_none(self):
        """VRE without agent_key leaves agent_id as None."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        result = vre.check(["file"])
        assert result.agent_id is None
        assert vre.identity is None

    def test_same_key_same_uuid_across_instances(self, tmp_path):
        """Two VRE instances with the same agent_key share the same agent_id."""
        file_p = _make_fully_grounded("file")
        path = tmp_path / "agents.json"
        repo = StubRepository([file_p])
        vre1 = VRE(repo, agent_key="shared-agent", registry_path=path, persist_traces=False)
        vre2 = VRE(repo, agent_key="shared-agent", registry_path=path, persist_traces=False)
        assert vre1.identity.agent_id == vre2.identity.agent_id

    def test_agent_name_stored_on_identity(self, tmp_path):
        """VRE with agent_name passes it through to the registered identity."""
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, agent_key="named-agent", agent_name="My Agent", registry_path=tmp_path / "agents.json", persist_traces=False)
        assert vre.identity.name == "My Agent"

    def test_check_policy_with_agent_key_passes(self, tmp_path):
        """check_policy runs without error when VRE has an agent identity."""
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, agent_key="policy-agent", registry_path=tmp_path / "agents.json", persist_traces=False)
        # Pass concepts as list to trigger internal grounding path
        policy_result = vre.check_policy(["file"])
        assert policy_result.action == PolicyAction.PASS


# ---------------------------------------------------------------------------
# Grounding contract: membership is a graph fact (no normalization)
# ---------------------------------------------------------------------------

def test_check_inflected_input_surfaces_existence_gap():
    """'files' is not in the graph; it must surface as an ExistenceGap, not be coerced to 'file'."""
    from vre.core.models import ExistenceGap
    file_p = _make_fully_grounded("file")
    vre = _make_vre_with_stub([file_p])
    result = vre.check(["files"])
    assert result.grounded is False
    assert any(
        isinstance(g, ExistenceGap) and g.primitive.name == "files"
        for g in result.gaps
    )


def test_check_case_insensitive_match_echoes_canonical_casing():
    """'FILE' matches stored 'file' case-insensitively; resolved echoes the canonical casing."""
    file_p = _make_fully_grounded("file")
    vre = _make_vre_with_stub([file_p])
    result = vre.check(["FILE"])
    assert result.grounded is True
    assert result.resolved == ["file"]


def test_check_preserves_order_and_canonical_casing_across_list():
    """Each input in a list maps to its stored canonical casing, in input order."""
    file_p = _make_fully_grounded("file")
    write_p = _make_fully_grounded("write")
    vre = _make_vre_with_stub([file_p, write_p])
    result = vre.check(["FILE", "write"])
    # resolved echoes canonical stored casing for every input, order preserved.
    # (Do NOT assert on `grounded` here — two unconnected roots legitimately
    #  produce a ReachabilityGap; this test is only about the resolved list.)
    assert result.resolved == ["file", "write"]


# ---------------------------------------------------------------------------
# Grounding facade derivation (Task 4)
# ---------------------------------------------------------------------------


_CAPTURED_RESOLVED = []


def _cb_capture_resolved(context):
    _CAPTURED_RESOLVED.append(list(context.grounding.resolved_concepts))
    return PolicyCallbackResult(passed=True)


class TestCheckPolicyGroundingFacade:
    """check_policy mints the GroundingContext facade from the GroundingResult."""

    def test_callback_sees_resolved_concepts(self):
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_capture_resolved, source="write", target="file", name="FacadeAware")
        vre = _make_vre_with_stub([src, target], registry=reg)
        _CAPTURED_RESOLVED.clear()
        vre.check_policy(["write", "file"], tool_call=ToolCallContext(tool_name="write_file"))
        assert _CAPTURED_RESOLVED  # callback was invoked
        assert set(_CAPTURED_RESOLVED[0]) == {"write", "file"}


class TestPolicyPlacementValidation:
    """VRE validates declared placements against the graph at construction (fail loud)."""

    def test_orphaned_placement_raises_at_init(self):
        """A declared placement whose edge is absent → PolicyPlacementError at construction."""
        target = _make_fully_grounded("file")
        reg = _registry_with(_cb_always_fail, source="delete", target="file", name="Orphan")
        with pytest.raises(PolicyPlacementError, match="Orphan"):
            VRE(StubRepository([target]), persist_traces=False, policy_registry=reg)

    def test_validate_policies_false_suppresses_and_returns_orphan(self):
        """validate_policies=False skips the init raise; the method still reports the orphan."""
        target = _make_fully_grounded("file")
        reg = _registry_with(_cb_always_fail, source="delete", target="file", name="Orphan")
        vre = VRE(StubRepository([target]), persist_traces=False, policy_registry=reg, validate_policies=False)
        orphans = vre.validate_policy_placements()
        assert len(orphans) == 1
        assert isinstance(orphans[0], OrphanedPlacement)
        assert orphans[0].name == "Orphan"

    def test_correct_placement_constructs_and_validates_clean(self):
        """A placement matching a real edge constructs and validates to []."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_always_fail, source="write", target="file", name="Good")
        vre = _make_vre_with_stub([src, target], registry=reg)
        assert vre.validate_policy_placements() == []

    def test_wrong_depth_is_orphaned(self):
        """A placement at a depth the edge does not live at → orphaned (fail loud)."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)  # edge lives at CAPABILITIES
        reg = _registry_with(_cb_always_fail, source="write", target="file",
                             source_depth=DepthLevel.CONSTRAINTS, name="WrongDepth")
        with pytest.raises(PolicyPlacementError):
            VRE(StubRepository([src, target]), persist_traces=False, policy_registry=reg)

    def test_expect_policies_mismatch_raises(self):
        """expect_policies that does not match the registered count → VREError."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_always_fail, source="write", target="file", name="Good")
        with pytest.raises(VREError, match="expected 2"):
            VRE(StubRepository([src, target]), persist_traces=False, policy_registry=reg, expect_policies=2)

    def test_expect_policies_match_constructs(self):
        """expect_policies matching the registered count constructs cleanly."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_always_fail, source="write", target="file", name="Good")
        vre = VRE(StubRepository([src, target]), persist_traces=False, policy_registry=reg, expect_policies=1)
        assert vre.validate_policy_placements() == []

    def test_registry_frozen_after_construction(self):
        """Once VRE is constructed, registering another policy on its registry raises."""
        target = _make_fully_grounded("file")
        src = _make_primitive_with_edge("write", target)
        reg = _registry_with(_cb_always_fail, source="write", target="file", name="Good")
        _make_vre_with_stub([src, target], registry=reg)
        with pytest.raises(VREError, match="before constructing VRE"):
            reg.register(_cb_always_fail, key="late", source_primitive="write", target_primitive="file",
                         source_depth=DepthLevel.CAPABILITIES, name="Late")

    def test_multiple_graphs_use_separate_registries(self):
        """Two VREs on disjoint graphs each validate/freeze only their own registry."""
        file_a = _make_fully_grounded("file")
        src_a = _make_primitive_with_edge("write", file_a)
        reg_a = _registry_with(_cb_always_fail, source="write", target="file", name="A")

        email_b = _make_fully_grounded("email")
        src_b = _make_primitive_with_edge("send", email_b)
        reg_b = _registry_with(_cb_always_fail, source="send", target="email", name="B")

        # Constructing vre_a validates reg_a against graph A and freezes reg_a only —
        # it must not freeze reg_b or validate B's policy against A's graph.
        vre_a = VRE(StubRepository([src_a, file_a]), persist_traces=False, policy_registry=reg_a)
        vre_b = VRE(StubRepository([src_b, email_b]), persist_traces=False, policy_registry=reg_b)

        assert vre_a.validate_policy_placements() == []
        assert vre_b.validate_policy_placements() == []


class TestCheckPolicyFailClosed:
    """check_policy fails closed on ungrounded/trace-less input (#93)."""

    def test_ungrounded_grounding_result_blocks(self):
        """An ungrounded GroundingResult → BLOCK, never a green PASS."""
        from vre.core.models import ExistenceGap
        ungrounded = GroundingResult(
            grounded=False, resolved=["nope"],
            gaps=[ExistenceGap(primitive=Primitive(name="nope", depths=[], provenance=_PROV))],
        )
        result = _make_vre_with_stub([]).check_policy(ungrounded)
        assert result.action == PolicyAction.BLOCK
        assert "grounded" in (result.reason or "")

    def test_ungrounded_concepts_block(self):
        """An ungrounded concept list → BLOCK, not PASS over a partial trace."""
        result = _make_vre_with_stub([]).check_policy(["unknown_concept"])
        assert result.action == PolicyAction.BLOCK

    def test_empty_concepts_block(self):
        """Empty concepts ground to nothing → BLOCK (was the old trace-None fail-open PASS)."""
        result = _make_vre_with_stub([]).check_policy([])
        assert result.action == PolicyAction.BLOCK

    def test_grounded_without_trace_blocks(self):
        """A hand-built grounded=True result with no trace is malformed → BLOCK, never PASS or crash."""
        circumvention = GroundingResult(grounded=True, resolved=["file"], gaps=[])  # trace defaults None
        result = _make_vre_with_stub([]).check_policy(circumvention)
        assert result.action == PolicyAction.BLOCK
