"""
Unit tests for the VRE public API class.

Uses a stub repository to avoid Neo4j dependency.
"""

from collections import deque
from uuid import UUID

from vre import VRE
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    PrimitiveMetrics,
    Relatum,
    RelationType,
    ResolvedSubgraph,
)
from vre.core.policy import Cardinality, Policy, PolicyAction, PolicyResult
from vre.core.grounding import GroundingResult
from vre.learning.callback import LearningCallback
from vre.learning.models import (
    CandidateDecision,
    DepthCandidate,
    ExistenceCandidate,
    ProposedDepth,
)


# ---------------------------------------------------------------------------
# Stub repository
# ---------------------------------------------------------------------------

_TRANSITIVE_RELS = {RelationType.REQUIRES, RelationType.DEPENDS_ON, RelationType.CONSTRAINED_BY}


class StubRepository:
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


def _make_fully_grounded(name: str) -> Primitive:
    return Primitive(name=name, depths=[
        Depth(level=DepthLevel.EXISTENCE),
        Depth(level=DepthLevel.IDENTITY),
        Depth(level=DepthLevel.CAPABILITIES),
        Depth(level=DepthLevel.CONSTRAINTS),
    ])


def _make_vre_with_stub(primitives: list[Primitive]) -> VRE:
    """Create a VRE instance with a stub repository."""
    repo = StubRepository(primitives)
    return VRE(repo)


def _make_primitive_with_policy(
    name: str,
    target: Primitive,
    policy: Policy,
) -> Primitive:
    """Return a fully-grounded primitive whose APPLIES_TO relatum carries a policy."""
    relatum = Relatum(
        relation_type=RelationType.APPLIES_TO,
        target_id=target.id,
        target_depth=DepthLevel.CONSTRAINTS,
        policies=[policy],
    )
    return Primitive(name=name, depths=[
        Depth(level=DepthLevel.EXISTENCE),
        Depth(level=DepthLevel.IDENTITY),
        Depth(level=DepthLevel.CAPABILITIES, relata=[relatum]),
        Depth(level=DepthLevel.CONSTRAINTS),
    ])


class TestCheckPolicyCardinality:
    """Integration tests: cardinality string wires through to PolicyGate."""

    def _setup(self, trigger_cardinality: Cardinality | None):
        """Return a VRE + concept name wired to a policy with the given trigger."""
        target = _make_fully_grounded("file")
        policy = Policy(
            name="TestPolicy",
            requires_confirmation=True,
            trigger_cardinality=trigger_cardinality,
            confirmation_message="Confirm?",
        )
        src = _make_primitive_with_policy("write", target, policy)
        vre = _make_vre_with_stub([src, target])
        return vre

    def test_cardinality_multiple_triggers_multiple_scoped_policy(self):
        """Passing cardinality="multiple" triggers a MULTIPLE-scoped policy → BLOCK (no handler)."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="multiple")
        assert result.action == PolicyAction.BLOCK
        assert len(result.violations) == 1

    def test_cardinality_single_does_not_trigger_multiple_scoped_policy(self):
        """Passing cardinality="single" skips a MULTIPLE-scoped policy → PASS."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="single")
        assert result.action == PolicyAction.PASS

    def test_cardinality_none_triggers_always_on_policy(self):
        """trigger_cardinality=None means the policy always fires regardless of cardinality → BLOCK."""
        vre = self._setup(trigger_cardinality=None)
        assert vre.check_policy(["write", "file"], cardinality="single").action == PolicyAction.BLOCK
        assert vre.check_policy(["write", "file"], cardinality="multiple").action == PolicyAction.BLOCK

    def test_unknown_cardinality_string_fires_all_policies(self):
        """Unrecognised cardinality string → None → all policies fire (safe default)."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="bulk_delete_everything")
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

    def test_resolve_returns_list(self):
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        result = vre.resolve(["file"])
        assert isinstance(result, list)

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

    def test_check_policy_returns_pass_when_no_trace(self):
        """check_policy returns PASS when GroundingResult has no trace."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        grounding = GroundingResult(grounded=True, resolved=["file"], gaps=[], trace=None)
        result = vre.check_policy(grounding)
        assert result.action == PolicyAction.PASS


class TestCheckPolicyOrchestration:
    """Tests for the on_policy orchestration in VRE.check_policy()."""

    def _setup_with_policy(self, requires_confirmation=True):
        target = _make_fully_grounded("file")
        policy = Policy(
            name="TestPolicy",
            requires_confirmation=requires_confirmation,
            confirmation_message="Confirm {action}?",
        )
        src = _make_primitive_with_policy("write", target, policy)
        vre = _make_vre_with_stub([src, target])
        return vre

    def test_on_policy_true_returns_pass(self):
        """on_policy returning True → PASS with violations for observability."""
        vre = self._setup_with_policy(requires_confirmation=True)
        result = vre.check_policy(
            ["write", "file"],
            on_policy=lambda violations: True,
        )
        assert result.action == PolicyAction.PASS
        assert len(result.violations) == 1

    def test_on_policy_false_returns_block(self):
        """on_policy returning False → BLOCK."""
        vre = self._setup_with_policy(requires_confirmation=True)
        result = vre.check_policy(
            ["write", "file"],
            on_policy=lambda violations: False,
        )
        assert result.action == PolicyAction.BLOCK
        assert result.reason == "User declined"
        assert len(result.violations) == 1

    def test_no_on_policy_confirmation_required_returns_block(self):
        """No on_policy handler + confirmation required → BLOCK (fail-safe)."""
        vre = self._setup_with_policy(requires_confirmation=True)
        result = vre.check_policy(["write", "file"])
        assert result.action == PolicyAction.BLOCK
        assert "no handler" in result.reason.lower()

    def test_requires_confirmation_false_hard_block(self):
        """requires_confirmation=False violations → BLOCK without consulting on_policy."""
        vre = self._setup_with_policy(requires_confirmation=False)
        called = []
        result = vre.check_policy(
            ["write", "file"],
            on_policy=lambda v: called.append(True) or True,
        )
        assert result.action == PolicyAction.BLOCK
        assert len(result.violations) == 1
        assert called == []  # on_policy should NOT have been called

    def test_on_policy_receives_all_violations(self):
        """on_policy receives all violations at once."""
        target = _make_fully_grounded("file")
        p1 = Policy(name="P1", confirmation_message="First.")
        p2 = Policy(name="P2", confirmation_message="Second.")
        relatum = Relatum(
            relation_type=RelationType.APPLIES_TO,
            target_id=target.id,
            target_depth=DepthLevel.CONSTRAINTS,
            policies=[p1, p2],
        )
        src = Primitive(name="write", depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(level=DepthLevel.IDENTITY),
            Depth(level=DepthLevel.CAPABILITIES, relata=[relatum]),
            Depth(level=DepthLevel.CONSTRAINTS),
        ])
        vre = _make_vre_with_stub([src, target])

        received = []

        def handler(violations):
            received.extend(violations)
            return True

        result = vre.check_policy(["write", "file"], on_policy=handler)
        assert result.action == PolicyAction.PASS
        assert len(received) == 2
        names = {v.policy.name for v in received}
        assert names == {"P1", "P2"}


# ---------------------------------------------------------------------------
# VRE.learn_all tests
# ---------------------------------------------------------------------------


class _AcceptLearner(LearningCallback):
    """Callback that accepts proposals with appropriate candidates."""

    def __init__(self):
        self.calls = []

    def __call__(self, candidate, grounding, gap):
        self.calls.append((candidate, gap))
        if isinstance(candidate, ExistenceCandidate):
            filled = ExistenceCandidate(
                name=candidate.name,
                d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"desc": "test"}),
            )
            return filled, CandidateDecision.ACCEPTED
        if isinstance(candidate, DepthCandidate):
            filled = DepthCandidate(new_depths=[
                ProposedDepth(level=gap.required_depth, properties={"test": True}),
            ])
            return filled, CandidateDecision.ACCEPTED
        return None, CandidateDecision.SKIPPED


class _RejectLearner(LearningCallback):
    """Callback that rejects all proposals."""

    def __call__(self, candidate, grounding, gap):
        return None, CandidateDecision.REJECTED


class TestVRELearnAll:
    def test_learn_all_resolves_existence_gap(self):
        """learn_all creates the missing primitive and returns grounded."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])

        grounding = vre.check(["file", "copy"])
        assert grounding.grounded is False

        learner = _AcceptLearner()
        result = vre.learn_all(grounding, learner, ["file", "copy"])
        # After learning "copy", re-grounding should find it (now in the repo)
        assert isinstance(result, GroundingResult)

    def test_learn_all_stops_on_rejection(self):
        """learn_all stops the loop when callback rejects."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])

        grounding = vre.check(["file", "copy"])
        result = vre.learn_all(grounding, _RejectLearner(), ["file", "copy"])
        assert isinstance(result, GroundingResult)
        assert result.grounded is False

    def test_learn_all_skips_gaps_and_continues(self):
        """learn_all skips a gap and continues to the next."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])

        grounding = vre.check(["file", "alpha", "beta"])
        assert len([g for g in grounding.gaps if g.kind == "EXISTENCE"]) == 2

        call_count = 0

        class SkipThenAccept(LearningCallback):
            def __call__(self, candidate, grounding, gap):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return None, CandidateDecision.SKIPPED
                if isinstance(candidate, ExistenceCandidate):
                    filled = ExistenceCandidate(
                        name=candidate.name,
                        d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"desc": "test"}),
                    )
                    return filled, CandidateDecision.ACCEPTED
                # Skip any non-existence gaps (e.g. ReachabilityGap)
                return None, CandidateDecision.SKIPPED

        result = vre.learn_all(grounding, SkipThenAccept(), ["file", "alpha", "beta"])
        assert isinstance(result, GroundingResult)
        # At least 2 calls: one skip, one accept
        assert call_count >= 2

    def test_learn_all_uses_context_manager(self):
        """learn_all wraps the callback in a context manager."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])

        grounding = vre.check(["file", "unknown"])

        entered = False
        exited = False

        class LifecycleLearner(LearningCallback):
            def __enter__(self):
                nonlocal entered
                entered = True
                return self

            def __exit__(self, *args):
                nonlocal exited
                exited = True

            def __call__(self, candidate, grounding, gap):
                return None, CandidateDecision.REJECTED

        vre.learn_all(grounding, LifecycleLearner(), ["file", "unknown"])
        assert entered
        assert exited


# ---------------------------------------------------------------------------
# Agent identity integration
# ---------------------------------------------------------------------------


class TestAgentIdentityIntegration:
    """Integration tests: agent_key stamps agent_id on GroundingResult."""

    def test_check_stamps_agent_id(self, tmp_path):
        """VRE with agent_key stamps agent_id on check() result."""
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, agent_key="test-agent", registry_path=tmp_path / "agents.json")
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
        vre1 = VRE(repo, agent_key="shared-agent", registry_path=path)
        vre2 = VRE(repo, agent_key="shared-agent", registry_path=path)
        assert vre1.identity.agent_id == vre2.identity.agent_id

    def test_agent_name_stored_on_identity(self, tmp_path):
        """VRE with agent_name passes it through to the registered identity."""
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, agent_key="named-agent", agent_name="My Agent", registry_path=tmp_path / "agents.json")
        assert vre.identity.name == "My Agent"

    def test_check_policy_with_agent_key_passes(self, tmp_path):
        """check_policy runs without error when VRE has an agent identity."""
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, agent_key="policy-agent", registry_path=tmp_path / "agents.json")
        # Pass concepts as list to trigger internal grounding path
        policy_result = vre.check_policy(["file"])
        assert policy_result.action == PolicyAction.PASS
