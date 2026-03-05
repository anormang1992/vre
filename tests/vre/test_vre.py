"""
Unit tests for the VRE public API class.

Uses a stub repository to avoid Neo4j dependency.
"""

from collections import deque
from uuid import UUID, uuid4

from vre import VRE
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    Relatum,
    RelationType,
    ResolvedSubgraph,
)
from vre.core.policy import Cardinality, Policy
from vre.core.grounding import GroundingResult
from vre.core.policy import PolicyResult


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
        """Passing cardinality="multiple" triggers a MULTIPLE-scoped policy."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="multiple")
        assert result.action == "PENDING"

    def test_cardinality_single_does_not_trigger_multiple_scoped_policy(self):
        """Passing cardinality="single" skips a MULTIPLE-scoped policy → PASS."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="single")
        assert result.action == "PASS"

    def test_cardinality_none_triggers_always_on_policy(self):
        """trigger_cardinality=None means the policy always fires regardless of cardinality."""
        vre = self._setup(trigger_cardinality=None)
        assert vre.check_policy(["write", "file"], cardinality="single").action == "PENDING"
        assert vre.check_policy(["write", "file"], cardinality="multiple").action == "PENDING"

    def test_unknown_cardinality_string_falls_back_to_single(self):
        """Unrecognised cardinality string → treated as SINGLE, not an error."""
        vre = self._setup(Cardinality.MULTIPLE)
        result = vre.check_policy(["write", "file"], cardinality="bulk_delete_everything")
        assert result.action == "PASS"  # falls back to SINGLE, MULTIPLE policy skipped


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
        """VRE.check_policy returns a PolicyResult with action PASS, PENDING, or BLOCK."""
        file_p = _make_fully_grounded("file")
        vre = _make_vre_with_stub([file_p])
        result = vre.check_policy(["file"])
        assert isinstance(result, PolicyResult)
        assert result.action in ("PASS", "PENDING", "BLOCK")
