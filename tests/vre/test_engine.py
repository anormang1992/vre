"""
Unit tests for the VRE GroundingEngine.

Uses a StubRepository to avoid Neo4j dependency.
"""

from collections import deque
from uuid import UUID, uuid4

from vre.core.grounding import GroundingEngine
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    Relatum,
    RelationType,
    ResolvedSubgraph,
)


# ---------------------------------------------------------------------------
# Stub repository
# ---------------------------------------------------------------------------

_TRANSITIVE_RELS = {RelationType.REQUIRES, RelationType.DEPENDS_ON, RelationType.CONSTRAINED_BY}


class StubRepository:
    """
    In-memory stand-in for PrimitiveRepository.
    Supports lookup by name (case-insensitive) and by UUID.
    Implements resolve_subgraph with BFS and relationship type filtering.
    """

    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self._by_id: dict[UUID, Primitive] = {}
        self._by_name: dict[str, Primitive] = {}
        for p in primitives or []:
            self._by_id[p.id] = p
            self._by_name[p.name.lower()] = p

    def list_names(self) -> list[str]:
        return list(self._by_name.keys())

    def resolve_subgraph(
        self,
        names: list[str],
    ) -> ResolvedSubgraph:
        roots = [self._by_name[n.lower()] for n in names if n.lower() in self._by_name]

        # BFS from all roots — follows TRANSITIVE_RELS at any depth
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

        nodes: list[Primitive] = []
        for uid in visited:
            p = self._by_id.get(uid)
            if p:
                nodes.append(p)
            else:
                nodes.append(Primitive(id=uid, name="<unknown>", depths=[]))

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_primitive(
    name: str,
    depths: list[Depth] | None = None,
    id: UUID | None = None,
) -> Primitive:
    return Primitive(id=id or uuid4(), name=name, depths=depths or [])


def _depth(level: DepthLevel, relata: list[Relatum] | None = None) -> Depth:
    return Depth(level=level, relata=relata or [])


def _relatum(
    target_id: UUID,
    rel_type: RelationType = RelationType.APPLIES_TO,
    target_depth: DepthLevel = DepthLevel.CAPABILITIES,
) -> Relatum:
    return Relatum(
        relation_type=rel_type,
        target_id=target_id,
        target_depth=target_depth,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRelatumDepthGap:
    """Edge-level depth failures become RelationalGaps, not DepthGaps."""

    def test_relatum_depth_gap_when_target_shallow(self) -> None:
        """Relatum demands D2 on target, but target only has D1 → RelationalGap."""
        b = _make_primitive("B", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        a = _make_primitive("A", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(b.id, RelationType.REQUIRES, DepthLevel.CAPABILITIES),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([a, b]))

        resp = engine.query(["A"])

        relational = [g for g in resp.result.gaps if g.kind == "RELATIONAL"]
        assert len(relational) == 1
        gap = relational[0]
        assert gap.source.name == "A"
        assert gap.target.name == "B"
        assert gap.required_depth == DepthLevel.CAPABILITIES
        assert gap.current_depth == DepthLevel.IDENTITY

    def test_no_relatum_depth_gap_when_target_sufficient(self) -> None:
        """Relatum demands D2, target has D2 → no RelationalGap."""
        b = _make_primitive("B", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _make_primitive("A", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(b.id, RelationType.REQUIRES, DepthLevel.CAPABILITIES),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([a, b]))

        resp = engine.query(["A"])

        relational = [g for g in resp.result.gaps if g.kind == "RELATIONAL"]
        assert len(relational) == 0

    def test_relatum_depth_deduplication(self) -> None:
        """Two edges A→B requiring D2 and D3 → one RelationalGap with required_depth=D3."""
        b = _make_primitive("B", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        a = _make_primitive("A", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(b.id, RelationType.REQUIRES, DepthLevel.CAPABILITIES),
            ]),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(b.id, RelationType.REQUIRES, DepthLevel.CONSTRAINTS),
            ]),
        ])
        engine = GroundingEngine(StubRepository([a, b]))

        resp = engine.query(["A"])

        relational = [g for g in resp.result.gaps if g.kind == "RELATIONAL"]
        assert len(relational) == 1
        assert relational[0].required_depth == DepthLevel.CONSTRAINTS


# ---------------------------------------------------------------------------
# Tests: query (flat-concept undirected connectivity model)
# ---------------------------------------------------------------------------


class TestFlatQuery:
    """query() uses undirected connected-component check over collected edges."""

    def test_unknown_concept_gets_existence_gap(self) -> None:
        """Unknown concept → ExistenceGap; grounded concept alone → no ReachabilityGap."""
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([file_p]))

        resp = engine.query(["compile", "file"])

        existence_gaps = [g for g in resp.result.gaps if g.kind == "EXISTENCE"]
        assert any(g.primitive.name == "compile" for g in existence_gaps)
        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 0

    def test_unknown_concept_gets_existence_gap_no_reachability(self) -> None:
        """Concept not found → ExistenceGap (transient), no ReachabilityGap."""
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p]))

        resp = engine.query(["create", "widget"])

        existence_gaps = [g for g in resp.result.gaps if g.kind == "EXISTENCE"]
        assert any(g.primitive.name == "widget" for g in existence_gaps)
        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 0

    def test_connected_concepts_grounded(self) -> None:
        """Two concepts connected by an edge → no ReachabilityGap."""
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CONSTRAINTS),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p, file_p]))

        resp = engine.query(["create", "file"])

        assert len(resp.result.gaps) == 0

    def test_transitively_connected_concepts_grounded(self) -> None:
        """Three concepts connected via a chain → no ReachabilityGap."""
        permission_p = _make_primitive("permission", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(permission_p.id, RelationType.REQUIRES, DepthLevel.CONSTRAINTS),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CONSTRAINTS),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p, file_p, permission_p]))

        resp = engine.query(["create", "file", "permission"])

        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 0

    def test_disconnected_concept_gets_reachability_gap(self) -> None:
        """Concept in graph but isolated → ReachabilityGap."""
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CONSTRAINTS),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        # "network" is in the graph but has no connection to create or file
        network_p = _make_primitive("network", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p, file_p, network_p]))

        resp = engine.query(["create", "file", "network"])

        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 1
        assert reachability_gaps[0].primitive.name == "network"

    def test_connected_concept_with_depth_gap(self) -> None:
        """Concept connected but insufficient depth → DepthGap, no ReachabilityGap."""
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            # Missing CAPABILITIES and CONSTRAINTS — always requires D3
        ])
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CONSTRAINTS),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p, file_p]))

        resp = engine.query(["create", "file"])

        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert any(g.primitive.name == "file" for g in depth_gaps)
        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 0

    def test_single_concept_grounded(self) -> None:
        """Single fully-grounded concept → no gaps."""
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p]))

        resp = engine.query(["create"])

        assert len(resp.result.gaps) == 0


# ---------------------------------------------------------------------------
# Tests: monotonic contiguous depth validation
# ---------------------------------------------------------------------------


class TestMonotonicDepth:

    def test_missing_intermediate_depth_is_a_gap(self) -> None:
        """D0 + D3 present but D1/D2 missing → DepthGap with current_depth=D0."""
        p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.CONSTRAINTS),   # D1 and D2 absent
        ])
        engine = GroundingEngine(StubRepository([p]))
        resp = engine.query(["create"])
        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].current_depth == DepthLevel.EXISTENCE

    def test_contiguous_depths_pass_grounding(self) -> None:
        """D0 through D3 all present → no gaps."""
        p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([p]))
        resp = engine.query(["create"])
        assert len(resp.result.gaps) == 0

    def test_d3_edge_visible_in_trace(self) -> None:
        """CONSTRAINED_BY at D3 appears in pathway."""
        permission_p = _make_primitive("permission", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        directory_p = _make_primitive("directory", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(permission_p.id, RelationType.CONSTRAINED_BY, DepthLevel.CONSTRAINTS),
            ]),
        ])
        engine = GroundingEngine(StubRepository([directory_p, permission_p]))
        resp = engine.query(["directory", "permission"])
        assert len([g for g in resp.result.gaps if g.kind == "REACHABILITY"]) == 0
        assert RelationType.CONSTRAINED_BY in {e.relation_type for e in resp.result.pathway}

    def test_d3_edge_pulls_unseen_node_into_subgraph(self) -> None:
        """BFS follows CONSTRAINED_BY at D3 to discover a node not explicitly submitted."""
        permission_p = _make_primitive("permission", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        directory_p = _make_primitive("directory", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(permission_p.id, RelationType.CONSTRAINED_BY, DepthLevel.CONSTRAINTS),
            ]),
        ])
        engine = GroundingEngine(StubRepository([directory_p, permission_p]))
        # Only "directory" submitted — permission discovered via BFS
        resp = engine.query(["directory"])
        primitive_names = {p.name for p in resp.result.primitives}
        assert "permission" in primitive_names
