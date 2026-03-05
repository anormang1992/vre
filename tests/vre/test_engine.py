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

    def test_connected_concept_with_relational_gap(self) -> None:
        """Concept connected but target too shallow → RelationalGap, no ReachabilityGap."""
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            # Missing CAPABILITIES and CONSTRAINTS
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

        # Edge is visible (create at D3 >= source_depth D2). Target file at D1 < D3 → RelationalGap.
        relational = [g for g in resp.result.gaps if g.kind == "RELATIONAL"]
        assert len(relational) == 1
        assert relational[0].target.name == "file"
        assert relational[0].required_depth == DepthLevel.CONSTRAINTS
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

    def test_missing_intermediate_depth_produces_depth_gap_via_gated_edge(self) -> None:
        """D0 + D3 present but D1/D2 missing → edge at D2 is gated → DepthGap."""
        target = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        # create has D0 and D3, but D1/D2 absent → contiguous max = D0.
        # Edge lives at D2 (CAPABILITIES) so it's gated.
        p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(target.id, RelationType.APPLIES_TO, DepthLevel.CAPABILITIES),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([p, target]))
        resp = engine.query(["create", "file"])
        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].primitive.name == "create"
        assert depth_gaps[0].required_depth == DepthLevel.CAPABILITIES
        assert depth_gaps[0].current_depth == DepthLevel.EXISTENCE

    def test_no_edges_no_depth_gap_without_min_depth(self) -> None:
        """Root with no edges → no DepthGap (graph structure determines requirements)."""
        p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.CONSTRAINTS),   # D1 and D2 absent, but no edges
        ])
        engine = GroundingEngine(StubRepository([p]))
        resp = engine.query(["create"])
        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 0

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


# ---------------------------------------------------------------------------
# Tests: source depth gating
# ---------------------------------------------------------------------------


class TestSourceDepthGating:
    """Graph-structural depth enforcement via edge source_depth."""

    @staticmethod
    def _gated_delete_and_file():
        """
        Shared fixture: delete has D0+D1 (contiguous max = D1) with a
        relatum at D3 pointing to file. The D2 gap breaks contiguity,
        so the D3 edge is gated.
        """
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        delete_p = _make_primitive("delete", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CAPABILITIES),
            ]),
        ])
        return delete_p, file_p

    def test_edge_at_d3_gated_when_source_at_d1(self) -> None:
        """
        Edge at D3 on source, source contiguous to D1 → DepthGap + ReachabilityGap.
        """
        delete_p, file_p = self._gated_delete_and_file()
        engine = GroundingEngine(StubRepository([delete_p, file_p]))

        resp = engine.query(["delete", "file"])

        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].primitive.name == "delete"
        assert depth_gaps[0].required_depth == DepthLevel.CONSTRAINTS
        assert depth_gaps[0].current_depth == DepthLevel.IDENTITY
        # Gated edge excluded from connectivity → roots are disconnected
        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 1

    def test_edge_at_d2_visible_when_source_at_d2(self) -> None:
        """
        Edge at D2, source contiguous to D2 → visible, no gaps.
        """
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        read_p = _make_primitive("read", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CAPABILITIES),
            ]),
        ])
        engine = GroundingEngine(StubRepository([read_p, file_p]))

        resp = engine.query(["read", "file"])

        assert len(resp.result.gaps) == 0

    def test_visible_edge_with_shallow_target_produces_relational_gap(self) -> None:
        """
        Source sees the edge, but target too shallow → RelationalGap.
        """
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        create_p = _make_primitive("create", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(file_p.id, RelationType.APPLIES_TO, DepthLevel.CAPABILITIES),
            ]),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([create_p, file_p]))

        resp = engine.query(["create", "file"])

        relational = [g for g in resp.result.gaps if g.kind == "RELATIONAL"]
        assert len(relational) == 1
        assert relational[0].source.name == "create"
        assert relational[0].target.name == "file"
        assert relational[0].required_depth == DepthLevel.CAPABILITIES
        assert relational[0].current_depth == DepthLevel.IDENTITY

    def test_min_depth_produces_depth_gap_on_shallow_root(self) -> None:
        """
        min_depth=D3 on a root at D2 → DepthGap even without gated edges.
        """
        read_p = _make_primitive("read", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        engine = GroundingEngine(StubRepository([read_p]))

        resp = engine.query(["read"], min_depth=DepthLevel.CONSTRAINTS)

        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].primitive.name == "read"
        assert depth_gaps[0].required_depth == DepthLevel.CONSTRAINTS
        assert depth_gaps[0].current_depth == DepthLevel.CAPABILITIES

    def test_default_min_depth_no_gap_without_gated_edges(self) -> None:
        """
        No min_depth and no gated edges → no DepthGap.
        """
        read_p = _make_primitive("read", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        engine = GroundingEngine(StubRepository([read_p]))

        resp = engine.query(["read"])

        assert len(resp.result.gaps) == 0

    def test_gated_only_edges_produce_reachability_gap(self) -> None:
        """
        Two roots connected only by gated edges → ReachabilityGap.
        """
        delete_p, file_p = self._gated_delete_and_file()
        engine = GroundingEngine(StubRepository([delete_p, file_p]))

        resp = engine.query(["delete", "file"])

        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 1

    def test_min_depth_cannot_lower_gated_edge_requirement(self) -> None:
        """
        min_depth=D2 does NOT suppress a DepthGap from a gated edge at D3.
        """
        delete_p, file_p = self._gated_delete_and_file()
        engine = GroundingEngine(StubRepository([delete_p, file_p]))

        resp = engine.query(["delete", "file"], min_depth=DepthLevel.CAPABILITIES)

        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].primitive.name == "delete"
        assert depth_gaps[0].required_depth == DepthLevel.CONSTRAINTS  # D3, not D2

    def test_gated_edges_excluded_from_pathway(self) -> None:
        """
        Gated edges do not appear in the response pathway.
        """
        delete_p, file_p = self._gated_delete_and_file()
        engine = GroundingEngine(StubRepository([delete_p, file_p]))

        resp = engine.query(["delete", "file"])

        assert len(resp.result.pathway) == 0
