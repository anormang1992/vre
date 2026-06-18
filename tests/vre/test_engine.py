"""
Unit tests for the VRE GroundingEngine.

Uses a StubRepository to avoid Neo4j dependency.
"""

from collections import deque
from uuid import UUID, uuid4

import pytest

from vre.core.backends import Repository
from vre.core.errors import GraphIntegrityError
from vre.core.grounding import GroundingEngine
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

_PROV = Provenance(source=ProvenanceSource.AUTHORED)


# ---------------------------------------------------------------------------
# Stub repository
# ---------------------------------------------------------------------------

_TRANSITIVE_RELS = {RelationType.REQUIRES, RelationType.DEPENDS_ON, RelationType.CONSTRAINED_BY}


class StubRepository(Repository):
    """
    In-memory stand-in for Repository.
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
                nodes.append(Primitive(id=uid, name="<unknown>", depths=[], provenance=_PROV))

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

    def find_by_id(self, id: UUID) -> Primitive | None:
        raise NotImplementedError

    def find_by_name(self, name: str) -> Primitive | None:
        raise NotImplementedError

    def save_primitive(self, primitive: Primitive) -> None:
        raise NotImplementedError

    def delete_primitive(self, id: UUID) -> bool:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        raise NotImplementedError

    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]:
        raise NotImplementedError

    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_primitive(
    name: str,
    depths: list[Depth] | None = None,
    id: UUID | None = None,
) -> Primitive:
    return Primitive(id=id or uuid4(), name=name, depths=depths or [], provenance=_PROV)


def _depth(level: DepthLevel, relata: list[Relatum] | None = None) -> Depth:
    # Non-D0 depths carry content so they survive the vacuity floor (#80);
    # D0 grounds bare regardless.
    return Depth(level=level, properties={"_": level.name}, relata=relata or [], provenance=_PROV)


def _relatum(
    target_id: UUID,
    rel_type: RelationType = RelationType.APPLIES_TO,
    target_depth: DepthLevel = DepthLevel.CAPABILITIES,
) -> Relatum:
    return Relatum(
        relation_type=rel_type,
        target_id=target_id,
        target_depth=target_depth,
        provenance=_PROV,
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

    def test_unknown_concept_transient_carries_synthetic_provenance(self) -> None:
        """The transient placeholder for an absent concept is stamped SYNTHETIC —
        engine-generated knowledge of an absence, not a forged human genealogy."""
        engine = GroundingEngine(StubRepository([]))

        resp = engine.query(["frobnicate"])

        gap = next(g for g in resp.result.gaps if g.kind == "EXISTENCE")
        assert gap.primitive.name == "frobnicate"
        assert gap.primitive.provenance.source == ProvenanceSource.SYNTHETIC
        assert "frobnicate" in gap.primitive.provenance.detail

    def test_repeated_unknown_concept_dedupes_to_one_transient(self) -> None:
        """A repeated or case-variant unknown concept resolves to a single
        transient placeholder — one synthetic id, one ExistenceGap, one trace
        primitive, one query root id — not one per occurrence (#130)."""
        engine = GroundingEngine(StubRepository([]))

        for concepts in (["widget", "widget"], ["Widget", "widget"]):
            resp = engine.query(concepts)

            existence_gaps = [g for g in resp.result.gaps if g.kind == "EXISTENCE"]
            assert len(existence_gaps) == 1
            assert len({g.primitive.id for g in existence_gaps}) == 1
            widgets = [p for p in resp.result.primitives if p.name_lower == "widget"]
            assert len(widgets) == 1
            assert resp.query.concept_ids == [existence_gaps[0].primitive.id]

    def test_repeated_known_concept_dedupes_concept_ids(self) -> None:
        """A repeated or case-variant known concept collapses to a single query
        root id. Folded-name dedup runs over the raw input, so concept_ids never
        carries duplicates for known concepts either (#130)."""
        file_p = _make_primitive("file", [_depth(DepthLevel.EXISTENCE)])
        engine = GroundingEngine(StubRepository([file_p]))

        for concepts in (["file", "file"], ["File", "file"]):
            resp = engine.query(concepts)
            assert resp.query.concept_ids == [file_p.id]

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
        Gated edges do not appear in the pathway, and the queried roots are
        preserved in primitives even when connected only by a gated edge
        (#94: a queried root is never pruned).
        """
        delete_p, file_p = self._gated_delete_and_file()
        engine = GroundingEngine(StubRepository([delete_p, file_p]))

        resp = engine.query(["delete", "file"])

        assert len(resp.result.pathway) == 0
        assert {p.name for p in resp.result.primitives} == {"delete", "file"}


class TestJustifiedEnvelope:
    """The response surfaces only grounded content (#94 Findings A + D1)."""

    @staticmethod
    def _delete_constrained_by_permission(constraint_target_depth: DepthLevel):
        """delete (D0,D1,D3) with a CONSTRAINED_BY @ D3 edge to a full permission."""
        permission = _make_primitive("permission", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        delete = _make_primitive("delete", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(permission.id, RelationType.CONSTRAINED_BY, constraint_target_depth),
            ]),
        ])
        return delete, permission

    def test_gated_reached_nonroot_node_pruned_from_primitives(self) -> None:
        """
        A non-root node reached only through a gated edge never enters
        result.primitives. The block is surfaced on the source (which the agent
        can see), never as a gap that names the hidden target.
        """
        delete, permission = self._delete_constrained_by_permission(DepthLevel.CONSTRAINTS)
        engine = GroundingEngine(StubRepository([delete, permission]))

        resp = engine.query(["delete"])

        assert {p.name for p in resp.result.primitives} == {"delete"}
        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        assert len(depth_gaps) == 1
        assert depth_gaps[0].primitive.name == "delete"
        assert depth_gaps[0].required_depth == DepthLevel.CONSTRAINTS
        assert all(g.primitive.name != "permission" for g in resp.result.gaps)

    def test_gated_reached_node_makes_result_ungrounded(self) -> None:
        """The same case grounds False — the action is blocked, the target unseen."""
        delete, permission = self._delete_constrained_by_permission(DepthLevel.IDENTITY)
        engine = GroundingEngine(StubRepository([delete, permission]))

        result = engine.ground(["delete"])

        assert result.grounded is False
        assert "permission" not in {p.name for p in result.trace.result.primitives}

    def test_multihop_gated_chain_emits_only_frontier_gap(self) -> None:
        """
        a --gated--> b --gated--> c, querying a alone. Only the frontier gap
        (on a) is emitted; b and c are pruned and neither is named by any gap,
        so the agent never learns about structure it cannot yet reach.
        """
        c = _make_primitive("c", [_depth(DepthLevel.EXISTENCE), _depth(DepthLevel.IDENTITY)])
        b = _make_primitive("b", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(c.id, RelationType.REQUIRES, DepthLevel.IDENTITY),
            ]),
        ])
        a = _make_primitive("a", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(b.id, RelationType.REQUIRES, DepthLevel.IDENTITY),
            ]),
        ])
        engine = GroundingEngine(StubRepository([a, b, c]))

        resp = engine.query(["a"])

        assert {p.name for p in resp.result.primitives} == {"a"}
        assert len(resp.result.gaps) == 1
        gap = resp.result.gaps[0]
        assert gap.kind == "DEPTH"
        assert gap.primitive.name == "a"

    def test_visible_reached_nonroot_node_survives(self) -> None:
        """A non-root node reached through a *visible* edge stays in the result."""
        b = _make_primitive("b", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _make_primitive("a", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, [
                _relatum(b.id, RelationType.REQUIRES, DepthLevel.CAPABILITIES),
            ]),
        ])
        engine = GroundingEngine(StubRepository([a, b]))

        resp = engine.query(["a"])

        assert {p.name for p in resp.result.primitives} == {"a", "b"}

    def test_noncontiguous_depth_content_stripped_from_trace(self) -> None:
        """
        {D0, D1, D3} grounds green to D1, and the D3 content is absent from the
        trace — nothing above contiguous_max_depth surfaces (#94 Finding D1).
        """
        widget = _make_primitive("widget", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS),  # D3 with D2 missing → above contiguous max
        ])
        engine = GroundingEngine(StubRepository([widget]))

        resp = engine.query(["widget"])

        assert len(resp.result.gaps) == 0
        levels = {d.level for d in resp.result.primitives[0].depths}
        assert levels == {DepthLevel.EXISTENCE, DepthLevel.IDENTITY}


class _DanglingEdgeRepository(StubRepository):
    """Returns one edge with an endpoint absent from the node set (#94 C1)."""

    def __init__(self, dangle: str) -> None:
        self._a = _make_primitive("a", [_depth(DepthLevel.EXISTENCE), _depth(DepthLevel.IDENTITY)])
        super().__init__([self._a])
        self._dangle = dangle

    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        ghost = uuid4()
        src, tgt = (self._a.id, ghost) if self._dangle == "target" else (ghost, self._a.id)
        edge = EpistemicStep(
            source_id=src,
            target_id=tgt,
            relation_type=RelationType.REQUIRES,
            source_depth=DepthLevel.IDENTITY,
            target_depth=DepthLevel.IDENTITY,
        )
        return ResolvedSubgraph(roots=[self._a], nodes=[self._a], edges=[edge])


class TestBackendIntegrity:
    """A non-conformant backend's dangling edge fails loud, both sides (#94 C1)."""

    @pytest.mark.parametrize("dangle", ["target", "source"])
    def test_dangling_edge_endpoint_raises(self, dangle: str) -> None:
        engine = GroundingEngine(_DanglingEdgeRepository(dangle))
        with pytest.raises(GraphIntegrityError):
            engine.query(["a"])


class _UpstreamLinkRepository(StubRepository):
    """
    Two query roots (a, b) whose only link is an *upstream* node x with visible
    edges x->a and x->b. No visible path runs from either root to x, so x is off
    the justified frontier. The real backends never return such a node (their
    closure is forward-only, from roots along outgoing edges), so a hand-rolled
    repository is the only way to exercise the engine's frontier-restricted
    reachability check (#94).
    """

    def __init__(self) -> None:
        self._a = _make_primitive("a", [_depth(DepthLevel.EXISTENCE), _depth(DepthLevel.IDENTITY)])
        self._b = _make_primitive("b", [_depth(DepthLevel.EXISTENCE), _depth(DepthLevel.IDENTITY)])
        # x is grounded to D2, so its D1 edges are visible (D2 >= D1).
        self._x = _make_primitive("x", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY, [
                _relatum(self._a.id, RelationType.REQUIRES, DepthLevel.IDENTITY),
                _relatum(self._b.id, RelationType.REQUIRES, DepthLevel.IDENTITY),
            ]),
            _depth(DepthLevel.CAPABILITIES),
        ])
        super().__init__([self._a, self._b, self._x])

    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        edges = [
            EpistemicStep(
                source_id=self._x.id,
                target_id=target.id,
                relation_type=RelationType.REQUIRES,
                source_depth=DepthLevel.IDENTITY,
                target_depth=DepthLevel.IDENTITY,
            )
            for target in (self._a, self._b)
        ]
        return ResolvedSubgraph(
            roots=[self._a, self._b], nodes=[self._a, self._b, self._x], edges=edges,
        )


class TestFrontierReachability:
    """The reachability check runs over frontier-visible edges, not raw visible
    edges, so it stays consistent with the envelope actually returned (#94)."""

    def test_roots_linked_only_through_offfrontier_node_are_disconnected(self) -> None:
        """
        When two roots' only link is an upstream node pruned from the response,
        the roots are surfaced as isolated from each other. Reporting them as
        connected would cite a node the trace never returns.
        """
        engine = GroundingEngine(_UpstreamLinkRepository())

        resp = engine.query(["a", "b"])

        # The upstream linking node never enters the response...
        assert {p.name for p in resp.result.primitives} == {"a", "b"}
        # ...so the only gap is the reachability gap on one of the two roots,
        # and nothing names the pruned node.
        assert len(resp.result.gaps) == 1
        gap = resp.result.gaps[0]
        assert gap.kind == "REACHABILITY"
        assert gap.primitive.name in {"a", "b"}


# ---------------------------------------------------------------------------
# Tests: empty concepts and utility methods
# ---------------------------------------------------------------------------


class TestEmptyConcepts:
    def test_query_empty_concepts_returns_empty_response(self) -> None:
        """query([]) returns a valid but empty EpistemicResponse."""
        engine = GroundingEngine(StubRepository([]))
        resp = engine.query([])
        assert len(resp.result.primitives) == 0
        assert len(resp.result.gaps) == 0

    def test_list_primitive_names(self) -> None:
        """list_primitive_names returns all names in the repository."""
        p = _make_primitive("file", [_depth(DepthLevel.EXISTENCE)])
        engine = GroundingEngine(StubRepository([p]))
        names = engine.list_primitive_names()
        assert "file" in names


class TestReachabilityAnchorSelection:
    """Anchor selection uses the root with the largest reachable component."""

    def test_isolated_node_reported_not_majority(self) -> None:
        """
        With 3 roots where 2 are connected and 1 is isolated, the isolated
        one gets the ReachabilityGap regardless of submission order.
        """
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
        orphan_p = _make_primitive("orphan", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = GroundingEngine(StubRepository([file_p, permission_p, orphan_p]))

        # Submit orphan first to test that anchor selection picks the larger component
        resp = engine.query(["orphan", "file", "permission"])
        reachability_gaps = [g for g in resp.result.gaps if g.kind == "REACHABILITY"]
        assert len(reachability_gaps) == 1
        assert reachability_gaps[0].primitive.name == "orphan"


class TestMinDepthOnNonRootNodes:
    """min_depth only applies to root (submitted) nodes, not discovered ones."""

    def test_min_depth_skips_non_root_discovered_nodes(self) -> None:
        """
        Discovered node (via BFS) should not get a DepthGap from min_depth.
        """
        permission_p = _make_primitive("permission", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            # Only D0+D1, but it's not a root — min_depth should not apply
        ])
        file_p = _make_primitive("file", [
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS, [
                _relatum(permission_p.id, RelationType.CONSTRAINED_BY, DepthLevel.IDENTITY),
            ]),
        ])
        engine = GroundingEngine(StubRepository([file_p, permission_p]))

        # Only "file" is the root; "permission" is discovered via BFS
        resp = engine.query(["file"], min_depth=DepthLevel.CONSTRAINTS)

        depth_gaps = [g for g in resp.result.gaps if g.kind == "DEPTH"]
        # No DepthGap on permission — it's not a root
        for gap in depth_gaps:
            assert gap.primitive.name != "permission"
