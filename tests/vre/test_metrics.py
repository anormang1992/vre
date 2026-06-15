"""
Unit tests for aggregate usage metrics on Primitive nodes.

Uses a stub repository to avoid Neo4j dependency.
"""

from collections import deque
from datetime import datetime, timezone
from uuid import UUID

from vre import VRE
from vre.core.backends import Repository
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
from vre.core.grounding import GroundingResult


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fully_grounded(name: str) -> Primitive:
    return Primitive(name=name, depths=[
        Depth(level=DepthLevel.EXISTENCE),
        Depth(level=DepthLevel.IDENTITY, properties={"_": "identity"}),
        Depth(level=DepthLevel.CAPABILITIES, properties={"_": "capabilities"}),
        Depth(level=DepthLevel.CONSTRAINTS, properties={"_": "constraints"}),
    ])


def _make_vre(primitives: list[Primitive]) -> tuple[VRE, StubRepository]:
    repo = StubRepository(primitives)
    return VRE(repo, persist_traces=False), repo


# ---------------------------------------------------------------------------
# PrimitiveMetrics model tests
# ---------------------------------------------------------------------------

class TestPrimitiveMetricsModel:
    def test_defaults(self):
        m = PrimitiveMetrics()
        assert m.grounding_count == 0
        assert m.failure_count == 0
        assert m.last_grounded is None
        assert m.last_failed is None

    def test_last_exercised_both_none(self):
        m = PrimitiveMetrics()
        assert m.last_exercised is None

    def test_last_exercised_only_grounded(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        m = PrimitiveMetrics(last_grounded=t)
        assert m.last_exercised == t

    def test_last_exercised_only_failed(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        m = PrimitiveMetrics(last_failed=t)
        assert m.last_exercised == t

    def test_last_exercised_returns_max(self):
        early = datetime(2026, 1, 1, tzinfo=timezone.utc)
        late = datetime(2026, 6, 1, tzinfo=timezone.utc)
        m = PrimitiveMetrics(last_grounded=late, last_failed=early)
        assert m.last_exercised == late
        m = PrimitiveMetrics(last_grounded=early, last_failed=late)
        assert m.last_exercised == late

    def test_metrics_none_backward_compatible(self):
        p = Primitive(name="legacy")
        assert p.metrics is None


# ---------------------------------------------------------------------------
# Grounding metrics tests
# ---------------------------------------------------------------------------

class TestGroundingMetrics:
    def test_check_grounded_increments_grounding_count(self):
        file_p = _make_fully_grounded("file")
        vre, repo = _make_vre([file_p])
        vre.check(["file"])

        updated = repo.find_by_name("file")
        assert updated.metrics is not None
        assert updated.metrics.grounding_count == 1
        assert updated.metrics.failure_count == 0
        assert updated.metrics.last_grounded is not None
        assert updated.metrics.last_failed is None

    def test_check_ungrounded_increments_failure_count(self):
        """min_depth forces a DepthGap when the primitive lacks that depth."""
        file_p = Primitive(name="file", depths=[
            Depth(level=DepthLevel.EXISTENCE),
        ])
        vre, repo = _make_vre([file_p])
        vre.check(["file"], min_depth=DepthLevel.CONSTRAINTS)

        updated = repo.find_by_name("file")
        assert updated.metrics is not None
        assert updated.metrics.failure_count == 1
        assert updated.metrics.grounding_count == 0
        assert updated.metrics.last_failed is not None

    def test_metrics_accumulate_across_calls(self):
        file_p = _make_fully_grounded("file")
        vre, repo = _make_vre([file_p])
        vre.check(["file"])
        vre.check(["file"])
        vre.check(["file"])

        updated = repo.find_by_name("file")
        assert updated.metrics.grounding_count == 3

    def test_check_empty_concepts_no_crash(self):
        vre, repo = _make_vre([])
        result = vre.check([])
        # Empty concepts returns grounded=False with no trace — no crash
        assert isinstance(result, GroundingResult)

    def test_multiple_concepts_per_concept_metrics(self):
        file_p = _make_fully_grounded("file")
        create_p = Primitive(name="create", depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(level=DepthLevel.IDENTITY, properties={"_": "identity"}),
            Depth(level=DepthLevel.CAPABILITIES, relata=[
                Relatum(
                    relation_type=RelationType.APPLIES_TO,
                    target_id=file_p.id,
                    target_depth=DepthLevel.CAPABILITIES,
                ),
            ]),
            Depth(level=DepthLevel.CONSTRAINTS, properties={"_": "constraints"}),
        ])
        vre, repo = _make_vre([file_p, create_p])
        vre.check(["file", "create"])

        file_m = repo.find_by_name("file").metrics
        create_m = repo.find_by_name("create").metrics
        assert file_m.grounding_count == 1
        assert create_m.grounding_count == 1

    def test_timestamps_are_utc(self):
        file_p = _make_fully_grounded("file")
        vre, repo = _make_vre([file_p])
        vre.check(["file"])

        metrics = repo.find_by_name("file").metrics
        assert metrics.last_grounded.tzinfo is not None

    def test_nonexistent_concept_does_not_crash(self):
        vre, repo = _make_vre([])
        result = vre.check(["nonexistent"])
        assert result.grounded is False


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------

class TestMetricsSerialization:
    def test_roundtrip_via_model_dump(self):
        m = PrimitiveMetrics(
            last_grounded=datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
            last_failed=datetime(2026, 3, 10, 8, 0, 0, tzinfo=timezone.utc),
            grounding_count=42,
            failure_count=3,
        )
        data = m.model_dump(mode="json")
        restored = PrimitiveMetrics(**data)
        assert restored.grounding_count == 42
        assert restored.failure_count == 3
        assert restored.last_grounded == m.last_grounded
        assert restored.last_failed == m.last_failed

    def test_roundtrip_none_metrics_on_primitive(self):
        p = Primitive(name="test")
        data = p.model_dump(mode="json")
        restored = Primitive(**data)
        assert restored.metrics is None

    def test_roundtrip_with_metrics_on_primitive(self):
        m = PrimitiveMetrics(grounding_count=10)
        p = Primitive(name="test", metrics=m)
        data = p.model_dump(mode="json")
        restored = Primitive(**data)
        assert restored.metrics is not None
        assert restored.metrics.grounding_count == 10
