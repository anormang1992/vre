"""
Unit tests for the VRE auto-learning loop.

Tests cover template generation, engine persistence, and the iterative
learning loop.
"""

from collections import deque
from uuid import UUID, uuid4

import pytest

from vre.core.backends import Repository
from vre.core.errors import (
    CandidateValidationError,
    CyclicRelationshipError,
    GapResolvedError,
)
from vre.core.models import (
    Depth,
    DepthGap,
    DepthLevel,
    ExistenceGap,
    Primitive,
    PrimitiveMetrics,
    Provenance,
    ProvenanceSource,
    ReachabilityGap,
    ResolvedSubgraph,
    Relatum,
    RelationalGap,
    RelationType,
    TRANSITIVE_RELATION_TYPES,
)
from vre.learning.engine import LearningEngine, _make_provenance
from vre.learning.models import (
    DepthCandidate,
    ExistenceCandidate,
    ProposedDepth,
    ReachabilityCandidate,
    RelationalCandidate,
)
from vre.learning.templates import template_for_gap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prov(source: ProvenanceSource = ProvenanceSource.AUTHORED) -> Provenance:
    return Provenance(source=source)


def _primitive(
    name: str,
    depths: list[Depth] | None = None,
    id: UUID | None = None,
) -> Primitive:
    return Primitive(
        id=id or uuid4(),
        name=name,
        depths=depths or [],
        provenance=_prov(),
    )


def _depth(
    level: DepthLevel,
    properties: dict | None = None,
) -> Depth:
    return Depth(
        level=level,
        properties=properties or {},
        provenance=_prov(),
    )


class StubRepository(Repository):
    """
    In-memory repository for learning engine tests.
    """

    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self._by_id: dict[UUID, Primitive] = {}
        self._by_name: dict[str, Primitive] = {}
        for p in primitives or []:
            self._by_id[p.id] = p
            self._by_name[p.name.lower()] = p
        self.saved: list[Primitive] = []

    def find_by_id(self, id: UUID) -> Primitive | None:
        return self._by_id.get(id)

    def find_by_name(self, name: str) -> Primitive | None:
        return self._by_name.get(name.lower())

    def save_primitive(self, primitive: Primitive) -> None:
        for depth in primitive.depths:
            for relatum in depth.relata:
                if relatum.relation_type not in TRANSITIVE_RELATION_TYPES:
                    continue
                if relatum.target_id == primitive.id:
                    raise CyclicRelationshipError(
                        f"Self-referential {relatum.relation_type.value} "
                        f"on {primitive.name}"
                    )
                visited: set[UUID] = {relatum.target_id}
                queue: deque[UUID] = deque([relatum.target_id])
                while queue:
                    current = queue.popleft()
                    p = self._by_id.get(current)
                    if p is None:
                        continue
                    for d in p.depths:
                        for r in d.relata:
                            if r.relation_type not in TRANSITIVE_RELATION_TYPES:
                                continue
                            if r.target_id == primitive.id:
                                raise CyclicRelationshipError(
                                    f"{relatum.relation_type.value} from "
                                    f"{primitive.name} would create a cycle"
                                )
                            if r.target_id not in visited:
                                visited.add(r.target_id)
                                queue.append(r.target_id)
        self._by_id[primitive.id] = primitive
        self._by_name[primitive.name.lower()] = primitive
        self.saved.append(primitive)

    def list_names(self) -> list[str]:
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

    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestTemplateForGap:
    def test_existence_has_name(self):
        gap = ExistenceGap(primitive=_primitive("Copy"))
        template = template_for_gap(gap)
        assert isinstance(template, ExistenceCandidate)
        assert template.name == "Copy"
        assert template.d1 is None
        assert template.kind == "EXISTENCE"

    def test_depth_is_empty(self):
        prim = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE)
        template = template_for_gap(gap)
        assert isinstance(template, DepthCandidate)
        assert template.new_depths == []
        assert template.kind == "DEPTH"

    def test_relational_is_empty(self):
        gap = RelationalGap(
            source=_primitive("Create"),
            target=_primitive("File"),
            required_depth=DepthLevel.CAPABILITIES,
            current_depth=None,
        )
        template = template_for_gap(gap)
        assert isinstance(template, RelationalCandidate)
        assert template.new_depths == []
        assert template.kind == "RELATIONAL"

    def test_reachability_is_empty(self):
        prim = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        gap = ReachabilityGap(primitive=prim)
        template = template_for_gap(gap)
        assert isinstance(template, ReachabilityCandidate)
        assert template.source_name is None
        assert template.target_name is None
        assert template.relation_type is None
        assert template.source_depth_level is None
        assert template.target_depth_level is None
        assert template.kind == "REACHABILITY"


# ---------------------------------------------------------------------------
# Provenance derivation tests
# ---------------------------------------------------------------------------

class TestMakeProvenance:
    def test_learned_source(self):
        prov = _make_provenance(ProvenanceSource.LEARNED)
        assert prov.source == ProvenanceSource.LEARNED

    def test_authored_source(self):
        prov = _make_provenance(ProvenanceSource.AUTHORED)
        assert prov.source == ProvenanceSource.AUTHORED


# ---------------------------------------------------------------------------
# Engine persistence tests
# ---------------------------------------------------------------------------

class TestPersistExistence:
    def test_saves_primitive_with_d0_and_d1(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))

        filled = ExistenceCandidate(
            name="Copy",
            d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "Duplicates content"}),
        )

        engine.learn_gap(gap, filled)
        assert len(repo.saved) == 1
        saved = repo.saved[0]
        assert saved.name == "Copy"
        assert len(saved.depths) == 2
        levels = {d.level for d in saved.depths}
        assert levels == {DepthLevel.EXISTENCE, DepthLevel.IDENTITY}
        assert saved.provenance.source == ProvenanceSource.LEARNED

    def test_rejects_missing_d1(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))

        filled = ExistenceCandidate(name="Copy", d1=None)

        with pytest.raises(CandidateValidationError, match="missing D1"):
            engine.learn_gap(gap, filled)

    def test_rejects_renamed_candidate(self):
        # A candidate whose name diverges from the gap's primitive would create an
        # unrelated primitive while leaving the original gap unclosed.
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))

        filled = ExistenceCandidate(
            name="Duplicate",
            d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "x"}),
        )

        with pytest.raises(CandidateValidationError, match="must match the gapped concept"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_rejects_case_colliding_name(self):
        # A case-only rename collides per backend (SQLite NOCASE rejects, Neo4j
        # duplicates), so it must be rejected before persistence.
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))

        filled = ExistenceCandidate(
            name="copy",
            d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "x"}),
        )

        with pytest.raises(CandidateValidationError, match="must match the gapped concept"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_rejects_non_identity_d1(self):
        # D1 of an existence fill must be IDENTITY; any other level would leave a
        # hole above the auto-generated D0 (e.g. D0 + D2).
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))

        filled = ExistenceCandidate(
            name="Copy",
            d1=ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"description": "x"}),
        )

        with pytest.raises(CandidateValidationError, match="must be D1"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []


class TestPersistDepth:
    def test_merges_new_depth_into_existing(self):
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)

        gap = DepthGap(
            primitive=prim,
            required_depth=DepthLevel.CAPABILITIES,
            current_depth=DepthLevel.IDENTITY,
        )

        filled = DepthCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"can_read": "true"})],
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        levels = [d.level for d in saved.depths]
        assert DepthLevel.CAPABILITIES in levels
        assert levels == sorted(levels, key=int)

    def test_rejects_empty_new_depths(self):
        prim = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE)

        filled = DepthCandidate(new_depths=[])

        with pytest.raises(CandidateValidationError, match="no new depths"):
            engine.learn_gap(gap, filled)

    def test_rejects_depth_at_or_below_current(self):
        # Re-proposing an already-grounded level would overwrite authored
        # knowledge wholesale — the gap asked to extend, not to replace.
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"hijack": "true"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="already-grounded"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_rejects_depth_above_required(self):
        # A D2 planning gap must not be answered with D3 execution-level grounding.
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"y": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="escalate scope"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_rejects_non_contiguous_depths(self):
        # Filling {D2, D4} over a D1 chain leaves a hole at D3, so D4 would be
        # invisible to contiguity-strict grounding.
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.IMPLICATIONS, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
            ProposedDepth(level=DepthLevel.IMPLICATIONS, properties={"z": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="contiguous chain"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_none_current_depth_is_handled_and_anchored_at_d0(self):
        # A primitive with no contiguous chain reports current_depth=None; the
        # scope check must not choke on None (it would raise TypeError comparing
        # None < level) and contiguity must anchor the fill at D0.
        prim = _primitive("File", depths=[])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.IDENTITY, current_depth=None)

        hole = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"x": "true"}),
        ])
        with pytest.raises(CandidateValidationError, match="contiguous chain"):
            engine.learn_gap(gap, hole)
        assert repo.saved == []

    def test_accepts_contiguous_multi_level_fill(self):
        # D1 -> D3 fill of {D2, D3} extends the chain with no holes.
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"y": "true"}),
        ])

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        levels = {d.level for d in saved.depths}
        assert levels == {
            DepthLevel.EXISTENCE, DepthLevel.IDENTITY,
            DepthLevel.CAPABILITIES, DepthLevel.CONSTRAINTS,
        }


class TestPersistRelational:
    def test_merges_depth_into_target(self):
        source = _primitive("Create")
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)

        gap = RelationalGap(
            source=source,
            target=target,
            required_depth=DepthLevel.CAPABILITIES,
            current_depth=DepthLevel.EXISTENCE,
        )

        filled = RelationalCandidate(
            new_depths=[
                ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "a file"}),
                ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"writable": "true"}),
            ],
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        assert saved.id == target.id
        levels = {d.level for d in saved.depths}
        assert DepthLevel.CAPABILITIES in levels

    def test_rejects_depth_above_required(self):
        source = _primitive("Create")
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.IDENTITY,
        )

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"y": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="escalate scope"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_rejects_non_contiguous_depths(self):
        source = _primitive("Create")
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE,
        )

        # Skips D1 over a D0 chain -> hole.
        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="contiguous chain"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []


class TestPersistReachability:
    def test_attaches_relatum_to_source(self):
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)

        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        assert saved.id == source.id
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == target.id
        assert d2.relata[0].relation_type == RelationType.APPLIES_TO
        assert d2.relata[0].provenance.source == ProvenanceSource.LEARNED

    def test_attaches_relatum_with_reverse_direction(self):
        """Edge FROM a connected node TO the orphan (gap primitive is target)."""
        connected = _primitive("FileSystem", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        orphan = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([connected, orphan])
        engine = LearningEngine(repo)

        gap = ReachabilityGap(primitive=orphan)

        filled = ReachabilityCandidate(
            source_name="FileSystem",
            target_name="Delete",
            relation_type=RelationType.INCLUDES,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        assert saved.id == connected.id
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == orphan.id

    def test_rejects_when_neither_side_is_gap_primitive(self):
        """Neither source_name nor target_name matches the gap primitive."""
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Unrelated",
            target_name="AlsoUnrelated",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="must reference the gapped primitive"):
            engine.learn_gap(gap, filled)

    def test_rejects_missing_names(self):
        """Empty ReachabilityCandidate with no names at all."""
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate()

        with pytest.raises(CandidateValidationError, match="missing"):
            engine.learn_gap(gap, filled)

    def test_rejects_missing_source_name(self):
        """Has target_name but no source_name."""
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="missing"):
            engine.learn_gap(gap, filled)

    def test_rejects_when_source_depth_missing(self):
        """Source does not have the required depth level."""
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        source = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="DepthGap"):
            engine.learn_gap(gap, filled)

    def test_rejects_when_target_depth_missing(self):
        """Target does not have the required depth level."""
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="DepthGap"):
            engine.learn_gap(gap, filled)

    def test_places_edge_when_depths_already_present(self):
        """When both sides already have the required depths, edge placement succeeds."""
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY, {"description": "Removes content"}),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.EXISTENCE,
        )

        engine.learn_gap(gap, filled)
        saved = next(s for s in repo.saved if s.id == source.id)
        # D1 retains original properties
        d1 = next(d for d in saved.depths if d.level == DepthLevel.IDENTITY)
        assert d1.properties == {"description": "Removes content"}
        # Edge was placed
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == target.id

    def test_rejects_unresolvable_target_name(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="Nonexistent",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="Cannot resolve"):
            engine.learn_gap(gap, filled)

    def test_rejects_non_contiguous_source_depth(self):
        # Source {D0, D1, D3} *has* D3 but is only contiguously grounded to D1.
        # Placing the edge at D3 would make it invisible to grounding, so the
        # source-depth check must reject it despite exact-level membership.
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CONSTRAINTS,
            target_depth_level=DepthLevel.CONSTRAINTS,
        )

        with pytest.raises(CandidateValidationError, match="DepthGap"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_rejects_non_contiguous_target_depth(self):
        # Symmetric to the source case: the target's required depth must be
        # contiguously grounded for the edge to resolve.
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CONSTRAINTS,
            target_depth_level=DepthLevel.CONSTRAINTS,
        )

        with pytest.raises(CandidateValidationError, match="DepthGap"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []


# ---------------------------------------------------------------------------
# learn_gap tests
# ---------------------------------------------------------------------------

class TestLearnGap:
    def test_provenance_defaults_to_learned(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))

        filled = ExistenceCandidate(
            name="Copy",
            d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "Duplicates"}),
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        assert saved.provenance.source == ProvenanceSource.LEARNED

    def test_rejects_mismatched_gap_candidate_kind(self):
        # ExistenceGap fed a well-formed DepthCandidate: the kind guard must
        # reject the pair before any persistence is attempted.
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Widget"))
        filled = DepthCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "x"})],
        )

        with pytest.raises(CandidateValidationError, match="does not match gap kind"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_persist_rejects_unhandled_pair(self):
        # Direct _persist bypasses the learn_gap kind guard; the match's case _
        # backstop must still raise instead of silently persisting nothing.
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Widget"))
        filled = DepthCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "x"})],
        )

        with pytest.raises(CandidateValidationError, match="No persistence path"):
            engine._persist(gap, filled, _make_provenance(ProvenanceSource.LEARNED))
        assert repo.saved == []


# ---------------------------------------------------------------------------
# Reachability prerequisites tests
# ---------------------------------------------------------------------------

class TestReachabilityPrerequisites:
    def _make_engine(self, *primitives) -> LearningEngine:
        return LearningEngine(StubRepository(list(primitives)))

    def test_returns_empty_when_both_sides_have_required_depths(self):
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        engine = self._make_engine(source, target)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert prereqs == []

    def test_returns_depth_gap_when_source_missing_required_depth(self):
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        engine = self._make_engine(source, target)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert len(prereqs) == 1
        assert prereqs[0].primitive.id == source.id
        assert prereqs[0].required_depth == DepthLevel.CAPABILITIES
        assert prereqs[0].current_depth == DepthLevel.IDENTITY

    def test_returns_depth_gap_when_target_missing_required_depth(self):
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        engine = self._make_engine(source, target)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert len(prereqs) == 1
        assert prereqs[0].primitive.id == target.id
        assert prereqs[0].required_depth == DepthLevel.CAPABILITIES
        assert prereqs[0].current_depth == DepthLevel.IDENTITY

    def test_returns_both_depth_gaps_when_both_sides_missing(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        engine = self._make_engine(source, target)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert len(prereqs) == 2
        prim_ids = {p.primitive.id for p in prereqs}
        assert prim_ids == {source.id, target.id}

    def test_surfaces_depth_gap_when_required_level_present_but_non_contiguous(self):
        """{D0, D1, D3} contains D3 by exact membership, but its contiguous max is
        D1 — so requiring D3 must still surface a DepthGap (else the edge placed at
        D3 stays invisible to grounding and the ReachabilityGap never closes)."""
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        engine = self._make_engine(source, target)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CONSTRAINTS,
            target_depth_level=DepthLevel.CONSTRAINTS,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert len(prereqs) == 1
        assert prereqs[0].primitive.id == source.id
        assert prereqs[0].required_depth == DepthLevel.CONSTRAINTS
        assert prereqs[0].current_depth == DepthLevel.IDENTITY

    def test_current_depth_reflects_contiguous_max_not_highest_level(self):
        """A non-contiguous chain (D0, D1, D3) should report D1 as current."""
        source = _primitive("Create", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        engine = self._make_engine(source, target)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Create",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert len(prereqs) == 1
        assert prereqs[0].primitive.id == source.id
        assert prereqs[0].current_depth == DepthLevel.IDENTITY

    def test_works_with_reverse_direction(self):
        """When the gap primitive is the target, prerequisites still work."""
        connected = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        orphan = _primitive("Config", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        engine = self._make_engine(connected, orphan)
        gap = ReachabilityGap(primitive=orphan)
        candidate = ReachabilityCandidate(
            source_name="File",
            target_name="Config",
            relation_type=RelationType.INCLUDES,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.IDENTITY,
        )

        prereqs = engine.reachability_prerequisites(gap, candidate)
        assert len(prereqs) == 1
        assert prereqs[0].primitive.id == connected.id
        assert prereqs[0].required_depth == DepthLevel.CAPABILITIES

    def test_validates_candidate_first(self):
        """Missing source_name raises before any repo lookup."""
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        engine = self._make_engine(source)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="missing"):
            engine.reachability_prerequisites(gap, candidate)

    def test_validates_gap_primitive_match_first(self):
        """Edge that doesn't reference the gap primitive raises before repo lookups."""
        orphan = _primitive("Config", depths=[_depth(DepthLevel.CAPABILITIES)])
        a = _primitive("File", depths=[_depth(DepthLevel.CAPABILITIES)])
        b = _primitive("Write", depths=[_depth(DepthLevel.CAPABILITIES)])
        engine = self._make_engine(orphan, a, b)
        gap = ReachabilityGap(primitive=orphan)
        candidate = ReachabilityCandidate(
            source_name="File",
            target_name="Write",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="must reference the gapped primitive"):
            engine.reachability_prerequisites(gap, candidate)

    def test_unresolvable_name_raises(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        engine = self._make_engine(source)
        gap = ReachabilityGap(primitive=source)
        candidate = ReachabilityCandidate(
            source_name="Delete",
            target_name="Nonexistent",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="Cannot resolve"):
            engine.reachability_prerequisites(gap, candidate)


# ---------------------------------------------------------------------------
# Template edge cases
# ---------------------------------------------------------------------------

class TestTemplateForGapEdgeCases:
    def test_unknown_gap_type_raises(self):
        class UnknownGap:
            pass

        with pytest.raises(ValueError, match="Unknown gap type"):
            template_for_gap(UnknownGap())


# ---------------------------------------------------------------------------
# Engine persistence edge cases
# ---------------------------------------------------------------------------

class TestPersistDepthEdgeCases:
    def test_primitive_not_found_raises(self):
        prim = _primitive("Ghost")
        repo = StubRepository()  # empty -- primitive not in repo
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"a": "true"})],
        )

        with pytest.raises(CandidateValidationError, match="not found"):
            engine.learn_gap(gap, filled)

    def test_replaces_existing_depth_level(self):
        """Filling the hole below a detached level replaces that level in place.

        {D0, D1, D3} has contiguous max D1. Filling D2 makes D3 reachable; the
        candidate re-states D3, so its properties are replaced (not duplicated).
        """
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CONSTRAINTS, {"old": True}),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"cap": "true"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"updated": "true"}),
        ])

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        d3 = next(d for d in saved.depths if d.level == DepthLevel.CONSTRAINTS)
        assert d3.properties == {"updated": "true"}

    def test_preserves_relata_and_stamps_provenance_on_replaced_depth(self):
        """Replacing a depth carries forward its relata and stamps None provenance.

        {D0, D1, D3} has contiguous max D1. Filling D2 makes D3 reachable; the
        candidate re-states D3 (a replacement), which must carry forward D3's
        relata and stamp their missing provenance. The untouched D0 relatum
        stays unstamped.
        """
        target_id = uuid4()
        prim = _primitive("File", depths=[
            Depth(
                level=DepthLevel.EXISTENCE,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.APPLIES_TO,
                    target_id=target_id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=None,
                )],
            ),
            _depth(DepthLevel.IDENTITY),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={"old": "val"},
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=target_id,
                    target_depth=DepthLevel.IDENTITY,
                    provenance=None,
                )],
            ),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.IDENTITY)

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"cap": "read"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"desc": "constrained"}),
        ])

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        # D0 was NOT touched -- its relatum should remain unstamped
        d0 = next(d for d in saved.depths if d.level == DepthLevel.EXISTENCE)
        assert d0.relata[0].provenance is None
        # D3 WAS replaced -- relata carried forward and provenance stamped
        d3 = next(d for d in saved.depths if d.level == DepthLevel.CONSTRAINTS)
        assert len(d3.relata) == 1
        assert d3.relata[0].target_id == target_id
        assert d3.relata[0].provenance is not None
        assert d3.relata[0].provenance.source == ProvenanceSource.LEARNED
        # D3 properties updated
        assert d3.properties == {"desc": "constrained"}


class TestPersistRelationalEdgeCases:
    def test_empty_new_depths_raises(self):
        source = _primitive("Create")
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE,
        )

        filled = RelationalCandidate(new_depths=[])

        with pytest.raises(CandidateValidationError, match="no new depths"):
            engine.learn_gap(gap, filled)

    def test_target_not_found_raises(self):
        source = _primitive("Create")
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([source])  # target not in repo
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.IDENTITY,
        )

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"writable": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="not found"):
            engine.learn_gap(gap, filled)

    def test_replaces_existing_depth_on_target(self):
        # {D0, D2} has contiguous max D0. Filling D1 makes D2 reachable; the
        # candidate re-states D2, so its properties are replaced in place.
        source = _primitive("Create")
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.CAPABILITIES, {"old": True}),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE,
        )

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "a file"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"updated": "true"}),
        ])

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert d2.properties == {"updated": "true"}

    def test_does_not_stamp_provenance_on_untouched_depths(self):
        """Relata on depths not being replaced should remain untouched."""
        source = _primitive("Create")
        target = _primitive("File", depths=[
            Depth(
                level=DepthLevel.EXISTENCE,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.APPLIES_TO,
                    target_id=uuid4(),
                    target_depth=DepthLevel.IDENTITY,
                    provenance=None,
                )],
            ),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE,
        )

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "a file"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"cap": "true"}),
        ])

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        # D0 was not touched -- its relatum provenance should remain None
        d0 = next(d for d in saved.depths if d.level == DepthLevel.EXISTENCE)
        assert d0.relata[0].provenance is None

    def test_preserves_relata_on_replaced_target_depth(self):
        """Replacing a depth on the target carries forward its relata.

        {D0, D2} has contiguous max D0. Filling D1 makes D2 reachable; the
        candidate re-states D2, so D2's relata carry forward and get stamped.
        """
        source = _primitive("Create")
        rel_target_id = uuid4()
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={"old": "val"},
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=rel_target_id,
                    target_depth=DepthLevel.IDENTITY,
                    provenance=None,
                )],
            ),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE,
        )

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "a file"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"updated": "true"}),
        ])

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert d2.properties == {"updated": "true"}
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == rel_target_id
        assert d2.relata[0].provenance is not None
        assert d2.relata[0].provenance.source == ProvenanceSource.LEARNED


class TestGateValidatesAgainstLiveState:
    """The persist gate must validate against the live primitive, not the gap
    snapshot. A gap is a value object the caller holds; its current_depth can be
    stale by the time learn_gap runs. Enforcing against the snapshot would let
    overwrites and holes through exactly when the graph has moved underneath."""

    def test_resolved_gap_raises_gap_resolved_error(self):
        # Snapshot captured File at {D0, D1}; by persist time the repo's File is
        # already grounded to D3. The gap is closed — report it, don't overwrite
        # grounded knowledge and don't mislabel it as a bad candidate.
        live = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
            _depth(DepthLevel.CONSTRAINTS),
        ])
        repo = StubRepository([live])
        engine = LearningEngine(repo)
        stale = _primitive(
            "File",
            depths=[_depth(DepthLevel.EXISTENCE), _depth(DepthLevel.IDENTITY)],
            id=live.id,
        )
        gap = DepthGap(primitive=stale, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.IDENTITY)
        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"x": "true"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"y": "true"}),
        ])

        with pytest.raises(GapResolvedError):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_resolved_relational_gap_raises_gap_resolved_error(self):
        # Same divergence on the relational path: the target is already grounded
        # past what the edge needs.
        source = _primitive("Create")
        live = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([source, live])
        engine = LearningEngine(repo)
        stale = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)], id=live.id)
        gap = RelationalGap(
            source=source, target=stale,
            required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE,
        )
        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"x": "true"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"y": "true"}),
        ])

        with pytest.raises(GapResolvedError):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_stale_snapshot_does_not_overwrite_grounded_authored_depth(self):
        # Snapshot said current=D1; live File already carries authored D2. A fill
        # built against the stale snapshot must not clobber D2.
        live = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES, {"authored": "keep"}),
        ])
        repo = StubRepository([live])
        engine = LearningEngine(repo)
        stale = _primitive(
            "File",
            depths=[_depth(DepthLevel.EXISTENCE), _depth(DepthLevel.IDENTITY)],
            id=live.id,
        )
        gap = DepthGap(primitive=stale, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.IDENTITY)
        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"hijack": "true"}),
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"y": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="already-grounded"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []

    def test_stale_snapshot_does_not_reintroduce_hole(self):
        # Snapshot claimed contiguity to D2; live File regressed to {D0, D1}.
        # A fill of just D3 (valid against the stale snapshot) would punch a hole
        # at D2 in the live primitive.
        live = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
        ])
        repo = StubRepository([live])
        engine = LearningEngine(repo)
        stale = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ], id=live.id)
        gap = DepthGap(primitive=stale, required_depth=DepthLevel.CONSTRAINTS, current_depth=DepthLevel.CAPABILITIES)
        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"y": "true"}),
        ])

        with pytest.raises(CandidateValidationError, match="contiguous chain"):
            engine.learn_gap(gap, filled)
        assert repo.saved == []


class TestPersistReachabilityEdgeCases:
    def test_missing_depth_levels_raises(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=None,
            target_depth_level=None,
        )

        with pytest.raises(CandidateValidationError, match="missing"):
            engine.learn_gap(gap, filled)

    def test_source_not_found_raises(self):
        source = _primitive("Ghost")
        target = _primitive("File", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([target])  # source not in repo
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Ghost",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="Cannot resolve"):
            engine.learn_gap(gap, filled)

    def test_target_not_found_raises(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])  # target not in repo
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)

        filled = ReachabilityCandidate(
            source_name="Delete",
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CandidateValidationError, match="Cannot resolve"):
            engine.learn_gap(gap, filled)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    """Cycle detection via save_primitive -> CyclicRelationshipError propagated."""

    def test_self_referential_transitive_raises(self):
        """A->A via REQUIRES is a trivial cycle -> CyclicRelationshipError."""
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([a])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=a)

        filled = ReachabilityCandidate(
            source_name="A",
            target_name="A",
            relation_type=RelationType.REQUIRES,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CyclicRelationshipError):
            engine.learn_gap(gap, filled)

    def test_direct_cycle_raises(self):
        """A->B via REQUIRES exists; B->A via REQUIRES would cycle -> CyclicRelationshipError."""
        b = _primitive("B", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=b.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
                provenance=_prov(),
            ),
        ])
        repo = StubRepository([a, b])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=b)

        filled = ReachabilityCandidate(
            source_name="B",
            target_name="A",
            relation_type=RelationType.REQUIRES,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CyclicRelationshipError):
            engine.learn_gap(gap, filled)

    def test_indirect_cycle_raises(self):
        """A->B->C via DEPENDS_ON exists; C->A would cycle -> CyclicRelationshipError."""
        c = _primitive("C", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        b = _primitive("B", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.DEPENDS_ON,
                    target_id=c.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
                provenance=_prov(),
            ),
        ])
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.DEPENDS_ON,
                    target_id=b.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
                provenance=_prov(),
            ),
        ])
        repo = StubRepository([a, b, c])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=c)

        filled = ReachabilityCandidate(
            source_name="C",
            target_name="A",
            relation_type=RelationType.DEPENDS_ON,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CyclicRelationshipError):
            engine.learn_gap(gap, filled)

    def test_mixed_transitive_types_cycle_raises(self):
        """A->B via REQUIRES exists; B->A via CONSTRAINED_BY would cycle -> CyclicRelationshipError."""
        b = _primitive("B", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=b.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
                provenance=_prov(),
            ),
        ])
        repo = StubRepository([a, b])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=b)

        filled = ReachabilityCandidate(
            source_name="B",
            target_name="A",
            relation_type=RelationType.CONSTRAINED_BY,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        with pytest.raises(CyclicRelationshipError):
            engine.learn_gap(gap, filled)

    def test_non_transitive_cycle_allowed(self):
        """A->B via APPLIES_TO exists; B->A via APPLIES_TO is fine."""
        b = _primitive("B", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                relata=[Relatum(
                    relation_type=RelationType.APPLIES_TO,
                    target_id=b.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
                provenance=_prov(),
            ),
        ])
        repo = StubRepository([a, b])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=b)

        filled = ReachabilityCandidate(
            source_name="B",
            target_name="A",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1

    def test_non_transitive_self_ref_allowed(self):
        """A->A via INCLUDES is fine."""
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([a])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=a)

        filled = ReachabilityCandidate(
            source_name="A",
            target_name="A",
            relation_type=RelationType.INCLUDES,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        engine.learn_gap(gap, filled)
        assert len(repo.saved) == 1

    def test_valid_transitive_edge_accepted(self):
        """A->B via REQUIRES with no path B->A."""
        b = _primitive("B", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _primitive("A", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([a, b])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=a)

        filled = ReachabilityCandidate(
            source_name="A",
            target_name="B",
            relation_type=RelationType.REQUIRES,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        engine.learn_gap(gap, filled)
        saved = repo.saved[0]
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == b.id
        assert d2.relata[0].relation_type == RelationType.REQUIRES


class TestStubRepositoryCycleDetection:
    """Defense-in-depth: save_primitive raises on transitive cycles."""

    def test_save_raises_on_self_referential_transitive(self):
        a = _primitive("A", depths=[
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                provenance=_prov(),
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=uuid4(),  # placeholder, overwritten below
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
            ),
        ])
        a.depths[0].relata[0].target_id = a.id
        repo = StubRepository()
        with pytest.raises(CyclicRelationshipError):
            repo.save_primitive(a)

    def test_save_raises_on_two_node_cycle(self):
        b = _primitive("B", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.CAPABILITIES),
        ])
        a = _primitive("A", depths=[
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                provenance=_prov(),
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=b.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
            ),
        ])
        repo = StubRepository([a])  # A->B already in repo
        # Now try to save B with B->A
        b_with_edge = _primitive("B", depths=[
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={},
                provenance=_prov(),
                relata=[Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=a.id,
                    target_depth=DepthLevel.CAPABILITIES,
                    provenance=_prov(),
                )],
            ),
        ], id=b.id)
        with pytest.raises(CyclicRelationshipError):
            repo.save_primitive(b_with_edge)
