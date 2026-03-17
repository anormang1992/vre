"""
Unit tests for the VRE auto-learning loop.

Tests cover template generation, engine persistence, and the iterative
learning loop.
"""

from uuid import UUID, uuid4

import pytest

from vre.core.grounding.models import GroundingResult
from vre.core.models import (
    Depth,
    DepthGap,
    DepthLevel,
    EpistemicQuery,
    EpistemicResponse,
    EpistemicResult,
    ExistenceGap,
    Primitive,
    Provenance,
    ProvenanceSource,
    ReachabilityGap,
    Relatum,
    RelationalGap,
    RelationType,
)
from vre.learning.callback import LearningCallback
from vre.learning.engine import LearningEngine, _make_provenance
from vre.learning.models import (
    CandidateDecision,
    DepthCandidate,
    ExistenceCandidate,
    ProposedDepth,
    ReachabilityCandidate,
    RelationalCandidate,
)
from vre.learning.templates import TemplateFactory


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


def _grounding_result(
    grounded: bool,
    gaps: list | None = None,
    primitives: list[Primitive] | None = None,
) -> GroundingResult:
    prims = primitives or []
    trace = EpistemicResponse(
        query=EpistemicQuery(concept_ids=[]),
        result=EpistemicResult(primitives=prims, gaps=[], pathway=[]),
    ) if prims else None
    return GroundingResult(
        grounded=grounded,
        resolved=[],
        gaps=gaps or [],
        trace=trace,
    )


class StubRepository:
    """
    In-memory repository for learning engine tests.
    """

    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self._by_id: dict[UUID, Primitive] = {}
        for p in primitives or []:
            self._by_id[p.id] = p
        self.saved: list[Primitive] = []

    def find_by_id(self, id: UUID) -> Primitive | None:
        return self._by_id.get(id)

    def save_primitive(self, primitive: Primitive) -> None:
        self._by_id[primitive.id] = primitive
        self.saved.append(primitive)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestTemplateFactory:
    def test_existence_has_name(self):
        gap = ExistenceGap(primitive=_primitive("Copy"))
        template = TemplateFactory.from_gap(gap)
        assert isinstance(template, ExistenceCandidate)
        assert template.name == "Copy"
        assert template.d1 is None
        assert template.kind == "EXISTENCE"

    def test_depth_is_empty(self):
        prim = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE)
        template = TemplateFactory.from_gap(gap)
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
        template = TemplateFactory.from_gap(gap)
        assert isinstance(template, RelationalCandidate)
        assert template.new_depths == []
        assert template.kind == "RELATIONAL"

    def test_reachability_is_empty(self):
        prim = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        gap = ReachabilityGap(primitive=prim)
        template = TemplateFactory.from_gap(gap)
        assert isinstance(template, ReachabilityCandidate)
        assert template.target_name is None
        assert template.relation_type is None
        assert template.source_depth_level is None
        assert template.target_depth_level is None
        assert template.kind == "REACHABILITY"


# ---------------------------------------------------------------------------
# Provenance derivation tests
# ---------------------------------------------------------------------------

class TestMakeProvenance:
    def test_accepted_is_learned(self):
        prov = _make_provenance(CandidateDecision.ACCEPTED)
        assert prov.source == ProvenanceSource.LEARNED
        assert "accepted" in prov.detail

    def test_modified_is_conversational(self):
        prov = _make_provenance(CandidateDecision.MODIFIED)
        assert prov.source == ProvenanceSource.CONVERSATIONAL
        assert "modified" in prov.detail


# ---------------------------------------------------------------------------
# Engine persistence tests
# ---------------------------------------------------------------------------

class TestPersistExistence:
    def test_saves_primitive_with_d0_and_d1(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = ExistenceCandidate(
            name="Copy",
            d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "Duplicates content"}),
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.ACCEPTED
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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = ExistenceCandidate(name="Copy", d1=None)

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="missing D1"):
            engine.learn_at(grounding, 0, callback)

    def test_modified_gets_conversational_provenance(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = ExistenceCandidate(
            name="Copy",
            d1=ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "Duplicates"}),
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.MODIFIED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.MODIFIED
        saved = repo.saved[0]
        assert saved.provenance.source == ProvenanceSource.CONVERSATIONAL


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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = DepthCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"can_read": "true"})],
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        levels = [d.level for d in saved.depths]
        assert DepthLevel.CAPABILITIES in levels
        assert levels == sorted(levels, key=int)

    def test_rejects_empty_new_depths(self):
        prim = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE)
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = DepthCandidate(new_depths=[])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="no new depths"):
            engine.learn_at(grounding, 0, callback)


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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = RelationalCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"writable": "true"})],
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        assert saved.id == target.id
        levels = {d.level for d in saved.depths}
        assert DepthLevel.CAPABILITIES in levels


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
        grounding = _grounding_result(
            grounded=False, gaps=[gap], primitives=[source, target],
        )

        filled = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        assert saved.id == source.id
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == target.id
        assert d2.relata[0].relation_type == RelationType.APPLIES_TO
        assert d2.relata[0].provenance.source == ProvenanceSource.LEARNED

    def test_rejects_missing_target_name(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source])

        filled = ReachabilityCandidate()

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="missing"):
            engine.learn_at(grounding, 0, callback)

    def test_learns_missing_source_depth_before_placing_edge(self):
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        source = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        edge_candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CONSTRAINTS,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        call_count = 0

        def callback(candidate, gr, gap):
            nonlocal call_count
            call_count += 1
            if isinstance(candidate, DepthCandidate):
                # Agent fills in the missing depths
                filled = DepthCandidate(new_depths=[
                    ProposedDepth(level=DepthLevel.IDENTITY, properties={"description": "Remove"}),
                    ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"can_delete": "true"}),
                    ProposedDepth(level=DepthLevel.CONSTRAINTS, properties={"requires_perm": "true"}),
                ])
                return filled, CandidateDecision.ACCEPTED
            return edge_candidate, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        # Callback invoked twice: once for edge, once for source depths
        assert call_count == 2
        # Source should have depths + relatum
        saved_source = next(s for s in repo.saved if s.id == source.id and any(
            r for d in s.depths for r in d.relata
        ))
        d3 = next(d for d in saved_source.depths if d.level == DepthLevel.CONSTRAINTS)
        assert d3.properties == {"requires_perm": "true"}
        assert len(d3.relata) == 1
        assert d3.relata[0].target_id == target.id

    def test_abandons_edge_if_depth_learning_rejected(self):
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        source = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        edge_candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CONSTRAINTS,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            if isinstance(candidate, DepthCandidate):
                return None, CandidateDecision.REJECTED
            return edge_candidate, CandidateDecision.ACCEPTED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.REJECTED
        # No relata should have been placed
        for saved in repo.saved:
            for d in saved.depths:
                assert len(d.relata) == 0

    def test_skips_depth_learning_when_already_present(self):
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY, {"description": "Removes content"}),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        edge_candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.EXISTENCE,
        )

        call_count = 0

        def callback(candidate, gr, gap):
            nonlocal call_count
            call_count += 1
            return edge_candidate, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        # Only called once — no depth learning needed
        assert call_count == 1
        # D1 retains original properties
        saved = next(s for s in repo.saved if s.id == source.id)
        d1 = next(d for d in saved.depths if d.level == DepthLevel.IDENTITY)
        assert d1.properties == {"description": "Removes content"}

    def test_rejects_unresolvable_target_name(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source])

        filled = ReachabilityCandidate(
            target_name="Nonexistent",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="Cannot resolve"):
            engine.learn_at(grounding, 0, callback)


# ---------------------------------------------------------------------------
# Decision flow tests
# ---------------------------------------------------------------------------

class TestDecisionFlow:
    def test_rejected_does_not_persist(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))
        grounding = _grounding_result(grounded=False, gaps=[gap])

        def callback(candidate, gr, gap):
            return None, CandidateDecision.REJECTED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.REJECTED
        assert len(repo.saved) == 0

    def test_skipped_does_not_persist(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))
        grounding = _grounding_result(grounded=False, gaps=[gap])

        def callback(candidate, gr, gap):
            return None, CandidateDecision.SKIPPED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.SKIPPED
        assert len(repo.saved) == 0

    def test_no_gaps_raises(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        grounding = _grounding_result(grounded=True, gaps=[])

        with pytest.raises(ValueError, match="No gaps"):
            engine.learn_at(grounding, 0, lambda c, g, gap: (None, CandidateDecision.REJECTED))


# ---------------------------------------------------------------------------
# learn_at tests
# ---------------------------------------------------------------------------

class TestLearnAt:
    def test_processes_gap_at_index(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap0 = ExistenceGap(primitive=_primitive("Copy"))
        gap1 = ExistenceGap(primitive=_primitive("Move"))
        grounding = _grounding_result(grounded=False, gaps=[gap0, gap1])

        received_names = []

        def callback(candidate, gr, gap):
            received_names.append(candidate.name)
            return None, CandidateDecision.SKIPPED

        engine.learn_at(grounding, 1, callback)
        assert received_names == ["Move"]

    def test_out_of_range_raises(self):
        repo = StubRepository()
        engine = LearningEngine(repo)
        gap = ExistenceGap(primitive=_primitive("Copy"))
        grounding = _grounding_result(grounded=False, gaps=[gap])

        with pytest.raises(ValueError, match="out of range"):
            engine.learn_at(grounding, 5, lambda c, g, gap: (None, CandidateDecision.REJECTED))


# ---------------------------------------------------------------------------
# Callback lifecycle tests
# ---------------------------------------------------------------------------

class TestLearningCallbackLifecycle:
    def test_default_enter_returns_self(self):
        class SimpleLearner(LearningCallback):
            def __call__(self, candidate, grounding, gap):
                return None, CandidateDecision.REJECTED

        cb = SimpleLearner()
        assert cb.__enter__() is cb

    def test_default_exit_is_noop(self):
        class SimpleLearner(LearningCallback):
            def __call__(self, candidate, grounding, gap):
                return None, CandidateDecision.REJECTED

        cb = SimpleLearner()
        cb.__exit__(None, None, None)  # should not raise

    def test_usable_as_context_manager(self):
        class SimpleLearner(LearningCallback):
            def __call__(self, candidate, grounding, gap):
                return None, CandidateDecision.REJECTED

        cb = SimpleLearner()
        with cb as entered:
            assert entered is cb


# ---------------------------------------------------------------------------
# Template edge cases
# ---------------------------------------------------------------------------

class TestTemplateFactoryEdgeCases:
    def test_unknown_gap_type_raises(self):
        class UnknownGap:
            pass

        with pytest.raises(ValueError, match="Unknown gap type"):
            TemplateFactory.from_gap(UnknownGap())


# ---------------------------------------------------------------------------
# Engine persistence edge cases
# ---------------------------------------------------------------------------

class TestPersistDepthEdgeCases:
    def test_primitive_not_found_raises(self):
        prim = _primitive("Ghost")
        repo = StubRepository()  # empty — primitive not in repo
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.EXISTENCE)
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = DepthCandidate(
            new_depths=[ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"a": "true"})],
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="not found"):
            engine.learn_at(grounding, 0, callback)

    def test_replaces_existing_depth_level(self):
        """When a candidate proposes a depth that already exists, it should replace it."""
        prim = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY, {"old": True}),
        ])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.IDENTITY)
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"updated": "true"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"cap": "true"}),
        ])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        d1 = next(d for d in saved.depths if d.level == DepthLevel.IDENTITY)
        assert d1.properties == {"updated": "true"}

    def test_preserves_relata_and_stamps_provenance_on_replaced_depth(self):
        """Replacing a depth carries forward its relata and stamps None provenance."""
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
            Depth(
                level=DepthLevel.IDENTITY,
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
        gap = DepthGap(primitive=prim, required_depth=DepthLevel.CAPABILITIES, current_depth=DepthLevel.IDENTITY)
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = DepthCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.IDENTITY, properties={"desc": "a file"}),
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"cap": "read"}),
        ])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        # D0 was NOT touched — its relatum should remain unstamped
        d0 = next(d for d in saved.depths if d.level == DepthLevel.EXISTENCE)
        assert d0.relata[0].provenance is None
        # D1 WAS replaced — relata carried forward and provenance stamped
        d1 = next(d for d in saved.depths if d.level == DepthLevel.IDENTITY)
        assert len(d1.relata) == 1
        assert d1.relata[0].target_id == target_id
        assert d1.relata[0].provenance is not None
        assert d1.relata[0].provenance.source == ProvenanceSource.LEARNED
        # D1 properties updated
        assert d1.properties == {"desc": "a file"}


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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = RelationalCandidate(new_depths=[])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="no new depths"):
            engine.learn_at(grounding, 0, callback)

    def test_target_not_found_raises(self):
        source = _primitive("Create")
        target = _primitive("File")
        repo = StubRepository([source])  # target not in repo
        engine = LearningEngine(repo)
        gap = RelationalGap(
            source=source, target=target,
            required_depth=DepthLevel.CAPABILITIES, current_depth=None,
        )
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"writable": "true"}),
        ])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="not found"):
            engine.learn_at(grounding, 0, callback)

    def test_replaces_existing_depth_on_target(self):
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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"updated": "true"}),
        ])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"cap": "true"}),
        ])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        # D0 was not touched — its relatum provenance should remain None
        d0 = next(d for d in saved.depths if d.level == DepthLevel.EXISTENCE)
        assert d0.relata[0].provenance is None

    def test_preserves_relata_on_replaced_target_depth(self):
        """Replacing a depth on the target carries forward its relata."""
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
        grounding = _grounding_result(grounded=False, gaps=[gap])

        filled = RelationalCandidate(new_depths=[
            ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"updated": "true"}),
        ])

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        engine.learn_at(grounding, 0, callback)
        saved = repo.saved[0]
        d2 = next(d for d in saved.depths if d.level == DepthLevel.CAPABILITIES)
        assert d2.properties == {"updated": "true"}
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == rel_target_id
        assert d2.relata[0].provenance is not None
        assert d2.relata[0].provenance.source == ProvenanceSource.LEARNED


class TestPersistReachabilityEdgeCases:
    def test_missing_depth_levels_raises(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source])

        filled = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=None,
            target_depth_level=None,
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="missing"):
            engine.learn_at(grounding, 0, callback)

    def test_source_not_found_raises(self):
        source = _primitive("Ghost")
        target = _primitive("File", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([target])  # source not in repo
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        filled = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="not found"):
            engine.learn_at(grounding, 0, callback)

    def test_target_not_found_raises(self):
        source = _primitive("Delete", depths=[_depth(DepthLevel.CAPABILITIES)])
        target = _primitive("File", depths=[_depth(DepthLevel.CAPABILITIES)])
        repo = StubRepository([source])  # target not in repo
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        filled = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            return filled, CandidateDecision.ACCEPTED

        with pytest.raises(ValueError, match="not found"):
            engine.learn_at(grounding, 0, callback)

    def test_learns_missing_target_depth_and_refreshes(self):
        """When only the target needs depth learning, verifies the target is refreshed."""
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        edge_candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            if isinstance(candidate, DepthCandidate):
                filled = DepthCandidate(new_depths=[
                    ProposedDepth(level=DepthLevel.IDENTITY, properties={"desc": "a file"}),
                    ProposedDepth(level=DepthLevel.CAPABILITIES, properties={"readable": "true"}),
                ])
                return filled, CandidateDecision.ACCEPTED
            return edge_candidate, CandidateDecision.ACCEPTED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.ACCEPTED
        # Edge should be placed on source
        saved_source = next(s for s in repo.saved if s.id == source.id and any(
            r for d in s.depths for r in d.relata
        ))
        d2 = next(d for d in saved_source.depths if d.level == DepthLevel.CAPABILITIES)
        assert len(d2.relata) == 1
        assert d2.relata[0].target_id == target.id

    def test_rejects_target_depth_learning_abandons_edge(self):
        """When target depth learning is rejected, edge placement is abandoned."""
        source = _primitive("Delete", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.IDENTITY),
            _depth(DepthLevel.CAPABILITIES),
        ])
        target = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        edge_candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            if isinstance(candidate, DepthCandidate):
                return None, CandidateDecision.REJECTED
            return edge_candidate, CandidateDecision.ACCEPTED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.REJECTED

    def test_skips_source_depth_learning_abandons_edge(self):
        """When source depth learning is skipped, edge placement is abandoned."""
        source = _primitive("Delete", depths=[_depth(DepthLevel.EXISTENCE)])
        target = _primitive("File", depths=[
            _depth(DepthLevel.EXISTENCE),
            _depth(DepthLevel.CAPABILITIES),
        ])
        repo = StubRepository([source, target])
        engine = LearningEngine(repo)
        gap = ReachabilityGap(primitive=source)
        grounding = _grounding_result(grounded=False, gaps=[gap], primitives=[source, target])

        edge_candidate = ReachabilityCandidate(
            target_name="File",
            relation_type=RelationType.APPLIES_TO,
            source_depth_level=DepthLevel.CAPABILITIES,
            target_depth_level=DepthLevel.CAPABILITIES,
        )

        def callback(candidate, gr, gap):
            if isinstance(candidate, DepthCandidate):
                return None, CandidateDecision.SKIPPED
            return edge_candidate, CandidateDecision.ACCEPTED

        result = engine.learn_at(grounding, 0, callback)
        assert result.decision == CandidateDecision.SKIPPED


class TestLearnMissingDepthsEdgeCases:
    def test_filled_none_treated_as_rejected(self):
        """If callback returns filled=None with ACCEPTED, treat as REJECTED."""
        prim = _primitive("File", depths=[_depth(DepthLevel.EXISTENCE)])
        repo = StubRepository([prim])
        engine = LearningEngine(repo)

        # Call _learn_missing_depths directly
        grounding = _grounding_result(grounded=False)

        def callback(candidate, gr, gap):
            return None, CandidateDecision.ACCEPTED

        result = engine._learn_missing_depths(
            prim, DepthLevel.CAPABILITIES, grounding, callback,
        )
        assert result == CandidateDecision.REJECTED
