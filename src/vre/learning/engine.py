# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
LearningEngine — orchestrates the auto-learning loop.

Processes one gap at a time: template creation -> callback invocation ->
validation -> persistence -> re-grounding.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from vre.core.graph import PrimitiveRepository
from vre.core.grounding.models import GroundingResult
from vre.core.errors import CandidateValidationError, CyclicRelationshipError
from vre.core.models import (
    Depth,
    DepthGap,
    DepthLevel,
    ExistenceGap,
    KnowledgeGap,
    Primitive,
    Provenance,
    ProvenanceSource,
    ReachabilityGap,
    Relatum,
    RelationalGap,
)
from vre.learning.callback import LearningCallback
from vre.learning.models import (
    CandidateDecision,
    DepthCandidate,
    ExistenceCandidate,
    LearningCandidate,
    LearningResult,
    ProposedDepth,
    ReachabilityCandidate,
    RelationalCandidate,
)
from vre.learning.templates import TemplateFactory


def _make_provenance(decision: CandidateDecision) -> Provenance:
    """
    Derive provenance from the candidate decision.
    """
    source = (
        ProvenanceSource.LEARNED
        if decision == CandidateDecision.ACCEPTED
        else ProvenanceSource.CONVERSATIONAL
    )
    detail = (
        "auto-learning: accepted as proposed"
        if decision == CandidateDecision.ACCEPTED
        else "auto-learning: modified by user"
    )
    now = datetime.now(timezone.utc)
    return Provenance(source=source, created_at=now, updated_at=now, detail=detail)


def _to_depth(proposed: ProposedDepth, provenance: Provenance) -> Depth:
    """
    Convert a ProposedDepth (agent-facing) to a Depth (graph-facing) with provenance.
    """
    return Depth(level=proposed.level, properties=proposed.properties, provenance=provenance)


def _resolve_name_to_id(name: str, grounding: GroundingResult) -> UUID:
    """
    Resolve a primitive name to its UUID from the grounding trace.
    """
    if grounding.trace:
        for p in grounding.trace.result.primitives:
            if p.name.lower() == name.lower():
                return p.id
    raise CandidateValidationError(f"Cannot resolve '{name}' to a primitive ID from the grounding trace")


logger = logging.getLogger(__name__)


class LearningEngine:
    """
    Processes knowledge gaps via the auto-learning loop.

    The engine:
    1. Creates a template from the gap
    2. Invokes the callback (agent fills template, user reviews)
    3. Validates and persists accepted/modified candidates using gap context
    4. Returns the result with the decision
    """

    def __init__(self, repository: PrimitiveRepository) -> None:
        self._repo = repository

    def _persist_existence(
        self, gap: ExistenceGap, candidate: ExistenceCandidate, provenance: Provenance,
    ) -> None:
        """
        Persist a new primitive with D0 (auto-generated) and agent-provided D1.
        """
        if candidate.d1 is None:
            logger.warning("ExistenceCandidate %r missing D1 (identity)", candidate.name)
            raise CandidateValidationError(f"ExistenceCandidate '{candidate.name}' is missing D1 (identity)")

        d0 = Depth(
            level=DepthLevel.EXISTENCE,
            properties={"exists": True},
            provenance=provenance,
        )
        d1 = _to_depth(candidate.d1, provenance)

        primitive = Primitive(
            name=candidate.name,
            depths=[d0, d1],
            provenance=provenance,
        )
        self._repo.save_primitive(primitive)
        logger.info("Persisted new primitive %r with D0+D1", candidate.name)

    def _merge_depths(
        self, primitive: Primitive, new_depths: list[ProposedDepth], provenance: Provenance,
    ) -> None:
        """
        Merge proposed depth levels into a primitive and persist.

        Converts ProposedDepth → Depth, replaces existing depth levels when the
        level already exists, appends otherwise. Sorts and saves.
        """
        logger.debug(
            "Merging %d proposed depth(s) into %r (existing levels: %s)",
            len(new_depths), primitive.name,
            sorted(int(d.level) for d in primitive.depths),
        )
        existing_levels = {d.level for d in primitive.depths}
        touched_levels: set[DepthLevel] = set()
        for proposed in new_depths:
            depth = _to_depth(proposed, provenance)
            touched_levels.add(depth.level)
            if depth.level in existing_levels:
                # Carry forward relata from the old depth — edges and properties
                # are independent concerns; replacing descriptive knowledge should
                # not silently drop validated relationships.
                old = next(d for d in primitive.depths if d.level == depth.level)
                depth.relata = old.relata
                primitive.depths = [
                    d if d.level != depth.level else depth for d in primitive.depths
                ]
            else:
                primitive.depths.append(depth)
            existing_levels.add(depth.level)

        # Stamp provenance on carried-forward relata that lack it
        for depth in primitive.depths:
            if depth.level not in touched_levels:
                continue
            for relatum in depth.relata:
                if relatum.provenance is None:
                    relatum.provenance = provenance

        primitive.depths.sort(key=lambda d: int(d.level))
        self._repo.save_primitive(primitive)

    def _persist_depth(
        self, gap: DepthGap, candidate: DepthCandidate, provenance: Provenance,
    ) -> None:
        """
        Merge new depth levels into an existing primitive and persist.
        """
        if not candidate.new_depths:
            logger.warning("DepthCandidate for %r has no new depths", gap.primitive.name)
            raise CandidateValidationError(f"DepthCandidate for '{gap.primitive.name}' has no new depths")

        existing = self._repo.find_by_id(gap.primitive.id)
        if existing is None:
            raise CandidateValidationError(f"Primitive '{gap.primitive.name}' ({gap.primitive.id}) not found")

        self._merge_depths(existing, candidate.new_depths, provenance)
        logger.debug(
            "Merged depths into %r: levels=%s",
            existing.name, [int(d.level) for d in candidate.new_depths],
        )

    def _persist_relational(
        self, gap: RelationalGap, candidate: RelationalCandidate, provenance: Provenance,
    ) -> None:
        """
        Merge new depth levels into the target primitive and persist.
        """
        if not candidate.new_depths:
            logger.warning("RelationalCandidate for %r has no new depths", gap.target.name)
            raise CandidateValidationError(f"RelationalCandidate for '{gap.target.name}' has no new depths")

        target = self._repo.find_by_id(gap.target.id)
        if target is None:
            raise CandidateValidationError(f"Target '{gap.target.name}' ({gap.target.id}) not found")

        self._merge_depths(target, candidate.new_depths, provenance)
        logger.debug("Merged relational depths into target %r", target.name)

    def _learn_missing_depths(
        self,
        primitive: Primitive,
        required_level: DepthLevel,
        grounding: GroundingResult,
        callback: LearningCallback,
    ) -> CandidateDecision | None:
        """
        If the primitive lacks the required depth level, synthesize a DepthGap
        and invoke the callback to learn the missing depths. Returns the
        decision if learning was needed, or None if the depth already exists.
        """
        existing_levels = {d.level for d in primitive.depths}
        if required_level in existing_levels:
            logger.debug("Depth D%d already present on %r, skipping sub-learning", int(required_level), primitive.name)
            return None

        current_depth = max(existing_levels, key=lambda lv: lv.value) if existing_levels else None
        logger.debug(
            "Synthesized DepthGap for %r: requires D%d, current=%s",
            primitive.name, int(required_level),
            ("D" + str(int(current_depth))) if current_depth is not None else "None",
        )
        gap = DepthGap(
            primitive=primitive,
            required_depth=required_level,
            current_depth=current_depth,
        )
        template = TemplateFactory.from_gap(gap)
        filled, decision = callback(template, grounding, gap)

        if decision in (CandidateDecision.SKIPPED, CandidateDecision.REJECTED):
            return decision
        if filled is None:
            logger.warning("Callback returned None for synthesized depth gap on %r, treating as REJECTED", primitive.name)
            return CandidateDecision.REJECTED

        provenance = _make_provenance(decision)
        self._persist_depth(gap, filled, provenance)
        return decision

    def _persist_reachability(
        self,
        gap: ReachabilityGap,
        candidate: ReachabilityCandidate,
        grounding: GroundingResult,
        callback: LearningCallback,
        provenance: Provenance,
    ) -> CandidateDecision:
        """
        Two-phase edge placement:
        1. Learn any missing depths on source and target via the callback
        2. Place the edge once both sides have the required depths

        If depth learning is rejected or skipped, edge placement is abandoned.
        """
        if candidate.target_name is None or candidate.relation_type is None:
            logger.warning(
                "ReachabilityCandidate for %r missing target_name or relation_type",
                gap.primitive.name,
            )
            raise CandidateValidationError(
                f"ReachabilityCandidate for '{gap.primitive.name}' is missing "
                f"target_name or relation_type"
            )
        if candidate.source_depth_level is None or candidate.target_depth_level is None:
            raise CandidateValidationError(
                f"ReachabilityCandidate for '{gap.primitive.name}' is missing "
                f"source_depth_level or target_depth_level"
            )

        target_id = _resolve_name_to_id(candidate.target_name, grounding)

        source = self._repo.find_by_id(gap.primitive.id)
        if source is None:
            raise CandidateValidationError(f"Source '{gap.primitive.name}' ({gap.primitive.id}) not found")

        target = self._repo.find_by_id(target_id)
        if target is None:
            raise CandidateValidationError(f"Target '{candidate.target_name}' ({target_id}) not found")

        # Phase 1: learn missing depths on source, then target
        logger.debug(
            "Reachability two-phase: learning missing depths on source=%r, target=%r",
            source.name, target.name,
        )
        result = self._learn_missing_depths(source, candidate.source_depth_level, grounding, callback)
        if result in (CandidateDecision.REJECTED, CandidateDecision.SKIPPED):
            return result
        if result is not None:
            source = self._repo.find_by_id(source.id)

        result = self._learn_missing_depths(target, candidate.target_depth_level, grounding, callback)
        if result in (CandidateDecision.REJECTED, CandidateDecision.SKIPPED):
            return result

        # Phase 2: place the edge
        logger.debug(
            "Reachability two-phase: placing %s edge from %r (D%d) to %r (D%d)",
            candidate.relation_type.value, source.name, int(candidate.source_depth_level),
            candidate.target_name, int(candidate.target_depth_level),
        )
        depth_obj = next(d for d in source.depths if d.level == candidate.source_depth_level)

        new_relatum = Relatum(
            relation_type=candidate.relation_type,
            target_id=target_id,
            target_depth=candidate.target_depth_level,
            provenance=provenance,
        )
        depth_obj.relata.append(new_relatum)

        source.depths.sort(key=lambda d: int(d.level))

        try:
            self._repo.save_primitive(source)
        except CyclicRelationshipError as exc:
            logger.error(
                "Cyclic relationship error placing %s edge from %r (%s) to %s: %s",
                candidate.relation_type.value, gap.primitive.name,
                gap.primitive.id, target_id, exc,
            )
            depth_obj.relata.remove(new_relatum)
            raise

        return CandidateDecision.ACCEPTED

    def _persist(
        self,
        gap: KnowledgeGap,
        candidate: LearningCandidate,
        grounding: GroundingResult,
        callback: LearningCallback,
        provenance: Provenance,
    ) -> CandidateDecision | None:
        """
        Validate and persist a filled candidate to the graph.
        Returns a decision override for reachability (two-phase), or None.
        """
        if isinstance(gap, ExistenceGap) and isinstance(candidate, ExistenceCandidate):
            self._persist_existence(gap, candidate, provenance)
        elif isinstance(gap, DepthGap) and isinstance(candidate, DepthCandidate):
            self._persist_depth(gap, candidate, provenance)
        elif isinstance(gap, RelationalGap) and isinstance(candidate, RelationalCandidate):
            self._persist_relational(gap, candidate, provenance)
        elif isinstance(gap, ReachabilityGap) and isinstance(candidate, ReachabilityCandidate):
            return self._persist_reachability(gap, candidate, grounding, callback, provenance)
        return None

    def learn_at(
        self,
        grounding: GroundingResult,
        gap_index: int,
        callback: LearningCallback,
    ) -> LearningResult:
        """
        Process the gap at the given index in the grounding result.
        """
        if not grounding.gaps:
            raise CandidateValidationError("No gaps to learn from")
        if gap_index < 0 or gap_index >= len(grounding.gaps):
            raise CandidateValidationError(f"Gap index {gap_index} out of range (0..{len(grounding.gaps) - 1})")

        gap = grounding.gaps[gap_index]
        logger.info("Learning at gap_index=%d, gap_kind=%s", gap_index, gap.kind)
        candidate = TemplateFactory.from_gap(gap)
        filled, decision = callback(candidate, grounding, gap)

        if decision == CandidateDecision.SKIPPED:
            logger.info("Gap %d skipped by callback", gap_index)
            return LearningResult(decision=CandidateDecision.SKIPPED, candidate=candidate)

        if decision == CandidateDecision.REJECTED or filled is None:
            logger.info("Gap %d rejected by callback", gap_index)
            return LearningResult(decision=CandidateDecision.REJECTED, candidate=candidate)

        logger.debug(
            "Persisting %s candidate for gap %d (decision=%s)",
            type(filled).__name__, gap_index, decision.value,
        )
        provenance = _make_provenance(decision)
        override = self._persist(gap, filled, grounding, callback, provenance)

        # Reachability two-phase can override the decision if depth
        # learning was rejected or skipped
        if override in (CandidateDecision.REJECTED, CandidateDecision.SKIPPED):
            logger.info("Reachability two-phase override: %s", override.value)
            return LearningResult(decision=override, candidate=filled)

        return LearningResult(decision=decision, candidate=filled)

