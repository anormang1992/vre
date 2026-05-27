# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
LearningEngine — validates and persists candidate fills for knowledge gaps.

Integrators identify gaps via VRE.check(), create templates via
template_for_gap(), fill them however they choose, and pass the filled
candidate here for validation and persistence.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from vre.core.errors import CandidateValidationError, CyclicRelationshipError
from vre.core.backends import Repository
from vre.core.models import (
    Depth,
    DepthLevel,
    ExistenceGap,
    KnowledgeGap,
    Primitive,
    Provenance,
    ProvenanceSource,
    ReachabilityGap,
    Relatum,
    RelationalGap,
    DepthGap,
)
from vre.learning.models import (
    DepthCandidate,
    ExistenceCandidate,
    LearningCandidate,
    ProposedDepth,
    ReachabilityCandidate,
    RelationalCandidate,
)


def _make_provenance(source: ProvenanceSource) -> Provenance:
    """
    Create a timestamped provenance record with the given source.
    """
    now = datetime.now(timezone.utc)
    return Provenance(source=source, created_at=now, updated_at=now)


def _to_depth(proposed: ProposedDepth, provenance: Provenance) -> Depth:
    """
    Convert a ProposedDepth (agent-facing) to a Depth (graph-facing) with provenance.
    """
    return Depth(level=proposed.level, properties=proposed.properties, provenance=provenance)


logger = logging.getLogger(__name__)


class LearningEngine:
    """
    Validates and persists candidate fills for knowledge gaps.

    The engine does not orchestrate a learning loop — that is the
    integrator's responsibility. It provides a single entry point,
    `learn_gap`, which validates a filled candidate against its gap
    and persists it to the graph.
    """

    def __init__(self, repository: Repository) -> None:
        self._repo = repository

    def reachability_prerequisites(
        self,
        gap: ReachabilityGap,
        candidate: ReachabilityCandidate,
    ) -> list[DepthGap]:
        """
        Return DepthGaps that must be filled before this edge can be placed.

        Checks that both source and target have the required depth levels.
        Returns an empty list if no prerequisites are missing.
        """
        candidate.validate_for_gap(gap)

        prerequisites: list[DepthGap] = []
        for name, required_level in [
            (candidate.source_name, candidate.source_depth_level),
            (candidate.target_name, candidate.target_depth_level),
        ]:
            primitive = self._repo.find_by_name(name)
            if primitive is None:
                raise CandidateValidationError(f"Cannot resolve '{name}' to a primitive ID")
            existing_levels = {d.level for d in primitive.depths}
            if required_level not in existing_levels:
                prerequisites.append(DepthGap(
                    primitive=primitive,
                    required_depth=required_level,
                    current_depth=primitive.contiguous_max_depth,
                ))

        return prerequisites

    def _resolve_name_to_id(self, name: str) -> UUID:
        """
        Resolve a primitive name to its UUID from the repository.
        """
        primitive = self._repo.find_by_name(name)
        if primitive is None:
            raise CandidateValidationError(f"Cannot resolve '{name}' to a primitive ID")
        return primitive.id

    def _persist_existence(
        self, gap: ExistenceGap, candidate: ExistenceCandidate, provenance: Provenance,
    ) -> None:
        """
        Persist a new primitive with D0 (auto-generated) and agent-provided D1.
        """
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

        Converts ProposedDepth -> Depth, replaces existing depth levels when the
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
                old = next(d for d in primitive.depths if d.level == depth.level)
                depth.relata = old.relata
                primitive.depths = [
                    d if d.level != depth.level else depth for d in primitive.depths
                ]
            else:
                primitive.depths.append(depth)
            existing_levels.add(depth.level)

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
        target = self._repo.find_by_id(gap.target.id)
        if target is None:
            raise CandidateValidationError(f"Target '{gap.target.name}' ({gap.target.id}) not found")

        self._merge_depths(target, candidate.new_depths, provenance)
        logger.debug("Merged relational depths into target %r", target.name)

    def _persist_reachability(
        self,
        gap: ReachabilityGap,
        candidate: ReachabilityCandidate,
        provenance: Provenance,
    ) -> None:
        """
        Resolve names, check depth requirements, and place an edge.
        """
        source_id = self._resolve_name_to_id(candidate.source_name)
        target_id = self._resolve_name_to_id(candidate.target_name)

        source = self._repo.find_by_id(source_id)
        if source is None:
            raise CandidateValidationError(f"Source '{candidate.source_name}' ({source_id}) not found")

        target = self._repo.find_by_id(target_id)
        if target is None:
            raise CandidateValidationError(f"Target '{candidate.target_name}' ({target_id}) not found")

        source_levels = {d.level for d in source.depths}
        if candidate.source_depth_level not in source_levels:
            raise CandidateValidationError(
                f"Source '{source.name}' requires D{candidate.source_depth_level.value} "
                f"({candidate.source_depth_level.name}) but only has "
                f"{sorted('D' + str(int(lv)) for lv in source_levels)}. "
                f"Fill the DepthGap first."
            )

        target_levels = {d.level for d in target.depths}
        if candidate.target_depth_level not in target_levels:
            raise CandidateValidationError(
                f"Target '{target.name}' requires D{candidate.target_depth_level.value} "
                f"({candidate.target_depth_level.name}) but only has "
                f"{sorted('D' + str(int(lv)) for lv in target_levels)}. "
                f"Fill the DepthGap first."
            )

        logger.debug(
            "Placing %s edge from %r (D%d) to %r (D%d)",
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
        except CyclicRelationshipError:
            logger.exception(
                "Cyclic relationship error placing %s edge from %r (%s) to %r (%s)",
                candidate.relation_type.value,
                source.name,
                source.id,
                target.name,
                target.id,
            )
            depth_obj.relata.remove(new_relatum)
            raise

    def _persist(
        self,
        gap: KnowledgeGap,
        candidate: LearningCandidate,
        provenance: Provenance,
    ) -> None:
        """
        Persist a validated candidate to the graph.
        """
        match (gap, candidate):
            case (ExistenceGap(), ExistenceCandidate()):
                self._persist_existence(gap, candidate, provenance)
            case (DepthGap(), DepthCandidate()):
                self._persist_depth(gap, candidate, provenance)
            case (RelationalGap(), RelationalCandidate()):
                self._persist_relational(gap, candidate, provenance)
            case (ReachabilityGap(), ReachabilityCandidate()):
                self._persist_reachability(gap, candidate, provenance)

    def learn_gap(
        self,
        gap: KnowledgeGap,
        candidate: LearningCandidate,
        source: ProvenanceSource = ProvenanceSource.LEARNED,
    ) -> None:
        """
        Validate and persist a filled candidate for the given gap.

        Raises CandidateValidationError if the candidate is invalid.
        """
        logger.info("Learning gap: %s", gap.kind)
        candidate.validate_for_gap(gap)
        provenance = _make_provenance(source)
        self._persist(gap, candidate, provenance)
