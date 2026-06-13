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

from vre.core.errors import (
    CandidateValidationError,
    CyclicRelationshipError,
    GapResolvedError,
)
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
    format_depth_label,
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


def _raise_if_resolved(
    name: str,
    live_current: DepthLevel | None,
    required: DepthLevel,
) -> None:
    """
    Raise GapResolvedError if the live primitive already satisfies the gap.

    A gap is a snapshot the caller holds; by the time learn_gap runs, the live
    primitive may already be grounded to (or past) the required depth — closed by
    a concurrent learn round, a seeder, or a sibling gap that cascaded. That is a
    benign divergence, not a bad candidate: persisting now would only overwrite
    grounded knowledge, so we report it and let the integrator re-ground.
    """
    if live_current is not None and live_current >= required:
        raise GapResolvedError(
            f"Gap for '{name}' is already resolved: live depth "
            f"{format_depth_label(live_current)} satisfies required "
            f"{format_depth_label(required)} — nothing to learn"
        )


def _validate_depth_fill(
    new_depths: list[ProposedDepth],
    current: DepthLevel | None,
    required: DepthLevel,
    subject: str,
) -> None:
    """
    Validate proposed depth levels against the gap's scope and the contiguity
    that grounding requires.

    Enforced at the persistence gate against the *live* primitive's contiguous
    max depth (`current`), never a gap snapshot — see `_persist_depth` /
    `_persist_relational`. Three guarantees, in order:

    - Non-empty, no duplicates: there is something to learn, and each level is
      proposed at most once (a repeated level would silently last-write-win in
      the merge).
    - Scope (gap 1): every level satisfies `current < level <= required`. This
      bounds what the candidate may *write* — not the depth the primitive ends up
      grounded to. Filling a contiguity hole legitimately reactivates pre-existing
      authored deeper levels (e.g. filling D2 over `{D0, D1, D3}` re-grounds the
      authored D3 and its edges); that is the contiguity model, not candidate
      escalation. The candidate still wrote only within `(current, required]`.
    - Contiguity (gap 4): the levels extend the existing contiguous chain with
      no holes. `current` is the live contiguous max, so a valid fill is the
      run `current+1 .. max(level)`. A hole would leave the highest level
      invisible to contiguity-strict grounding, never closing the gap.
    """
    if not new_depths:
        raise CandidateValidationError(f"{subject} has no new depths")

    proposed_levels = [proposed.level for proposed in new_depths]
    levels = set(proposed_levels)
    if len(proposed_levels) != len(levels):
        raise CandidateValidationError(
            f"{subject} proposes duplicate depth levels "
            f"({sorted(format_depth_label(level) for level in proposed_levels)}); "
            f"each level may appear at most once"
        )
    for level in sorted(levels):
        if current is not None and level <= current:
            raise CandidateValidationError(
                f"{subject} proposes {format_depth_label(level)} at or below the "
                f"primitive's current depth {format_depth_label(current)} — learning "
                f"may not overwrite already-grounded depths"
            )
        if level > required:
            raise CandidateValidationError(
                f"{subject} proposes {format_depth_label(level)} above the depth "
                f"the gap requires ({format_depth_label(required)}) — learning may "
                f"not escalate scope"
            )

    start = DepthLevel.EXISTENCE if current is None else DepthLevel(int(current) + 1)
    expected = {DepthLevel(value) for value in range(int(start), int(max(levels)) + 1)}
    if levels != expected:
        raise CandidateValidationError(
            f"{subject} must form a contiguous chain extending "
            f"{format_depth_label(current)}; got "
            f"{sorted(format_depth_label(level) for level in levels)}, which "
            f"leaves a hole"
        )


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
            # Grounding gates an edge on the source's *contiguous* max depth, so a
            # level present but detached from D0 (e.g. {D0, D1, D3} requiring D3)
            # is not enough — surface the hole as a DepthGap to fill first.
            contiguous = primitive.contiguous_max_depth
            if contiguous is None or contiguous < required_level:
                prerequisites.append(DepthGap(
                    primitive=primitive,
                    required_depth=required_level,
                    current_depth=contiguous,
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
        # Existence analog of _raise_if_resolved: a stale ExistenceGap may be
        # replayed after the concept was created elsewhere. Creating a second node
        # would duplicate it (Neo4j) or hit the NOCASE unique index (SQLite); either
        # way the gap is already resolved, so report that instead.
        if self._repo.find_by_name(candidate.name) is not None:
            raise GapResolvedError(
                f"Concept '{candidate.name}' already exists — the existence gap is "
                f"already resolved; nothing to learn"
            )

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

        # Validate against the live primitive, not the gap snapshot: current is the
        # live contiguous max, so scope and contiguity guard what is actually written.
        live_current = existing.contiguous_max_depth
        _raise_if_resolved(existing.name, live_current, gap.required_depth)
        _validate_depth_fill(
            candidate.new_depths, live_current, gap.required_depth,
            f"DepthCandidate for '{existing.name}'",
        )

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

        # Validate against the live target, not the gap snapshot (see _persist_depth).
        live_current = target.contiguous_max_depth
        _raise_if_resolved(target.name, live_current, gap.required_depth)
        _validate_depth_fill(
            candidate.new_depths, live_current, gap.required_depth,
            f"RelationalCandidate for '{target.name}'",
        )

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

        # An edge resolves only when both endpoints are *contiguously* grounded to
        # the required level; a detached level (e.g. {D0, D1, D3}) leaves the edge
        # invisible to grounding, so reject it rather than place a dead edge.
        source_contiguous = source.contiguous_max_depth
        if source_contiguous is None or source_contiguous < candidate.source_depth_level:
            raise CandidateValidationError(
                f"Source '{source.name}' requires "
                f"{format_depth_label(candidate.source_depth_level)} but is only "
                f"contiguously grounded to {format_depth_label(source_contiguous)}. "
                f"Fill the DepthGap first."
            )

        target_contiguous = target.contiguous_max_depth
        if target_contiguous is None or target_contiguous < candidate.target_depth_level:
            raise CandidateValidationError(
                f"Target '{target.name}' requires "
                f"{format_depth_label(candidate.target_depth_level)} but is only "
                f"contiguously grounded to {format_depth_label(target_contiguous)}. "
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
            case _:
                raise CandidateValidationError(
                    f"No persistence path for gap '{gap.kind}' with candidate '{candidate.kind}'"
                )

    def learn_gap(
        self,
        gap: KnowledgeGap,
        candidate: LearningCandidate,
    ) -> None:
        """
        Validate and persist a filled candidate for the given gap.

        Knowledge persisted through this path is always stamped LEARNED: by
        construction it originates from an agent-proposed candidate that a human
        approved at this boundary. There is no way to forge AUTHORED provenance
        here — true from-scratch authoring goes through the repository directly.

        Raises CandidateValidationError if the candidate is invalid.
        """
        logger.info("Learning gap: %s", gap.kind)
        if gap.kind != candidate.kind:
            raise CandidateValidationError(
                f"Candidate kind '{candidate.kind}' does not match gap kind '{gap.kind}'"
            )
        candidate.validate_for_gap(gap)
        provenance = _make_provenance(ProvenanceSource.LEARNED)
        self._persist(gap, candidate, provenance)
