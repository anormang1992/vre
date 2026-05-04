# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
MetricsManager — coordinates per-primitive metric updates after grounding
and learning operations. Updates are best-effort; failures are logged and
never raise to the caller.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from vre.core.graph import PrimitiveRepository
from vre.core.grounding.models import GroundingResult
from vre.core.models import (
    DepthGap,
    ExistenceGap,
    KnowledgeGap,
    PrimitiveMetrics,
    ReachabilityGap,
    RelationalGap,
    gap_primitive_ids,
)
from vre.learning.models import CandidateDecision


logger = logging.getLogger(__name__)


class MetricsManager:
    """
    Internal coordinator for primitive-level grounding and learning metrics.
    """

    def __init__(self, repository: PrimitiveRepository) -> None:
        self._repo = repository

    def update_grounding(self, result: GroundingResult) -> None:
        """
        Update per-primitive grounding metrics after a `check` call.

        Batch-reads current metrics for all resolved root concepts, computes
        increments in-process, and batch-writes the results.
        """
        resolved_lower = {r.lower() for r in result.resolved}
        target_prims = [
            prim for prim in result.get_primitives()
            if prim.name.lower() in resolved_lower
        ]

        current_metrics: dict[UUID, PrimitiveMetrics | None] | None = None
        if target_prims:
            try:
                current_metrics = self._repo.batch_read_metrics([p.id for p in target_prims])
            except Exception:
                logger.warning("Failed to batch-read metrics", exc_info=True)

        updates: dict[UUID, PrimitiveMetrics] = {}
        if current_metrics is not None:
            now = datetime.now(timezone.utc)
            gap_ids = gap_primitive_ids(result.gaps)
            for prim in target_prims:
                if prim.id not in current_metrics:
                    continue
                metrics = current_metrics[prim.id] or PrimitiveMetrics()
                if prim.id in gap_ids:
                    metrics.failure_count += 1
                    metrics.last_failed = now
                else:
                    metrics.grounding_count += 1
                    metrics.last_grounded = now
                updates[prim.id] = metrics

        if updates:
            try:
                self._repo.batch_update_metrics(updates)
            except Exception:
                logger.warning("Failed to batch-update metrics for %d primitives", len(updates), exc_info=True)

    def update_learning(
        self,
        gap: KnowledgeGap,
        decision: CandidateDecision,
    ) -> None:
        """
        Update learning metrics on the primitive targeted by a gap.

        Increments `learning_count` for accepted/modified decisions and
        `rejection_count` for rejected decisions. SKIPPED decisions are
        ignored. Looks up the primitive by ID first, falling back to name
        for ExistenceGaps where the gap carries a transient ID.
        """
        prim_id: UUID | None = None
        prim_name: str | None = None
        if decision != CandidateDecision.SKIPPED:
            match gap:
                case RelationalGap():
                    prim_id, prim_name = gap.target.id, gap.target.name
                case DepthGap() | ExistenceGap() | ReachabilityGap():
                    prim_id, prim_name = gap.primitive.id, gap.primitive.name

        found = None
        if prim_id is not None:
            found = self._repo.find_by_id(prim_id)
            if found is None and prim_name is not None:
                found = self._repo.find_by_name(prim_name)

        if found is not None:
            metrics = found.metrics or PrimitiveMetrics()
            if decision in (CandidateDecision.ACCEPTED, CandidateDecision.MODIFIED):
                metrics.learning_count += 1
            elif decision == CandidateDecision.REJECTED:
                metrics.rejection_count += 1
            try:
                self._repo.update_metrics(found.id, metrics)
            except Exception:
                logger.warning("Failed to update learning metrics for %r", prim_name, exc_info=True)
