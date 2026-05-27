# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
MetricsManager — coordinates per-primitive metric updates after grounding
operations. Updates are best-effort; failures are logged and never raise
to the caller.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from vre.core.backends import Repository
from vre.core.grounding.models import GroundingResult
from vre.core.models import (
    PrimitiveMetrics,
    gap_primitive_ids,
)


logger = logging.getLogger(__name__)


class MetricsManager:
    """
    Internal coordinator for primitive-level grounding metrics.
    """

    def __init__(self, repository: Repository) -> None:
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
