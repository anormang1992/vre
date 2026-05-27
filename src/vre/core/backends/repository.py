# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Abstract repository contract for the Volute Reasoning Engine.

Defines the persistence interface that all backends must implement.
Engine modules depend on this contract, never on a concrete backend.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Self
from uuid import UUID

from vre.core.models import (
    Primitive,
    PrimitiveMetrics,
    ResolvedSubgraph,
)


logger = logging.getLogger(__name__)


class Repository(ABC):
    """
    Abstract persistence contract for epistemic primitives.

    Concrete backends (Neo4j, SQLite, in-memory) implement the abstract
    methods; shared logic like upsert and context-manager support lives here.
    """

    # ------------------------------------------------------------------
    # Concrete: lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Concrete: composite operations
    # ------------------------------------------------------------------

    def upsert_primitive(self, primitive: Primitive) -> Primitive:
        """
        Save `primitive`, preserving the id of any existing primitive with the
        same name. Returns the reconciled primitive so callers can cite its
        canonical id in downstream relata.

        Within-domain idempotency only: depths and relata are full-replaced
        (per save_primitive). Cross-domain merging is not handled here.
        """
        existing = self.find_by_name(primitive.name)
        if existing is None:
            canonical = primitive
        else:
            logger.info(
                "Upserting %r: overwriting existing primitive (id=%s)",
                primitive.name, existing.id,
            )
            canonical = primitive.model_copy(update={"id": existing.id})
        self.save_primitive(canonical)
        return canonical

    # ------------------------------------------------------------------
    # Abstract: CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def find_by_id(self, id: UUID) -> Primitive | None:
        """Look up a primitive by UUID. Returns `None` if not found."""
        ...

    @abstractmethod
    def find_by_name(self, name: str) -> Primitive | None:
        """Look up a primitive by name (case-insensitive). Returns `None` if not found."""
        ...

    @abstractmethod
    def save_primitive(self, primitive: Primitive) -> None:
        """Persist a primitive — full replace of depths and relata."""
        ...

    @abstractmethod
    def list_names(self) -> list[str]:
        """Return the names of all primitives, sorted alphabetically."""
        ...

    @abstractmethod
    def delete_primitive(self, id: UUID) -> bool:
        """Delete the primitive with the given UUID and all its relationships. Returns `True` if deleted."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Delete every primitive and its relationships. Returns the count deleted."""
        ...

    # ------------------------------------------------------------------
    # Abstract: metrics
    # ------------------------------------------------------------------

    @abstractmethod
    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        """Update only the metrics on an existing primitive."""
        ...

    @abstractmethod
    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]:
        """Read current metrics for multiple primitives in a single query."""
        ...

    @abstractmethod
    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None:
        """Persist metrics for multiple primitives in a single query."""
        ...

    # ------------------------------------------------------------------
    # Abstract: graph traversal
    # ------------------------------------------------------------------

    @abstractmethod
    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        """Single-query traversal that resolves a subgraph for the given concept names."""
        ...
