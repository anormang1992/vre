"""
Unit tests for Repository.upsert_primitive — the concrete method on the
abstract base class. _FakeRepo implements the minimum abstract surface
needed by upsert_primitive.
"""

import logging
from uuid import UUID

import pytest

from vre.core.backends import Repository
from vre.core.models import (
    Depth,
    DepthLevel,
    Primitive,
    PrimitiveMetrics,
    Provenance,
    ProvenanceSource,
    ResolvedSubgraph,
)


SEED_PROVENANCE = Provenance(source=ProvenanceSource.AUTHORED)


def _make_primitive(name: str) -> Primitive:
    return Primitive(
        name=name,
        provenance=SEED_PROVENANCE,
        depths=[Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE)],
    )


class _FakeRepo(Repository):
    """Repository with find_by_name and save_primitive stubbed."""

    def __init__(self) -> None:
        self._existing: Primitive | None = None
        self._saved: list[Primitive] = []

    def find_by_name(self, name: str) -> Primitive | None:
        return self._existing

    def save_primitive(self, primitive: Primitive) -> None:
        self._saved.append(primitive)

    def find_by_id(self, id: UUID) -> Primitive | None: raise NotImplementedError
    def list_names(self) -> list[str]: raise NotImplementedError
    def delete_primitive(self, id: UUID) -> bool: raise NotImplementedError
    def clear(self) -> int: raise NotImplementedError
    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None: raise NotImplementedError
    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]: raise NotImplementedError
    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None: raise NotImplementedError
    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph: raise NotImplementedError


def test_upsert_creates_when_no_existing() -> None:
    repo = _FakeRepo()
    incoming = _make_primitive("file")
    original_id = incoming.id

    saved = repo.upsert_primitive(incoming)

    assert saved.id == original_id
    assert repo._saved == [saved]


def test_upsert_preserves_existing_id() -> None:
    repo = _FakeRepo()
    existing = _make_primitive("file")
    repo._existing = existing
    incoming = _make_primitive("file")

    saved = repo.upsert_primitive(incoming)

    assert saved.id == existing.id
    assert saved.id != incoming.id
    assert repo._saved == [saved]


def test_upsert_logs_on_overwrite(caplog: pytest.LogCaptureFixture) -> None:
    repo = _FakeRepo()
    repo._existing = _make_primitive("file")
    incoming = _make_primitive("file")

    with caplog.at_level(logging.INFO, logger="vre.core.backends.repository"):
        repo.upsert_primitive(incoming)

    assert any("Upserting 'file'" in r.message for r in caplog.records)


def test_upsert_does_not_log_when_new(caplog: pytest.LogCaptureFixture) -> None:
    repo = _FakeRepo()
    incoming = _make_primitive("file")

    with caplog.at_level(logging.INFO, logger="vre.core.backends.repository"):
        repo.upsert_primitive(incoming)

    assert not any("Upserting" in r.message for r in caplog.records)
