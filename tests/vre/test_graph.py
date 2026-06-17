"""
Unit tests for Repository.upsert_primitive — the concrete method on the
abstract base class. _FakeRepo implements the minimum abstract surface
needed by upsert_primitive.
"""

import logging
from uuid import UUID

import pytest

from vre.core.backends import Repository
from vre.core.backends.repository import CURRENT_SCHEMA_VERSION, reconcile_schema_version
from vre.core.errors import SchemaVersionError
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


class TestReconcileSchemaVersion:
    """Pure three-case comparator from repository.py — no DB involved."""

    def test_equal_version_passes(self) -> None:
        # disk == current: returns None, raises nothing.
        assert reconcile_schema_version(CURRENT_SCHEMA_VERSION) is None

    def test_newer_disk_version_fails_loud(self) -> None:
        with pytest.raises(SchemaVersionError):
            reconcile_schema_version(CURRENT_SCHEMA_VERSION + 1)

    def test_older_disk_version_fails_loud(self) -> None:
        # disk < current with no migration logic wired: must fail loud rather
        # than silently load an old-format store under new-format assumptions.
        # `current` is injected so the case is reachable while CURRENT is 1.
        with pytest.raises(SchemaVersionError):
            reconcile_schema_version(disk=1, current=2)

    def test_constant_is_one(self) -> None:
        # Pins the current format. Bump this (and the assertion) only when the
        # persisted format actually changes.
        assert CURRENT_SCHEMA_VERSION == 1
