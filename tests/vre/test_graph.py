"""
Unit tests for PrimitiveRepository methods that have pure Python logic
on top of the Neo4j calls. We bypass __init__ (which would require a
live driver) and substitute the lower-level methods we need.
"""

import logging

import pytest

from vre.core.graph import PrimitiveRepository
from vre.core.models import Depth, DepthLevel, Primitive, Provenance, ProvenanceSource


SEED_PROVENANCE = Provenance(source=ProvenanceSource.AUTHORED)


def _make_primitive(name: str) -> Primitive:
    return Primitive(
        name=name,
        provenance=SEED_PROVENANCE,
        depths=[Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE)],
    )


class _FakeRepo(PrimitiveRepository):
    """PrimitiveRepository with find_by_name and save_primitive stubbed."""

    def __init__(self) -> None:
        self._existing: Primitive | None = None
        self._saved: list[Primitive] = []

    def find_by_name(self, name: str) -> Primitive | None:
        return self._existing

    def save_primitive(self, primitive: Primitive) -> None:
        self._saved.append(primitive)


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

    with caplog.at_level(logging.INFO, logger="vre.core.graph"):
        repo.upsert_primitive(incoming)

    assert any("Upserting 'file'" in r.message for r in caplog.records)


def test_upsert_does_not_log_when_new(caplog: pytest.LogCaptureFixture) -> None:
    repo = _FakeRepo()
    incoming = _make_primitive("file")

    with caplog.at_level(logging.INFO, logger="vre.core.graph"):
        repo.upsert_primitive(incoming)

    assert not any("Upserting" in r.message for r in caplog.records)
