# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Backend-conformance suite (#83): one set of assertions run against every
Repository implementation via the parameterized ``repository`` fixture, so
parity is *verified* rather than asserted in the README.

Seeded here with the case-insensitive name-identity invariants that #78/#96
(uniqueness + loud detection) and #99 item 3 (Unicode folding) are about. The
non-ASCII assertions are only meaningful because both backends now fold names
through the single ``Primitive.fold_name`` (NFC + casefold) definition.

Neo4j params skip cleanly when ``VRE_TEST_NEO4J_URI`` is unset (see conftest).
"""

import unicodedata
from uuid import uuid4

import pytest

from vre.core.errors import GraphError, PersistenceError
from vre.core.models import Depth, DepthLevel, Primitive, Provenance, ProvenanceSource

# e + U+0301 (combining acute). Built via chr() so the code point is explicit in
# pure-ASCII source and can't be silently re-normalized by an editor or git.
_COMBINING_ACUTE = chr(0x0301)


def _prov() -> Provenance:
    return Provenance(source=ProvenanceSource.AUTHORED)


def _make_primitive(name: str, max_depth: DepthLevel = DepthLevel.EXISTENCE) -> Primitive:
    depths = [
        Depth(level=DepthLevel(i), properties={}, provenance=_prov())
        for i in range(max_depth + 1)
    ]
    return Primitive(name=name, depths=depths, provenance=_prov())


class TestNameIdentityConformance:
    """The 'one primitive per case-insensitive name' invariant, both backends."""

    def test_save_and_find_by_id_round_trip(self, repository) -> None:
        # Representative non-uniqueness assertion: proves the harness exercises
        # a basic save/load on every backend, not just the uniqueness paths.
        p = _make_primitive("file", DepthLevel.IDENTITY)
        repository.save_primitive(p)
        loaded = repository.find_by_id(p.id)
        assert loaded is not None
        assert loaded.id == p.id
        assert loaded.name == "file"
        assert {int(d.level) for d in loaded.depths} == {int(d.level) for d in p.depths}

    def test_find_by_name_is_case_insensitive(self, repository) -> None:
        repository.save_primitive(_make_primitive("File"))
        assert repository.find_by_name("file") is not None
        assert repository.find_by_name("FILE") is not None
        assert repository.find_by_name("FiLe") is not None

    def test_find_by_name_missing_returns_none(self, repository) -> None:
        # The not-found contract preserved alongside the loud-on-duplicate guard.
        assert repository.find_by_name("nonexistent") is None

    def test_upsert_preserves_existing_id(self, repository) -> None:
        existing = _make_primitive("file")
        repository.save_primitive(existing)
        incoming = _make_primitive("file")  # same name, new id
        result = repository.upsert_primitive(incoming)
        assert result.id == existing.id
        assert result.id != incoming.id

    def test_case_collision_new_id_is_rejected(self, repository) -> None:
        # #78: a *new id* whose name only differs in case must be rejected loud on
        # BOTH backends — SQLite via the NOCASE->name_lower unique index, Neo4j via
        # the primitive_name_lower_unique constraint. save_primitive (not upsert)
        # is the raw write path that has no id-reuse safety net.
        repository.save_primitive(_make_primitive("File"))
        with pytest.raises(PersistenceError):
            repository.save_primitive(_make_primitive("file"))  # different id

    def test_unicode_case_collision_is_rejected(self, repository) -> None:
        # #99 item 3: with folding unified on casefold(), an accented name and its
        # upper-case variant are the same concept on every backend. Previously
        # SQLite NOCASE (ASCII-only) kept both while Neo4j toLower merged them.
        # .upper() derives the capital form so the pair can't drift out of sync.
        lower = "Cafe" + _COMBINING_ACUTE  # café
        repository.save_primitive(_make_primitive(lower))
        with pytest.raises(PersistenceError):
            repository.save_primitive(_make_primitive(lower.upper()))  # CAFÉ, different id

    def test_unicode_normalization_collision_is_rejected(self, repository) -> None:
        # fold_name NFC-normalizes before casefolding, so two spellings that render
        # identically but differ in code points are one concept. These differ ONLY
        # in normalization form (same case): precomposed "é" (U+00E9) vs "e" +
        # combining acute (U+0065 U+0301). Without the NFC pass, casefold() alone
        # would keep these as two separate primitives.
        base = "Cafe" + _COMBINING_ACUTE
        decomposed = unicodedata.normalize("NFD", base)
        precomposed = unicodedata.normalize("NFC", base)
        assert precomposed != decomposed  # genuinely distinct code-point sequences
        repository.save_primitive(_make_primitive(precomposed))
        assert repository.find_by_name(decomposed) is not None
        with pytest.raises(PersistenceError):
            repository.save_primitive(_make_primitive(decomposed))  # different id


@pytest.mark.requires_neo4j
def test_find_by_name_fails_loud_on_duplicate(neo4j_repository) -> None:
    """#96: if a folded-name duplicate ever exists (e.g. data predating the
    constraint), find_by_name must raise rather than silently return the first
    of multiple nodes. We manufacture that corrupt state by dropping the
    constraint and inserting two colliding nodes via raw Cypher.

    Neo4j-only by design: SQLite's find_by_name has no parallel loud-detection
    test because its corrupt state is unreachable in practice — `.fetchone()`
    can only return a duplicate if the unique idx_primitives_name_lower index
    is first dropped, which the SQLite path never does. Neo4j needs the runtime
    guard because its constraint backs a *materialized* key, so dupes can slip
    in (raw Cypher, data predating the constraint) and must fail loud on read.
    """
    repo = neo4j_repository
    prov = '{"source": "AUTHORED", "detail": null}'
    with repo._driver.session(database=repo._database) as session:
        session.run("DROP CONSTRAINT primitive_name_lower_unique IF EXISTS")
        for _ in range(2):
            session.run(
                "CREATE (:Primitive {id: $id, name: 'File', name_lower: 'file', "
                "depths_json: '[]', provenance_json: $prov, metrics_json: 'null'})",
                id=str(uuid4()),
                prov=prov,
            )

    with pytest.raises(GraphError, match="integrity"):
        repo.find_by_name("file")

    # Restore the invariant for subsequent tests: drop the dupes, re-add constraint.
    repo.clear()
    repo._ensure_constraints()
