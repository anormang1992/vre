# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Comprehensive test suite for SQLiteRepository.
"""

import pytest

from vre.core.backends.sqlite import SQLiteRepository
from vre.core.errors import CyclicRelationshipError
from vre.core.grounding import GroundingEngine
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    PrimitiveMetrics,
    Provenance,
    ProvenanceSource,
    Relatum,
    RelationType,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source=ProvenanceSource.AUTHORED)


def _make_primitive(
    name: str,
    max_depth: DepthLevel = DepthLevel.EXISTENCE,
) -> Primitive:
    depths = [
        Depth(level=DepthLevel(i), provenance=_prov())
        for i in range(max_depth + 1)
    ]
    return Primitive(name=name, depths=depths, provenance=_prov())


def _seed_filesystem_graph(repo: SQLiteRepository) -> dict[str, Primitive]:
    """Seed a small filesystem-domain graph and return primitives by name."""
    operating_system = _make_primitive("operating_system", DepthLevel.CONSTRAINTS)
    filesystem = _make_primitive("filesystem", DepthLevel.CONSTRAINTS)
    path = _make_primitive("path", DepthLevel.CONSTRAINTS)
    directory = _make_primitive("directory", DepthLevel.CONSTRAINTS)
    file = _make_primitive("file", DepthLevel.IDENTITY)
    list_prim = _make_primitive("list", DepthLevel.CAPABILITIES)
    read = _make_primitive("read", DepthLevel.CAPABILITIES)

    # filesystem DEPENDS_ON operating_system @ D2
    filesystem.depths[2].relata.append(
        Relatum(
            relation_type=RelationType.DEPENDS_ON,
            target_id=operating_system.id,
            target_depth=DepthLevel.CAPABILITIES,
            provenance=_prov(),
        )
    )

    # path DEPENDS_ON filesystem @ D2
    path.depths[2].relata.append(
        Relatum(
            relation_type=RelationType.DEPENDS_ON,
            target_id=filesystem.id,
            target_depth=DepthLevel.CAPABILITIES,
            provenance=_prov(),
        )
    )

    # directory REQUIRES path @ D1
    directory.depths[1].relata.append(
        Relatum(
            relation_type=RelationType.REQUIRES,
            target_id=path.id,
            target_depth=DepthLevel.IDENTITY,
            provenance=_prov(),
        )
    )
    # directory DEPENDS_ON filesystem @ D2
    directory.depths[2].relata.append(
        Relatum(
            relation_type=RelationType.DEPENDS_ON,
            target_id=filesystem.id,
            target_depth=DepthLevel.CAPABILITIES,
            provenance=_prov(),
        )
    )

    # file REQUIRES path @ D1
    file.depths[1].relata.append(
        Relatum(
            relation_type=RelationType.REQUIRES,
            target_id=path.id,
            target_depth=DepthLevel.IDENTITY,
            provenance=_prov(),
        )
    )

    # list APPLIES_TO directory @ D2
    list_prim.depths[2].relata.append(
        Relatum(
            relation_type=RelationType.APPLIES_TO,
            target_id=directory.id,
            target_depth=DepthLevel.CAPABILITIES,
            provenance=_prov(),
        )
    )

    # read APPLIES_TO file @ D2
    read.depths[2].relata.append(
        Relatum(
            relation_type=RelationType.APPLIES_TO,
            target_id=file.id,
            target_depth=DepthLevel.CAPABILITIES,
            provenance=_prov(),
        )
    )

    # Save order matters: targets before sources for FK constraints
    for p in [operating_system, filesystem, path, directory, file, list_prim, read]:
        repo.save_primitive(p)

    return {
        "operating_system": operating_system,
        "filesystem": filesystem,
        "path": path,
        "directory": directory,
        "file": file,
        "list": list_prim,
        "read": read,
    }


# ------------------------------------------------------------------
# TestSQLiteLifecycle
# ------------------------------------------------------------------


class TestSQLiteLifecycle:
    def test_constructor_creates_tables(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            cursor = repo._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            assert "primitives" in tables
            assert "relata" in tables

    def test_context_manager_closes_connection(self) -> None:
        repo = SQLiteRepository(":memory:")
        repo.__enter__()
        repo.__exit__(None, None, None)
        with pytest.raises(Exception):
            repo._conn.execute("SELECT 1")

    def test_foreign_keys_enabled(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            row = repo._conn.execute("PRAGMA foreign_keys").fetchone()
            assert row[0] == 1

    def test_wal_mode(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            row = repo._conn.execute("PRAGMA journal_mode").fetchone()
            # :memory: databases may report "memory" instead of "wal"
            assert row[0] in ("wal", "memory")

    def test_default_path(self, tmp_path: object) -> None:
        """Constructor with None path uses the default (not tested for real path, just not :memory:)."""
        # Just verify the attribute exists and is a string
        repo = SQLiteRepository(":memory:")
        assert repo._path == ":memory:"
        repo.close()

    def test_file_based_db(self, tmp_path: object) -> None:
        import pathlib

        db_path = pathlib.Path(str(tmp_path)) / "sub" / "test.db"
        with SQLiteRepository(db_path) as repo:
            repo.save_primitive(_make_primitive("alpha"))
        # Reopen to verify persistence
        with SQLiteRepository(db_path) as repo:
            assert repo.find_by_name("alpha") is not None


# ------------------------------------------------------------------
# TestSaveFindPrimitive
# ------------------------------------------------------------------


class TestSaveFindPrimitive:
    def test_save_and_find_by_id(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File", DepthLevel.IDENTITY)
            repo.save_primitive(prim)
            found = repo.find_by_id(prim.id)
            assert found is not None
            assert found.id == prim.id
            assert found.name == "File"
            assert len(found.depths) == 2

    def test_save_and_find_by_name(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("Directory")
            repo.save_primitive(prim)
            found = repo.find_by_name("Directory")
            assert found is not None
            assert found.name == "Directory"

    def test_find_by_name_case_insensitive(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File")
            repo.save_primitive(prim)
            assert repo.find_by_name("file") is not None
            assert repo.find_by_name("FILE") is not None
            assert repo.find_by_name("FiLe") is not None

    def test_find_by_id_not_found(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            from uuid import uuid4

            assert repo.find_by_id(uuid4()) is None

    def test_find_by_name_not_found(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            assert repo.find_by_name("nonexistent") is None

    def test_overwrite_primitive(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File", DepthLevel.EXISTENCE)
            repo.save_primitive(prim)
            # Overwrite with more depths
            updated = prim.model_copy(
                update={
                    "depths": [
                        Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                        Depth(level=DepthLevel.IDENTITY, provenance=_prov()),
                        Depth(level=DepthLevel.CAPABILITIES, provenance=_prov()),
                    ]
                }
            )
            repo.save_primitive(updated)
            found = repo.find_by_id(prim.id)
            assert found is not None
            assert len(found.depths) == 3

    def test_multiple_depths_with_properties(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = Primitive(
                name="File",
                depths=[
                    Depth(
                        level=DepthLevel.EXISTENCE,
                        properties={"exists": True},
                        provenance=_prov(),
                    ),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        properties={"type": "regular", "extensions": [".txt", ".py"]},
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(prim)
            found = repo.find_by_id(prim.id)
            assert found is not None
            assert found.depths[0].properties == {"exists": True}
            assert found.depths[1].properties == {
                "type": "regular",
                "extensions": [".txt", ".py"],
            }

    def test_relata_round_trip(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            target = _make_primitive("Path", DepthLevel.IDENTITY)
            repo.save_primitive(target)

            source = Primitive(
                name="File",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        relata=[
                            Relatum(
                                relation_type=RelationType.REQUIRES,
                                target_id=target.id,
                                target_depth=DepthLevel.IDENTITY,
                                provenance=_prov(),
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(source)
            found = repo.find_by_id(source.id)
            assert found is not None
            assert len(found.depths[1].relata) == 1
            rel = found.depths[1].relata[0]
            assert rel.relation_type == RelationType.REQUIRES
            assert rel.target_id == target.id
            assert rel.target_depth == DepthLevel.IDENTITY

    def test_relatum_metadata_round_trip(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            target = _make_primitive("Directory", DepthLevel.CAPABILITIES)
            repo.save_primitive(target)

            source = Primitive(
                name="Delete",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(level=DepthLevel.IDENTITY, provenance=_prov()),
                    Depth(
                        level=DepthLevel.CAPABILITIES,
                        relata=[
                            Relatum(
                                relation_type=RelationType.APPLIES_TO,
                                target_id=target.id,
                                target_depth=DepthLevel.CAPABILITIES,
                                metadata={"destructive": True, "tags": ["danger"]},
                                provenance=_prov(),
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(source)
            found = repo.find_by_id(source.id)
            assert found is not None
            rel = found.depths[2].relata[0]
            assert rel.metadata == {"destructive": True, "tags": ["danger"]}

    def test_provenance_round_trip(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File")
            repo.save_primitive(prim)
            found = repo.find_by_id(prim.id)
            assert found is not None
            assert found.provenance is not None
            assert found.provenance.source == ProvenanceSource.AUTHORED
            assert found.depths[0].provenance is not None
            assert found.depths[0].provenance.source == ProvenanceSource.AUTHORED

    def test_relata_provenance_round_trip(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            target = _make_primitive("Path")
            repo.save_primitive(target)

            rel_prov = Provenance(
                source=ProvenanceSource.LEARNED,
                detail="Discovered via execution failure",
            )
            source = Primitive(
                name="File",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        relata=[
                            Relatum(
                                relation_type=RelationType.REQUIRES,
                                target_id=target.id,
                                target_depth=DepthLevel.EXISTENCE,
                                provenance=rel_prov,
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(source)
            found = repo.find_by_id(source.id)
            assert found is not None
            rel = found.depths[1].relata[0]
            assert rel.provenance is not None
            assert rel.provenance.source == ProvenanceSource.LEARNED
            assert rel.provenance.detail == "Discovered via execution failure"

    def test_overwrite_replaces_relata(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            target_a = _make_primitive("TargetA")
            target_b = _make_primitive("TargetB")
            repo.save_primitive(target_a)
            repo.save_primitive(target_b)

            source = Primitive(
                name="Source",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        relata=[
                            Relatum(
                                relation_type=RelationType.APPLIES_TO,
                                target_id=target_a.id,
                                target_depth=DepthLevel.EXISTENCE,
                                provenance=_prov(),
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(source)

            # Overwrite with different relata
            updated = Primitive(
                id=source.id,
                name="Source",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        relata=[
                            Relatum(
                                relation_type=RelationType.APPLIES_TO,
                                target_id=target_b.id,
                                target_depth=DepthLevel.EXISTENCE,
                                provenance=_prov(),
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(updated)

            found = repo.find_by_id(source.id)
            assert found is not None
            assert len(found.depths[1].relata) == 1
            assert found.depths[1].relata[0].target_id == target_b.id


# ------------------------------------------------------------------
# TestListDeleteClear
# ------------------------------------------------------------------


class TestListDeleteClear:
    def test_list_names_sorted(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            for name in ["Zebra", "Alpha", "middle"]:
                repo.save_primitive(_make_primitive(name))
            names = repo.list_names()
            assert names == ["Alpha", "middle", "Zebra"]

    def test_list_names_empty(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            assert repo.list_names() == []

    def test_delete_returns_true(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File")
            repo.save_primitive(prim)
            assert repo.delete_primitive(prim.id) is True
            assert repo.find_by_id(prim.id) is None

    def test_delete_returns_false_not_found(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            from uuid import uuid4

            assert repo.delete_primitive(uuid4()) is False

    def test_delete_removes_outgoing_relata(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            target = _make_primitive("Target")
            repo.save_primitive(target)

            source = Primitive(
                name="Source",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        relata=[
                            Relatum(
                                relation_type=RelationType.APPLIES_TO,
                                target_id=target.id,
                                target_depth=DepthLevel.EXISTENCE,
                                provenance=_prov(),
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(source)
            repo.delete_primitive(source.id)
            # Relationships should be gone
            count = repo._conn.execute(
                "SELECT COUNT(*) FROM relata"
            ).fetchone()[0]
            assert count == 0

    def test_delete_removes_incoming_relata(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            target = _make_primitive("Target")
            repo.save_primitive(target)

            source = Primitive(
                name="Source",
                depths=[
                    Depth(level=DepthLevel.EXISTENCE, provenance=_prov()),
                    Depth(
                        level=DepthLevel.IDENTITY,
                        relata=[
                            Relatum(
                                relation_type=RelationType.APPLIES_TO,
                                target_id=target.id,
                                target_depth=DepthLevel.EXISTENCE,
                                provenance=_prov(),
                            )
                        ],
                        provenance=_prov(),
                    ),
                ],
                provenance=_prov(),
            )
            repo.save_primitive(source)
            repo.delete_primitive(target.id)
            # Relationship row should be gone (deleted as incoming)
            count = repo._conn.execute(
                "SELECT COUNT(*) FROM relata"
            ).fetchone()[0]
            assert count == 0

    def test_clear(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            for name in ["A", "B", "C"]:
                repo.save_primitive(_make_primitive(name))
            deleted = repo.clear()
            assert deleted == 3
            assert repo.list_names() == []

    def test_clear_empty(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            assert repo.clear() == 0


# ------------------------------------------------------------------
# TestMetrics
# ------------------------------------------------------------------


class TestMetrics:
    def test_update_metrics(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File")
            repo.save_primitive(prim)
            from datetime import datetime, timezone

            metrics = PrimitiveMetrics(
                grounding_count=5,
                last_grounded=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            repo.update_metrics(prim.id, metrics)
            found = repo.find_by_id(prim.id)
            assert found is not None
            assert found.metrics is not None
            assert found.metrics.grounding_count == 5

    def test_batch_read_metrics(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            from datetime import datetime, timezone

            p1 = _make_primitive("A")
            p2 = _make_primitive("B")
            repo.save_primitive(p1)
            repo.save_primitive(p2)
            metrics = PrimitiveMetrics(
                grounding_count=3,
                last_grounded=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            repo.update_metrics(p1.id, metrics)

            result = repo.batch_read_metrics([p1.id, p2.id])
            assert p1.id in result
            assert result[p1.id] is not None
            assert result[p1.id].grounding_count == 3
            assert p2.id in result
            assert result[p2.id] is None

    def test_batch_read_metrics_empty(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            assert repo.batch_read_metrics([]) == {}

    def test_batch_update_metrics(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            from datetime import datetime, timezone

            p1 = _make_primitive("A")
            p2 = _make_primitive("B")
            repo.save_primitive(p1)
            repo.save_primitive(p2)

            m1 = PrimitiveMetrics(grounding_count=10)
            m2 = PrimitiveMetrics(
                failure_count=2,
                last_failed=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
            repo.batch_update_metrics({p1.id: m1, p2.id: m2})

            result = repo.batch_read_metrics([p1.id, p2.id])
            assert result[p1.id].grounding_count == 10
            assert result[p2.id].failure_count == 2

    def test_batch_update_metrics_empty(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            repo.batch_update_metrics({})  # should not raise

    def test_metrics_preserved_on_save(self) -> None:
        """Metrics set via update_metrics survive a save_primitive overwrite when included."""
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File")
            repo.save_primitive(prim)
            metrics = PrimitiveMetrics(grounding_count=7)
            repo.update_metrics(prim.id, metrics)

            # Re-save with metrics attached
            found = repo.find_by_id(prim.id)
            repo.save_primitive(found)
            refound = repo.find_by_id(prim.id)
            assert refound.metrics is not None
            assert refound.metrics.grounding_count == 7


# ------------------------------------------------------------------
# TestCycleDetection
# ------------------------------------------------------------------


class TestCycleDetection:
    def test_self_referential_raises(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("A", DepthLevel.IDENTITY)
            repo.save_primitive(prim)

            prim.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=prim.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            with pytest.raises(CyclicRelationshipError):
                repo.save_primitive(prim)

    def test_direct_cycle_raises(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            a = _make_primitive("A", DepthLevel.IDENTITY)
            b = _make_primitive("B", DepthLevel.IDENTITY)
            repo.save_primitive(a)
            repo.save_primitive(b)

            # A -> B
            a.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.DEPENDS_ON,
                    target_id=b.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(a)

            # B -> A would cycle
            b.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.DEPENDS_ON,
                    target_id=a.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            with pytest.raises(CyclicRelationshipError):
                repo.save_primitive(b)

    def test_transitive_cycle_raises(self) -> None:
        """A->B via REQUIRES, B->C via CONSTRAINED_BY, C->A via DEPENDS_ON = cycle."""
        with SQLiteRepository(":memory:") as repo:
            a = _make_primitive("A", DepthLevel.IDENTITY)
            b = _make_primitive("B", DepthLevel.IDENTITY)
            c = _make_primitive("C", DepthLevel.IDENTITY)
            repo.save_primitive(a)
            repo.save_primitive(b)
            repo.save_primitive(c)

            a.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=b.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(a)

            b.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.CONSTRAINED_BY,
                    target_id=c.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(b)

            # C -> A would create A->B->C->A cycle
            c.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.DEPENDS_ON,
                    target_id=a.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            with pytest.raises(CyclicRelationshipError):
                repo.save_primitive(c)

    def test_non_transitive_allows_cycle(self) -> None:
        """APPLIES_TO and INCLUDES should not trigger cycle detection."""
        with SQLiteRepository(":memory:") as repo:
            a = _make_primitive("A", DepthLevel.IDENTITY)
            b = _make_primitive("B", DepthLevel.IDENTITY)
            repo.save_primitive(a)
            repo.save_primitive(b)

            a.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.APPLIES_TO,
                    target_id=b.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(a)

            b.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.APPLIES_TO,
                    target_id=a.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            # Should not raise
            repo.save_primitive(b)

    def test_edge_replacement_allows_formerly_cyclic(self) -> None:
        """After replacing A->B with A->C, B->A should no longer cycle."""
        with SQLiteRepository(":memory:") as repo:
            a = _make_primitive("A", DepthLevel.IDENTITY)
            b = _make_primitive("B", DepthLevel.IDENTITY)
            c = _make_primitive("C", DepthLevel.IDENTITY)
            repo.save_primitive(a)
            repo.save_primitive(b)
            repo.save_primitive(c)

            # A -> B
            a.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=b.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(a)

            # Replace A->B with A->C
            a.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=c.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(a)

            # Now B -> A should be fine
            b.depths[1].relata = [
                Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=a.id,
                    target_depth=DepthLevel.EXISTENCE,
                    provenance=_prov(),
                )
            ]
            repo.save_primitive(b)  # Should not raise


# ------------------------------------------------------------------
# TestResolveSubgraph
# ------------------------------------------------------------------


class TestResolveSubgraph:
    def test_empty_names(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            result = repo.resolve_subgraph([])
            assert result.roots == []
            assert result.nodes == []
            assert result.edges == []

    def test_unknown_name(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            result = repo.resolve_subgraph(["nonexistent"])
            assert result.roots == []
            assert result.nodes == []
            assert result.edges == []

    def test_single_root_no_edges(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("Standalone", DepthLevel.CAPABILITIES)
            repo.save_primitive(prim)
            result = repo.resolve_subgraph(["Standalone"])
            assert len(result.roots) == 1
            assert result.roots[0].id == prim.id
            assert len(result.nodes) == 1
            assert result.edges == []

    def test_case_insensitive_resolution(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prim = _make_primitive("File", DepthLevel.EXISTENCE)
            repo.save_primitive(prim)
            result = repo.resolve_subgraph(["file"])
            assert len(result.roots) == 1
            assert result.roots[0].name == "File"

    def test_transitive_traversal(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            # directory -> (REQUIRES) path -> (DEPENDS_ON) filesystem -> (DEPENDS_ON) os
            result = repo.resolve_subgraph(["directory"])
            node_names = {n.name for n in result.nodes}
            assert "directory" in node_names
            assert "path" in node_names
            assert "filesystem" in node_names
            assert "operating_system" in node_names

    def test_non_transitive_edges_not_traversed(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            # list has APPLIES_TO directory. Starting from list, should NOT
            # traverse into directory's transitive subgraph.
            result = repo.resolve_subgraph(["list"])
            node_names = {n.name for n in result.nodes}
            assert "list" in node_names
            # APPLIES_TO is non-transitive, so directory should not be reached
            assert "directory" not in node_names

    def test_non_transitive_edges_included_between_reachable_nodes(self) -> None:
        """Even though APPLIES_TO is not traversed, if both ends are in the subgraph,
        the edge should be included."""
        with SQLiteRepository(":memory:") as repo:
            prims = _seed_filesystem_graph(repo)
            # Resolve both list and directory
            result = repo.resolve_subgraph(["list", "directory"])
            node_names = {n.name for n in result.nodes}
            assert "list" in node_names
            assert "directory" in node_names
            # The APPLIES_TO edge between list and directory should be present
            applies_to_edges = [
                e
                for e in result.edges
                if e.relation_type == RelationType.APPLIES_TO
                and e.source_id == prims["list"].id
                and e.target_id == prims["directory"].id
            ]
            assert len(applies_to_edges) == 1

    def test_roots_are_only_matched_names(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            result = repo.resolve_subgraph(["directory"])
            root_names = {r.name for r in result.roots}
            assert root_names == {"directory"}
            # path, filesystem, os should be in nodes but not roots
            node_names = {n.name for n in result.nodes}
            assert "path" in node_names

    def test_multiple_roots(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            result = repo.resolve_subgraph(["file", "directory"])
            root_names = {r.name for r in result.roots}
            assert root_names == {"file", "directory"}

    def test_hydrated_nodes_have_relata(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            result = repo.resolve_subgraph(["directory"])
            # Find directory node
            dir_nodes = [n for n in result.nodes if n.name == "directory"]
            assert len(dir_nodes) == 1
            dir_node = dir_nodes[0]
            # directory has relata on D1 (REQUIRES path) and D2 (DEPENDS_ON filesystem)
            all_relata = [r for d in dir_node.depths for r in d.relata]
            assert len(all_relata) >= 2

    def test_all_edges_between_nodes(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            prims = _seed_filesystem_graph(repo)
            result = repo.resolve_subgraph(["directory"])
            # Should have edges: directory->path (REQUIRES), directory->filesystem (DEPENDS_ON),
            # path->filesystem (DEPENDS_ON), filesystem->os (DEPENDS_ON)
            edge_pairs = {(e.source_id, e.target_id) for e in result.edges}
            assert (prims["directory"].id, prims["path"].id) in edge_pairs
            assert (prims["directory"].id, prims["filesystem"].id) in edge_pairs
            assert (prims["path"].id, prims["filesystem"].id) in edge_pairs
            assert (prims["filesystem"].id, prims["operating_system"].id) in edge_pairs

    def test_edges_are_epistemic_steps(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            result = repo.resolve_subgraph(["directory"])
            for edge in result.edges:
                assert isinstance(edge, EpistemicStep)
                assert isinstance(edge.relation_type, RelationType)
                assert isinstance(edge.source_depth, DepthLevel)
                assert isinstance(edge.target_depth, DepthLevel)

    def test_anchor_uses_name_index_not_table_scan(self) -> None:
        """The resolve_subgraph CTE anchor must hit idx_primitives_name_lower
        rather than full-scanning primitives (issue #82). Mirrors the production
        anchor in resolve_subgraph: a LOWER(name) wrapper would force a SCAN."""
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            plan = repo._conn.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT id FROM primitives WHERE name COLLATE NOCASE IN (?, ?)",
                ["file", "directory"],
            ).fetchall()
            detail = " ".join(row["detail"] for row in plan)
            assert "idx_primitives_name_lower" in detail
            assert "SCAN primitives" not in detail


# ------------------------------------------------------------------
# TestUpsertPrimitive
# ------------------------------------------------------------------


class TestUpsertPrimitive:
    def test_upsert_creates_new(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            p = _make_primitive("file")
            result = repo.upsert_primitive(p)
            assert result.id == p.id
            assert repo.find_by_name("file") is not None

    def test_upsert_preserves_existing_id(self) -> None:
        with SQLiteRepository(":memory:") as repo:
            existing = _make_primitive("file")
            repo.save_primitive(existing)
            incoming = _make_primitive("file")
            result = repo.upsert_primitive(incoming)
            assert result.id == existing.id
            assert result.id != incoming.id


# ------------------------------------------------------------------
# TestGroundingEngineWithSQLite
# ------------------------------------------------------------------


class TestGroundingEngineWithSQLite:
    def test_clean_pass(self) -> None:
        """list + directory with full grounding -> no gaps."""
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            engine = GroundingEngine(repo)
            response = engine.query(["list", "directory"])
            assert response.result.gaps == []

    def test_existence_gap(self) -> None:
        """Unknown concept -> ExistenceGap."""
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            engine = GroundingEngine(repo)
            response = engine.query(["unknown_concept"])
            assert len(response.result.gaps) == 1
            assert response.result.gaps[0].kind == "EXISTENCE"

    def test_relational_gap(self) -> None:
        """read targets file at D2 but file is only D1 -> RelationalGap."""
        with SQLiteRepository(":memory:") as repo:
            _seed_filesystem_graph(repo)
            engine = GroundingEngine(repo)
            response = engine.query(["read", "file"])
            gap_kinds = [g.kind for g in response.result.gaps]
            assert "RELATIONAL" in gap_kinds


# ------------------------------------------------------------------
# TestExports
# ------------------------------------------------------------------


class TestExports:
    def test_importable_from_backends(self) -> None:
        from vre.core.backends import SQLiteRepository as S
        assert S is SQLiteRepository

    def test_importable_from_vre(self) -> None:
        from vre import SQLiteRepository as S
        assert S is SQLiteRepository


# ------------------------------------------------------------------
# Legacy schema tolerance (#81 clean break)
# ------------------------------------------------------------------


def test_legacy_policies_column_warns(tmp_path, caplog):
    """A pre-#81 DB with stored policies opens fine and warns that they are now inert."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE relata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT, target_id TEXT, rel_type TEXT,
            source_depth INTEGER, target_depth INTEGER,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            policies TEXT NOT NULL DEFAULT '[]',
            provenance TEXT
        );
        INSERT INTO relata (source_id, target_id, rel_type, source_depth, target_depth, policies)
        VALUES ('s', 't', 'APPLIES_TO', 2, 3, '[{"name": "old_policy"}]');
        """
    )
    conn.close()

    with caplog.at_level("WARNING"):
        repo = SQLiteRepository(str(db))
    repo.close()
    assert any("Legacy graph-resident policies" in r.message for r in caplog.records)
