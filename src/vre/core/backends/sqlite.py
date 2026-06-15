# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
SQLite persistence backend for the Volute Reasoning Engine.

Provides SQLiteRepository -- a zero-external-dependency backend that
implements the Repository ABC using a local SQLite database.

Schema
------
::

    CREATE TABLE IF NOT EXISTS primitives (
        id              TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        name_lower      TEXT NOT NULL,
        depths_json     TEXT NOT NULL,
        provenance_json TEXT,
        metrics_json    TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_primitives_name_lower
        ON primitives(name_lower);

    CREATE TABLE IF NOT EXISTS relata (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id     TEXT NOT NULL REFERENCES primitives(id) ON DELETE CASCADE,
        target_id     TEXT NOT NULL REFERENCES primitives(id) ON DELETE CASCADE,
        rel_type      TEXT NOT NULL,
        source_depth  INTEGER NOT NULL,
        target_depth  INTEGER NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        provenance_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_rel_source ON relata(source_id);
    CREATE INDEX IF NOT EXISTS idx_rel_target ON relata(target_id);
    CREATE INDEX IF NOT EXISTS idx_rel_type   ON relata(rel_type);
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from vre.core.backends.repository import Repository
from vre.core.errors import (
    CyclicRelationshipError,
    HydrationError,
    PersistenceError,
)
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    PrimitiveMetrics,
    Provenance,
    Relatum,
    RelationType,
    ResolvedSubgraph,
    TRANSITIVE_RELATION_TYPES,
)


logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".vre" / "graph.db"

_TRANSITIVE_RELS = [rt.value for rt in TRANSITIVE_RELATION_TYPES]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS primitives (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    name_lower      TEXT NOT NULL,
    depths_json     TEXT NOT NULL,
    provenance_json TEXT,
    metrics_json    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_primitives_name_lower
    ON primitives(name_lower);

CREATE TABLE IF NOT EXISTS relata (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL REFERENCES primitives(id) ON DELETE CASCADE,
    target_id     TEXT NOT NULL REFERENCES primitives(id) ON DELETE CASCADE,
    rel_type      TEXT NOT NULL,
    source_depth  INTEGER NOT NULL,
    target_depth  INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relata(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relata(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type   ON relata(rel_type);
"""


class SQLiteRepository(Repository):
    """
    SQLite persistence backend for epistemic primitives.

    Stores primitives in a local SQLite database with depths serialized as
    JSON and relata stored in a separate ``relata`` table. Provides
    transitive cycle detection via recursive CTEs and subgraph resolution
    for the grounding engine.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = _DEFAULT_PATH
        resolved = Path(path)
        if str(resolved) != ":memory:":
            resolved.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(resolved)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _depths_to_json(depths: list[Depth]) -> str:
        """Serialize depth levels and their properties to a JSON string."""
        stripped = []
        for depth in depths:
            entry: dict[str, Any] = {
                "level": int(depth.level),
                "properties": depth.properties,
                "provenance": depth.provenance.model_dump(mode="json"),
            }
            stripped.append(entry)
        return json.dumps(stripped)

    @staticmethod
    def _dump_model_json(model: Any) -> str | None:
        """Serialize an optional Pydantic model to a JSON string, or None when absent."""
        return json.dumps(model.model_dump(mode="json")) if model is not None else None

    @staticmethod
    def _hydrate_primitive(
        node_data: dict[str, Any],
        relata: list[dict[str, Any]],
    ) -> Primitive:
        """Reconstruct a Primitive from raw SQLite row data and its relata records."""
        try:
            raw_depths = json.loads(node_data["depths_json"])
            depths_by_level: dict[int, Depth] = {}
            name = node_data["name"]
            for rd in raw_depths:
                if not rd.get("provenance"):
                    raise HydrationError(
                        f"Depth D{rd['level']} of '{name}' is missing provenance — "
                        f"provenance is required (no legacy/unstamped graphs)"
                    )
                depth = Depth(
                    level=DepthLevel(rd["level"]),
                    properties=rd.get("properties", {}),
                    provenance=Provenance(**rd["provenance"]),
                )
                depths_by_level[int(depth.level)] = depth

            for rel in relata:
                source_depth = rel["source_depth"]
                target_depth_val = rel["target_depth"]
                metadata_raw = rel.get("metadata_json", "{}")
                metadata = json.loads(metadata_raw) if metadata_raw else {}

                rel_prov_raw = rel.get("provenance_json")
                if not rel_prov_raw:
                    raise HydrationError(
                        f"Relatum {rel['rel_type']} on '{name}' is missing provenance — "
                        f"provenance is required (no legacy/unstamped graphs)"
                    )
                rel_prov = Provenance(**json.loads(rel_prov_raw))

                relatum = Relatum(
                    relation_type=RelationType(rel["rel_type"]),
                    target_id=UUID(rel["target_id"]),
                    target_depth=DepthLevel(target_depth_val),
                    metadata=metadata,
                    provenance=rel_prov,
                )

                if source_depth is not None and source_depth in depths_by_level:
                    depths_by_level[source_depth].relata.append(relatum)

            sorted_depths = sorted(depths_by_level.values(), key=lambda d: int(d.level))

            node_prov_raw = node_data.get("provenance_json")
            if not node_prov_raw:
                raise HydrationError(
                    f"Primitive '{name}' is missing provenance — "
                    f"provenance is required (no legacy/unstamped graphs)"
                )
            node_prov = Provenance(**json.loads(node_prov_raw))

            node_metrics_raw = node_data.get("metrics_json")
            node_metrics = None
            if node_metrics_raw:
                node_metrics = PrimitiveMetrics(**json.loads(node_metrics_raw))

            return Primitive(
                id=UUID(node_data["id"]),
                name=node_data["name"],
                depths=sorted_depths,
                provenance=node_prov,
                metrics=node_metrics,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Hydration failed for primitive %r: %s", node_data.get("name", "?"), exc)
            raise HydrationError(
                f"Failed to hydrate primitive '{node_data.get('name', '?')}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def _check_transitive_cycles(
        self,
        cursor: sqlite3.Cursor,
        source_id: str,
        relata_params: list[dict[str, Any]],
    ) -> None:
        """
        Verify that no new transitive edge would create a cycle.

        Called inside a transaction after old edges have been deleted.
        Seeds a single recursive CTE with every proposed transitive target at
        once -- carrying each target along as ``origin`` -- and asks which
        origins can reach ``source_id`` via existing transitive edges. Any such
        origin would close a cycle (source -> target -> ... -> source). This
        replaces a per-edge query loop with one round-trip per save. Raises
        CyclicRelationshipError on the first offending edge.
        """
        # Map each distinct transitive target to a rel_type for diagnostics, and
        # short-circuit self-references (which a recursive walk would also catch,
        # but this gives a clearer message and avoids the query entirely).
        target_rel_type: dict[str, str] = {}
        for rp in relata_params:
            if rp["rel_type"] not in _TRANSITIVE_RELS:
                continue

            if rp["target_id"] == source_id:
                logger.warning(
                    "Self-referential %s detected on primitive %s",
                    rp["rel_type"],
                    source_id,
                )
                raise CyclicRelationshipError(
                    f"Self-referential {rp['rel_type']} on {source_id}"
                )

            target_rel_type.setdefault(rp["target_id"], rp["rel_type"])

        # With no transitive targets there is nothing reachable to close a
        # cycle, so the query is skipped and the method falls through.
        if target_rel_type:
            targets = list(target_rel_type)
            seed_placeholders = ",".join("(?)" for _ in targets)
            rel_placeholders = ",".join("?" for _ in _TRANSITIVE_RELS)

            rows = cursor.execute(
                f"""
                WITH RECURSIVE reachable(origin, id) AS (
                    -- seed each proposed target as both its own origin and start
                    -- node (column1 is SQLite's auto-name for the VALUES column)
                    SELECT column1, column1 FROM (VALUES {seed_placeholders})
                    UNION
                    SELECT rc.origin, r.target_id
                    FROM relata r
                    INNER JOIN reachable rc ON r.source_id = rc.id
                    WHERE r.rel_type IN ({rel_placeholders})
                )
                SELECT DISTINCT origin FROM reachable WHERE id = ?
                """,
                targets + _TRANSITIVE_RELS + [source_id],
            ).fetchall()

            if rows:
                offending = rows[0]["origin"]
                rel_type = target_rel_type[offending]
                logger.warning(
                    "Cycle detected: %s from %s to %s would create a cycle",
                    rel_type,
                    source_id,
                    offending,
                )
                raise CyclicRelationshipError(
                    f"{rel_type} from {source_id} to {offending} would create a cycle"
                )

    # ------------------------------------------------------------------
    # Abstract method implementations: CRUD
    # ------------------------------------------------------------------

    def save_primitive(self, primitive: Primitive) -> None:
        """Persist a Primitive -- full replace of depths and relata."""
        primitive.validate_provenance()
        depths_json = self._depths_to_json(primitive.depths)

        relata_params: list[dict[str, Any]] = []
        for depth in primitive.depths:
            for relatum in depth.relata:
                relata_params.append(
                    {
                        "target_id": str(relatum.target_id),
                        "rel_type": relatum.relation_type.value,
                        "source_depth": int(depth.level),
                        "target_depth": int(relatum.target_depth),
                        "metadata_json": json.dumps(relatum.metadata) if relatum.metadata else "{}",
                        "provenance_json": self._dump_model_json(relatum.provenance),
                    }
                )

        node_provenance = self._dump_model_json(primitive.provenance)
        node_metrics = self._dump_model_json(primitive.metrics)

        logger.debug(
            "Saving primitive %r (id=%s, depths=%d, relata=%d)",
            primitive.name,
            primitive.id,
            len(primitive.depths),
            len(relata_params),
        )

        try:
            cursor = self._conn.cursor()
            cursor.execute("BEGIN")

            # UPSERT primitive row
            cursor.execute(
                """
                INSERT INTO primitives (id, name, name_lower, depths_json, provenance_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    name_lower = excluded.name_lower,
                    depths_json = excluded.depths_json,
                    provenance_json = excluded.provenance_json,
                    metrics_json = excluded.metrics_json
                """,
                (
                    str(primitive.id),
                    primitive.name,
                    primitive.name_lower,
                    depths_json,
                    node_provenance,
                    node_metrics,
                ),
            )

            # Delete old outgoing relata
            cursor.execute(
                "DELETE FROM relata WHERE source_id = ?",
                (str(primitive.id),),
            )

            # Check for transitive cycles
            self._check_transitive_cycles(cursor, str(primitive.id), relata_params)

            # Insert new relata
            for rp in relata_params:
                cursor.execute(
                    """
                    INSERT INTO relata
                        (source_id, target_id, rel_type, source_depth,
                         target_depth, metadata_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(primitive.id),
                        rp["target_id"],
                        rp["rel_type"],
                        rp["source_depth"],
                        rp["target_depth"],
                        rp["metadata_json"],
                        rp["provenance_json"],
                    ),
                )

            self._conn.commit()
        except CyclicRelationshipError:
            self._conn.rollback()
            raise
        except sqlite3.Error as exc:
            self._conn.rollback()
            logger.error("SQLite error saving primitive %r: %s", primitive.name, exc)
            raise PersistenceError(
                f"Failed to save primitive '{primitive.name}': {exc}"
            ) from exc

    def find_by_id(self, id: UUID) -> Primitive | None:
        """Look up a primitive by UUID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT id, name, depths_json, provenance_json, metrics_json "
            "FROM primitives WHERE id = ?",
            (str(id),),
        ).fetchone()

        if row is None:
            logger.debug("Primitive not found by id=%s", id)
            return None

        node_data = dict(row)
        relata = self._fetch_outgoing_relata(str(id))
        return self._hydrate_primitive(node_data, relata)

    def find_by_name(self, name: str) -> Primitive | None:
        """Look up a primitive by name (case-insensitive). Returns None if not found."""
        row = self._conn.execute(
            "SELECT id, name, depths_json, provenance_json, metrics_json "
            "FROM primitives WHERE name_lower = ?",
            (Primitive.fold_name(name),),
        ).fetchone()

        if row is None:
            logger.debug("Primitive not found by name=%r", name)
            return None

        node_data = dict(row)
        relata = self._fetch_outgoing_relata(node_data["id"])
        return self._hydrate_primitive(node_data, relata)

    def _fetch_outgoing_relata(self, primitive_id: str) -> list[dict[str, Any]]:
        """Fetch all outgoing relata for a primitive as dicts."""
        rows = self._conn.execute(
            """
            SELECT rel_type, target_id, source_depth, target_depth,
                   metadata_json, provenance_json
            FROM relata
            WHERE source_id = ?
            """,
            (primitive_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_names(self) -> list[str]:
        """Return the names of all primitives, sorted alphabetically."""
        rows = self._conn.execute(
            "SELECT name FROM primitives ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [r["name"] for r in rows]

    def delete_primitive(self, id: UUID) -> bool:
        """Delete the primitive with the given UUID. Relata are cascade-deleted by the FK constraint."""
        try:
            cursor = self._conn.execute(
                "DELETE FROM primitives WHERE id = ?",
                (str(id),),
            )
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise PersistenceError(
                f"Failed to delete primitive '{id}': {exc}"
            ) from exc

    def clear(self) -> int:
        """Delete every primitive. Relata are cascade-deleted by the FK constraint."""
        try:
            cursor = self._conn.execute("DELETE FROM primitives")
            return cursor.rowcount
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to clear graph: {exc}") from exc

    # ------------------------------------------------------------------
    # Abstract method implementations: metrics
    # ------------------------------------------------------------------

    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        """Update only the metrics JSON on an existing primitive."""
        metrics_json = json.dumps(metrics.model_dump(mode="json"))
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE primitives SET metrics_json = ? WHERE id = ?",
                (metrics_json, str(primitive_id)),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            logger.error("Failed to update metrics for primitive %s: %s", primitive_id, exc)
            raise PersistenceError(
                f"Failed to update metrics for '{primitive_id}': {exc}"
            ) from exc

    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]:
        """Read current metrics for multiple primitives in a single query."""
        result: dict[UUID, PrimitiveMetrics | None] = {}
        if not primitive_ids:
            return result

        placeholders = ",".join("?" for _ in primitive_ids)
        try:
            rows = self._conn.execute(
                f"SELECT id, metrics_json FROM primitives WHERE id IN ({placeholders})",
                [str(pid) for pid in primitive_ids],
            ).fetchall()

            for row in rows:
                pid = UUID(row["id"])
                raw = row["metrics_json"]
                if raw:
                    try:
                        result[pid] = PrimitiveMetrics(**json.loads(raw))
                    except Exception as exc:
                        raise HydrationError(
                            f"Failed to hydrate metrics for primitive '{pid}': {exc}"
                        ) from exc
                else:
                    result[pid] = None
        except HydrationError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to batch-read metrics: {exc}") from exc

        return result

    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None:
        """Persist metrics for multiple primitives in a single batch."""
        if not updates:
            return

        try:
            cursor = self._conn.cursor()
            cursor.execute("BEGIN")
            for pid, metrics in updates.items():
                metrics_json = json.dumps(metrics.model_dump(mode="json"))
                cursor.execute(
                    "UPDATE primitives SET metrics_json = ? WHERE id = ?",
                    (metrics_json, str(pid)),
                )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise PersistenceError(f"Failed to batch-update metrics: {exc}") from exc

    # ------------------------------------------------------------------
    # Abstract method implementations: graph traversal
    # ------------------------------------------------------------------

    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        """Single-query traversal that resolves a subgraph for the given concept names."""
        logger.debug("Resolving subgraph for names=%s", names)

        if not names:
            return ResolvedSubgraph(roots=[], nodes=[], edges=[])

        name_placeholders = ",".join("?" for _ in names)
        transitive_placeholders = ",".join("?" for _ in _TRANSITIVE_RELS)

        # Phase 1: Find all transitively reachable node IDs from roots.
        # The anchor matches the stored fold key (Primitive.name_lower) so the
        # unique idx_primitives_name_lower index can satisfy it, and so that
        # "case-insensitive" means exactly Primitive.fold_name everywhere.
        reachable_ids_rows = self._conn.execute(
            f"""
            WITH RECURSIVE reachable(id) AS (
                SELECT id FROM primitives WHERE name_lower IN ({name_placeholders})
                UNION
                SELECT r.target_id
                FROM relata r
                INNER JOIN reachable rc ON r.source_id = rc.id
                WHERE r.rel_type IN ({transitive_placeholders})
            )
            SELECT id FROM reachable
            """,
            [Primitive.fold_name(n) for n in names] + _TRANSITIVE_RELS,
        ).fetchall()

        reachable_ids = [row["id"] for row in reachable_ids_rows]

        if not reachable_ids:
            return ResolvedSubgraph(roots=[], nodes=[], edges=[])

        # Phase 2: Fetch all node rows
        node_placeholders = ",".join("?" for _ in reachable_ids)
        node_rows = self._conn.execute(
            f"""
            SELECT id, name, depths_json, provenance_json, metrics_json
            FROM primitives WHERE id IN ({node_placeholders})
            """,
            reachable_ids,
        ).fetchall()

        # Phase 3: Fetch all edges between reachable nodes
        edge_rows = self._conn.execute(
            f"""
            SELECT source_id, target_id, rel_type, source_depth, target_depth,
                   metadata_json, provenance_json
            FROM relata
            WHERE source_id IN ({node_placeholders})
              AND target_id IN ({node_placeholders})
            """,
            reachable_ids + reachable_ids,
        ).fetchall()

        # Build edges-by-source for hydration
        edges_by_source: dict[str, list[dict[str, Any]]] = {}
        for e in edge_rows:
            ed = dict(e)
            sid = ed["source_id"]
            edges_by_source.setdefault(sid, []).append(ed)

        # Identify root names (case-insensitive) for root filtering
        lowered_set = {Primitive.fold_name(n) for n in names}

        # Hydrate all nodes
        nodes: list[Primitive] = []
        roots: list[Primitive] = []
        for nr in node_rows:
            nd = dict(nr)
            prim = self._hydrate_primitive(nd, edges_by_source.get(nd["id"], []))
            nodes.append(prim)
            if prim.name_lower in lowered_set:
                roots.append(prim)

        # Build EpistemicStep edges
        edges: list[EpistemicStep] = []
        for e in edge_rows:
            edges.append(
                EpistemicStep(
                    source_id=UUID(e["source_id"]),
                    target_id=UUID(e["target_id"]),
                    relation_type=RelationType(e["rel_type"]),
                    source_depth=DepthLevel(e["source_depth"]),
                    target_depth=DepthLevel(e["target_depth"]),
                )
            )

        logger.debug(
            "Subgraph resolved: %d roots, %d nodes, %d edges",
            len(roots),
            len(nodes),
            len(edges),
        )
        return ResolvedSubgraph(roots=roots, nodes=nodes, edges=edges)
