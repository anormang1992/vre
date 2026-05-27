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
        depths_json     TEXT NOT NULL,
        provenance      TEXT,
        metrics_json    TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_primitives_name_lower
        ON primitives(name COLLATE NOCASE);

    CREATE TABLE IF NOT EXISTS relationships (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id     TEXT NOT NULL REFERENCES primitives(id),
        target_id     TEXT NOT NULL REFERENCES primitives(id),
        rel_type      TEXT NOT NULL,
        source_depth  INTEGER NOT NULL,
        target_depth  INTEGER NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        policies      TEXT NOT NULL DEFAULT '[]',
        provenance    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
    CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
    CREATE INDEX IF NOT EXISTS idx_rel_type   ON relationships(rel_type);
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
from vre.core.policy.models import parse_policy


logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".vre" / "graph.db"

_TRANSITIVE_RELS = [rt.value for rt in TRANSITIVE_RELATION_TYPES]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS primitives (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    depths_json     TEXT NOT NULL,
    provenance      TEXT,
    metrics_json    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_primitives_name_lower
    ON primitives(name COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS relationships (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL REFERENCES primitives(id),
    target_id     TEXT NOT NULL REFERENCES primitives(id),
    rel_type      TEXT NOT NULL,
    source_depth  INTEGER NOT NULL,
    target_depth  INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    policies      TEXT NOT NULL DEFAULT '[]',
    provenance    TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type   ON relationships(rel_type);
"""


class SQLiteRepository(Repository):
    """
    SQLite persistence backend for epistemic primitives.

    Stores primitives in a local SQLite database with depths serialized as
    JSON and relata stored in a separate ``relationships`` table. Provides
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
        self._conn = sqlite3.connect(self._path)
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
            }
            if depth.provenance:
                entry["provenance"] = depth.provenance.model_dump(mode="json")
            stripped.append(entry)
        return json.dumps(stripped)

    @staticmethod
    def _dump_model_json(model: Any) -> str | None:
        """Serialize an optional Pydantic model to a JSON string, or None when absent."""
        return json.dumps(model.model_dump(mode="json")) if model is not None else None

    @staticmethod
    def _hydrate_primitive(
        node_data: dict[str, Any],
        relationships: list[dict[str, Any]],
    ) -> Primitive:
        """Reconstruct a Primitive from raw SQLite row data and its relationship records."""
        try:
            raw_depths = json.loads(node_data["depths_json"])
            depths_by_level: dict[int, Depth] = {}
            for rd in raw_depths:
                depth = Depth(
                    level=DepthLevel(rd["level"]),
                    properties=rd.get("properties", {}),
                    provenance=Provenance(**rd["provenance"]) if rd.get("provenance") else None,
                )
                depths_by_level[int(depth.level)] = depth

            for rel in relationships:
                source_depth = rel["source_depth"]
                target_depth_val = rel["target_depth"]
                metadata_raw = rel.get("metadata_json", "{}")
                metadata = json.loads(metadata_raw) if metadata_raw else {}

                policies_raw = rel.get("policies", "[]")
                policies_data = json.loads(policies_raw) if policies_raw else []
                policies = [parse_policy(p) for p in policies_data]

                rel_prov_raw = rel.get("provenance")
                rel_prov = None
                if rel_prov_raw:
                    rel_prov = Provenance(**json.loads(rel_prov_raw))

                relatum = Relatum(
                    relation_type=RelationType(rel["rel_type"]),
                    target_id=UUID(rel["target_id"]),
                    target_depth=DepthLevel(target_depth_val),
                    metadata=metadata,
                    policies=policies,
                    provenance=rel_prov,
                )

                if source_depth is not None and source_depth in depths_by_level:
                    depths_by_level[source_depth].relata.append(relatum)

            sorted_depths = sorted(depths_by_level.values(), key=lambda d: int(d.level))

            node_prov_raw = node_data.get("provenance")
            node_prov = None
            if node_prov_raw:
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
        For each new transitive relatum, checks whether the target can
        already reach the source via transitive edges. Raises
        CyclicRelationshipError on the first cycle found.
        """
        placeholders = ",".join("?" for _ in _TRANSITIVE_RELS)

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

            row = cursor.execute(
                f"""
                WITH RECURSIVE reachable(id) AS (
                    SELECT ?
                    UNION
                    SELECT r.target_id
                    FROM relationships r
                    INNER JOIN reachable rc ON r.source_id = rc.id
                    WHERE r.rel_type IN ({placeholders})
                )
                SELECT EXISTS(SELECT 1 FROM reachable WHERE id = ?) AS would_cycle
                """,
                [rp["target_id"]] + _TRANSITIVE_RELS + [source_id],
            ).fetchone()

            if row and row["would_cycle"]:
                logger.warning(
                    "Cycle detected: %s from %s to %s would create a cycle",
                    rp["rel_type"],
                    source_id,
                    rp["target_id"],
                )
                raise CyclicRelationshipError(
                    f"{rp['rel_type']} from {source_id} to "
                    f"{rp['target_id']} would create a cycle"
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
                        "policies": json.dumps(
                            [p.model_dump() for p in relatum.policies]
                        ) if relatum.policies else "[]",
                        "provenance": self._dump_model_json(relatum.provenance),
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
                INSERT INTO primitives (id, name, depths_json, provenance, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    depths_json = excluded.depths_json,
                    provenance = excluded.provenance,
                    metrics_json = excluded.metrics_json
                """,
                (
                    str(primitive.id),
                    primitive.name,
                    depths_json,
                    node_provenance,
                    node_metrics,
                ),
            )

            # Delete old outgoing relationships
            cursor.execute(
                "DELETE FROM relationships WHERE source_id = ?",
                (str(primitive.id),),
            )

            # Check for transitive cycles
            self._check_transitive_cycles(cursor, str(primitive.id), relata_params)

            # Insert new relationships
            for rp in relata_params:
                cursor.execute(
                    """
                    INSERT INTO relationships
                        (source_id, target_id, rel_type, source_depth,
                         target_depth, metadata_json, policies, provenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(primitive.id),
                        rp["target_id"],
                        rp["rel_type"],
                        rp["source_depth"],
                        rp["target_depth"],
                        rp["metadata_json"],
                        rp["policies"],
                        rp["provenance"],
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
            "SELECT id, name, depths_json, provenance, metrics_json "
            "FROM primitives WHERE id = ?",
            (str(id),),
        ).fetchone()

        if row is None:
            logger.debug("Primitive not found by id=%s", id)
            return None

        node_data = dict(row)
        relationships = self._fetch_outgoing_relationships(str(id))
        return self._hydrate_primitive(node_data, relationships)

    def find_by_name(self, name: str) -> Primitive | None:
        """Look up a primitive by name (case-insensitive). Returns None if not found."""
        row = self._conn.execute(
            "SELECT id, name, depths_json, provenance, metrics_json "
            "FROM primitives WHERE name = ? COLLATE NOCASE",
            (name,),
        ).fetchone()

        if row is None:
            logger.debug("Primitive not found by name=%r", name)
            return None

        node_data = dict(row)
        relationships = self._fetch_outgoing_relationships(node_data["id"])
        return self._hydrate_primitive(node_data, relationships)

    def _fetch_outgoing_relationships(self, primitive_id: str) -> list[dict[str, Any]]:
        """Fetch all outgoing relationships for a primitive as dicts."""
        rows = self._conn.execute(
            """
            SELECT rel_type, target_id, source_depth, target_depth,
                   metadata_json, policies, provenance
            FROM relationships
            WHERE source_id = ?
            """,
            (primitive_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_names(self) -> list[str]:
        """Return the names of all primitives, sorted alphabetically."""
        rows = self._conn.execute(
            "SELECT name FROM primitives ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def delete_primitive(self, id: UUID) -> bool:
        """Delete the primitive with the given UUID and all its relationships. Returns True if deleted."""
        sid = str(id)
        # Delete relationships first (both directions)
        self._conn.execute(
            "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
            (sid, sid),
        )
        cursor = self._conn.execute(
            "DELETE FROM primitives WHERE id = ?",
            (sid,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """Delete every primitive and its relationships. Returns the count deleted."""
        self._conn.execute("DELETE FROM relationships")
        cursor = self._conn.execute("DELETE FROM primitives")
        self._conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Abstract method implementations: metrics
    # ------------------------------------------------------------------

    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        """Update only the metrics JSON on an existing primitive."""
        metrics_json = json.dumps(metrics.model_dump(mode="json"))
        try:
            self._conn.execute(
                "UPDATE primitives SET metrics_json = ? WHERE id = ?",
                (metrics_json, str(primitive_id)),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
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

        lowered = [n.lower() for n in names]
        name_placeholders = ",".join("?" for _ in lowered)
        transitive_placeholders = ",".join("?" for _ in _TRANSITIVE_RELS)

        # Phase 1: Find all transitively reachable node IDs from roots
        reachable_ids_rows = self._conn.execute(
            f"""
            WITH RECURSIVE reachable(id) AS (
                SELECT id FROM primitives WHERE LOWER(name) IN ({name_placeholders})
                UNION
                SELECT r.target_id
                FROM relationships r
                INNER JOIN reachable rc ON r.source_id = rc.id
                WHERE r.rel_type IN ({transitive_placeholders})
            )
            SELECT id FROM reachable
            """,
            lowered + _TRANSITIVE_RELS,
        ).fetchall()

        reachable_ids = [row["id"] for row in reachable_ids_rows]

        if not reachable_ids:
            return ResolvedSubgraph(roots=[], nodes=[], edges=[])

        # Phase 2: Fetch all node rows
        node_placeholders = ",".join("?" for _ in reachable_ids)
        node_rows = self._conn.execute(
            f"""
            SELECT id, name, depths_json, provenance, metrics_json
            FROM primitives WHERE id IN ({node_placeholders})
            """,
            reachable_ids,
        ).fetchall()

        # Phase 3: Fetch all edges between reachable nodes
        edge_rows = self._conn.execute(
            f"""
            SELECT source_id, target_id, rel_type, source_depth, target_depth,
                   metadata_json, policies, provenance
            FROM relationships
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

        # Identify root names (lowered) for root filtering
        lowered_set = set(lowered)

        # Hydrate all nodes
        nodes: list[Primitive] = []
        roots: list[Primitive] = []
        for nr in node_rows:
            nd = dict(nr)
            prim = self._hydrate_primitive(nd, edges_by_source.get(nd["id"], []))
            nodes.append(prim)
            if prim.name.lower() in lowered_set:
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
