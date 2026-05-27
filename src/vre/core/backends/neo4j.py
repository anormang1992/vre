# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Neo4j persistence backend for the Volute Reasoning Engine.

Provides Neo4jRepository — the bridge between Pydantic epistemic
models and the Neo4j graph database. Primitives are stored as nodes
with embedded depth JSON; relata are stored as typed Neo4j relationships.
"""

import json
import logging
from typing import Any, LiteralString, cast
from uuid import UUID

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from vre.core.backends.repository import Repository
from vre.core.errors import (
    CyclicRelationshipError,
    GraphError,
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


_TRANSITIVE_RELS = [rt.value for rt in TRANSITIVE_RELATION_TYPES]

logger = logging.getLogger(__name__)


class Neo4jRepository(Repository):
    """
    Neo4j persistence backend for epistemic primitives.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_categories=["UNRECOGNIZED"],
        )
        self._database = database
        self._ensure_constraints()

    def close(self) -> None:
        """Close the underlying Neo4j driver and release all connections."""
        self._driver.close()

    def _ensure_constraints(self) -> None:
        logger.debug("Ensuring uniqueness constraint on Primitive.id")
        try:
            with self._driver.session(database=self._database) as session:
                session.run(
                    cast(
                        LiteralString,
                        "CREATE CONSTRAINT primitive_id_unique IF NOT EXISTS "
                        "FOR (p:Primitive) REQUIRE p.id IS UNIQUE",
                    )
                )
        except Neo4jError as exc:
            logger.error("Failed to ensure Neo4j constraints: %s", exc)
            raise GraphError(f"Failed to ensure constraints: {exc}") from exc

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _depths_to_json(depths: list[Depth]) -> str:
        """Serialize depth levels and their properties to a JSON string for Neo4j storage."""
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
    def _parse_json_field(value: Any) -> Any:
        """Parse a Neo4j property that may be a JSON string or already-decoded dict/list."""
        return json.loads(value) if isinstance(value, str) else value

    @staticmethod
    def _dump_model_json(model: Any) -> str | None:
        """Serialize an optional Pydantic model to a JSON string, or None when absent."""
        return json.dumps(model.model_dump(mode="json")) if model is not None else None

    @staticmethod
    def _record_to_node_data(record: Any) -> dict[str, Any]:
        """Extract the node property dict from a Cypher record row."""
        return {
            "id": record["id"],
            "name": record["name"],
            "depths_json": record["depths_json"],
            "provenance": record["provenance"],
            "metrics_json": record["metrics_json"],
        }

    @staticmethod
    def _record_to_relationships(record: Any) -> list[dict[str, Any]]:
        """Extract the relationship list from a Cypher record row."""
        return [
            {
                "rel_type": r["rel_type"],
                "target_id": r["target_id"],
                "rel_props": {
                    "source_depth": r["source_depth"],
                    "target_depth": r["target_depth"],
                    "metadata_json": r.get("metadata_json") or "{}",
                    "policies": r.get("policies") or "[]",
                    "provenance": r.get("provenance"),
                },
            }
            for r in record["rels"]
            if r.get("rel_type") is not None
        ]

    @staticmethod
    def _hydrate_primitive(
        node_data: dict[str, Any],
        relationships: list[dict[str, Any]],
    ) -> Primitive:
        """Reconstruct a Primitive from raw Neo4j node data and its relationship records."""
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
                rel_props = rel.get("rel_props", {})
                source_depth = rel_props.get("source_depth")
                target_depth_val = rel_props.get("target_depth")
                metadata_json = rel_props.get("metadata_json", "{}")
                metadata = json.loads(metadata_json) if metadata_json else {}

                policies = rel_props.get("policies", "[]")
                policies_data = json.loads(policies) if policies else []
                policies = [parse_policy(p) for p in policies_data]

                rel_prov_json = rel_props.get("provenance")
                rel_prov = None
                if rel_prov_json:
                    rel_prov = Provenance(**Neo4jRepository._parse_json_field(rel_prov_json))

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

            node_prov_json = node_data.get("provenance")
            node_prov = None
            if node_prov_json:
                node_prov = Provenance(**Neo4jRepository._parse_json_field(node_prov_json))

            node_metrics_json = node_data.get("metrics_json")
            node_metrics = None
            if node_metrics_json:
                node_metrics = PrimitiveMetrics(**Neo4jRepository._parse_json_field(node_metrics_json))

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

    @staticmethod
    def _check_transitive_cycles(
        tx: Any,
        source_id: str,
        relata_params: list[dict[str, Any]],
        transitive_types: list[str],
    ) -> None:
        """Verify that no new transitive edge would create a cycle."""
        type_union = "|".join(transitive_types)

        for rp in relata_params:
            if rp["relation_type"] not in transitive_types:
                continue

            if rp["target_id"] == source_id:
                logger.warning("Self-referential %s detected on primitive %s", rp["relation_type"], source_id)
                raise CyclicRelationshipError(
                    f"Self-referential {rp['relation_type']} on {source_id}"
                )

            record = tx.run(
                cast(
                    LiteralString,
                    "MATCH (t:Primitive {id: $target_id}) "
                    "MATCH (s:Primitive {id: $source_id}) "
                    f"OPTIONAL MATCH path = (t)-[:{type_union}*1..]->(s) "
                    "RETURN path IS NOT NULL AS would_cycle "
                    "LIMIT 1",
                ),
                source_id=source_id,
                target_id=rp["target_id"],
            ).single()

            if record and record["would_cycle"]:
                logger.warning(
                    "Cycle detected: %s from %s to %s would create a cycle",
                    rp["relation_type"], source_id, rp["target_id"],
                )
                raise CyclicRelationshipError(
                    f"{rp['relation_type']} from {source_id} to "
                    f"{rp['target_id']} would create a cycle"
                )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def save_primitive(self, primitive: Primitive) -> None:
        primitive.validate_provenance()
        depths_json = self._depths_to_json(primitive.depths)

        relata_params: list[dict[str, Any]] = []
        for depth in primitive.depths:
            for relatum in depth.relata:
                relata_params.append(
                    {
                        "target_id": str(relatum.target_id),
                        "relation_type": relatum.relation_type.value,
                        "source_depth": int(depth.level),
                        "target_depth": int(relatum.target_depth),
                        "metadata_json": json.dumps(relatum.metadata) if relatum.metadata else "{}",
                        "policies": json.dumps([p.model_dump() for p in relatum.policies]) if relatum.policies else "[]",
                        "provenance": self._dump_model_json(relatum.provenance),
                    }
                )

        logger.debug(
            "Saving primitive %r (id=%s, depths=%d, relata=%d)",
            primitive.name, primitive.id, len(primitive.depths), len(relata_params),
        )

        rel_types = [rt.value for rt in RelationType]

        node_provenance = self._dump_model_json(primitive.provenance)
        node_metrics = self._dump_model_json(primitive.metrics)

        def _tx(tx: Any) -> None:
            tx.run(
                cast(
                    LiteralString,
                    "MERGE (p:Primitive {id: $id}) "
                    "SET p.name = $name, p.depths_json = $depths_json, "
                    "p.provenance = $provenance, p.metrics_json = $metrics_json",
                ),
                id=str(primitive.id),
                name=primitive.name,
                depths_json=depths_json,
                provenance=node_provenance,
                metrics_json=node_metrics,
            )

            for rt in rel_types:
                tx.run(
                    cast(
                        LiteralString,
                        f"MATCH (p:Primitive {{id: $id}})-[r:{rt}]->() DELETE r",
                    ),
                    id=str(primitive.id),
                )

            Neo4jRepository._check_transitive_cycles(
                tx, str(primitive.id), relata_params, _TRANSITIVE_RELS,
            )

            for rp in relata_params:
                tx.run(
                    cast(
                        LiteralString,
                        f"MATCH (s:Primitive {{id: $source_id}}) "
                        f"MATCH (t:Primitive {{id: $target_id}}) "
                        f"CREATE (s)-[:{rp['relation_type']} {{"
                        f"source_depth: $source_depth, "
                        f"target_depth: $target_depth, "
                        f"metadata_json: $metadata_json, "
                        f"policies: $policies, "
                        f"provenance: $provenance"
                        f"}}]->(t)",
                    ),
                    source_id=str(primitive.id),
                    target_id=rp["target_id"],
                    source_depth=rp["source_depth"],
                    target_depth=rp["target_depth"],
                    metadata_json=rp["metadata_json"],
                    policies=rp["policies"],
                    provenance=rp["provenance"],
                )

        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(_tx)
        except CyclicRelationshipError:
            raise
        except Neo4jError as exc:
            logger.error("Neo4j error saving primitive %r: %s", primitive.name, exc)
            raise PersistenceError(f"Failed to save primitive '{primitive.name}': {exc}") from exc

    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        metrics_json = json.dumps(metrics.model_dump(mode="json"))
        try:
            with self._driver.session(database=self._database) as session:
                session.run(
                    cast(
                        LiteralString,
                        "MATCH (p:Primitive {id: $id}) "
                        "SET p.metrics_json = $metrics_json",
                    ),
                    id=str(primitive_id),
                    metrics_json=metrics_json,
                )
        except Neo4jError as exc:
            logger.error("Failed to update metrics for primitive %s: %s", primitive_id, exc)
            raise PersistenceError(f"Failed to update metrics for '{primitive_id}': {exc}") from exc

    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]:
        result: dict[UUID, PrimitiveMetrics | None] = {}
        if primitive_ids:
            cypher = cast(
                LiteralString,
                "MATCH (p:Primitive) WHERE p.id IN $ids "
                "RETURN p.id AS id, p.metrics_json AS metrics_json",
            )
            try:
                with self._driver.session(database=self._database) as session:
                    records = session.run(cypher, ids=[str(pid) for pid in primitive_ids])
                    for record in records:
                        pid = UUID(record["id"])
                        raw = record["metrics_json"]
                        if raw:
                            try:
                                parsed = self._parse_json_field(raw)
                                result[pid] = PrimitiveMetrics.model_validate(parsed)
                            except Exception as exc:
                                raise HydrationError(
                                    f"Failed to hydrate metrics for primitive '{pid}': {exc}"
                                ) from exc
                        else:
                            result[pid] = None
            except HydrationError:
                raise
            except Neo4jError as exc:
                raise GraphError(f"Failed to batch-read metrics: {exc}") from exc
        return result

    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None:
        if not updates:
            return
        params = [
            {"id": str(pid), "metrics_json": json.dumps(m.model_dump(mode="json"))}
            for pid, m in updates.items()
        ]
        cypher = cast(
            LiteralString,
            "UNWIND $updates AS u "
            "MATCH (p:Primitive {id: u.id}) "
            "SET p.metrics_json = u.metrics_json",
        )
        try:
            with self._driver.session(database=self._database) as session:
                session.run(cypher, updates=params)
        except Neo4jError as exc:
            raise PersistenceError(f"Failed to batch-update metrics: {exc}") from exc

    def list_names(self) -> list[str]:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    cast(LiteralString, "MATCH (p:Primitive) RETURN p.name AS name ORDER BY p.name")
                )
                return [record["name"] for record in result]
        except Neo4jError as exc:
            raise GraphError(f"Failed to list primitive names: {exc}") from exc

    def find_by_id(self, id: UUID) -> Primitive | None:
        cypher = cast(
            LiteralString,
            """
            MATCH (p:Primitive {id: $id})
            OPTIONAL MATCH (p)-[r]->(t:Primitive)
            RETURN
              p.id AS id,
              p.name AS name,
              p.depths_json AS depths_json,
              p.provenance AS provenance,
              p.metrics_json AS metrics_json,
              collect({
                rel_type: type(r),
                target_id: t.id,
                source_depth: r.source_depth,
                target_depth: r.target_depth,
                metadata_json: coalesce(r.metadata_json, "{}"),
                policies: coalesce(r.policies, "[]"),
                provenance: r.provenance
              }) AS rels
            """,
        )

        try:
            with self._driver.session(database=self._database) as session:
                record = session.run(cypher, id=str(id)).single()
                if record is None or record["id"] is None:
                    logger.debug("Primitive not found by id=%s", id)
                    primitive = None
                else:
                    primitive = self._hydrate_primitive(
                        self._record_to_node_data(record),
                        self._record_to_relationships(record),
                    )
        except (GraphError, HydrationError):
            raise
        except Neo4jError as exc:
            logger.error("Neo4j error finding primitive by id=%s: %s", id, exc)
            raise GraphError(f"Failed to find primitive by id '{id}': {exc}") from exc
        return primitive

    def find_by_name(self, name: str) -> Primitive | None:
        cypher = cast(
            LiteralString,
            """
            MATCH (p:Primitive)
            WHERE toLower(p.name) = toLower($name)
            OPTIONAL MATCH (p)-[r]->(t:Primitive)
            RETURN
              p.id AS id,
              p.name AS name,
              p.depths_json AS depths_json,
              p.provenance AS provenance,
              p.metrics_json AS metrics_json,
              collect({
                rel_type: type(r),
                target_id: t.id,
                source_depth: r.source_depth,
                target_depth: r.target_depth,
                metadata_json: coalesce(r.metadata_json, "{}"),
                policies: coalesce(r.policies, "[]"),
                provenance: r.provenance
              }) AS rels
            """,
        )

        try:
            with self._driver.session(database=self._database) as session:
                record = session.run(cypher, name=name).single()
                if record is None or record["id"] is None:
                    logger.debug("Primitive not found by name=%r", name)
                    primitive = None
                else:
                    primitive = self._hydrate_primitive(
                        self._record_to_node_data(record),
                        self._record_to_relationships(record),
                    )
        except (GraphError, HydrationError):
            raise
        except Neo4jError as exc:
            logger.error("Neo4j error finding primitive by name=%r: %s", name, exc)
            raise GraphError(f"Failed to find primitive by name '{name}': {exc}") from exc
        return primitive

    def delete_primitive(self, id: UUID) -> bool:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    cast(
                        LiteralString,
                        "MATCH (p:Primitive {id: $id}) DETACH DELETE p RETURN count(p) AS deleted",
                    ),
                    id=str(id),
                ).single()
                return result is not None and result["deleted"] > 0
        except Neo4jError as exc:
            raise PersistenceError(f"Failed to delete primitive '{id}': {exc}") from exc

    def clear(self) -> int:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    cast(
                        LiteralString,
                        "MATCH (p:Primitive) DETACH DELETE p RETURN count(p) AS deleted",
                    ),
                ).single()
                return result["deleted"] if result else 0
        except Neo4jError as exc:
            raise PersistenceError(f"Failed to clear graph: {exc}") from exc

    def resolve_subgraph(
        self,
        names: list[str],
    ) -> ResolvedSubgraph:
        logger.debug("Resolving subgraph for names=%s", names)
        lowered = [n.lower() for n in names]

        cypher = cast(
            LiteralString,
            """
            // Phase 1: Resolve roots by name
            MATCH (root:Primitive)
            WHERE toLower(root.name) IN $names
            WITH collect(root) AS roots

            // Phase 2: Recursive traversal from all roots (no depth ceiling)
            UNWIND roots AS r
            OPTIONAL MATCH path = (r)-[rels*0..]->(reached:Primitive)
            WHERE all(rel IN rels WHERE type(rel) IN $transitive_types)
            WITH roots, collect(DISTINCT reached) AS traversed

            // Deduplicate roots + traversed nodes
            WITH roots,
                 [n IN roots + traversed WHERE n IS NOT NULL] AS raw_nodes
            UNWIND raw_nodes AS raw
            WITH roots, collect(DISTINCT raw) AS nodes

            // Phase 3: All edges between collected nodes (no depth ceiling)
            UNWIND nodes AS src
            OPTIONAL MATCH (src)-[r]->(tgt:Primitive)
            WHERE tgt IN nodes
            RETURN
              [r IN roots | {id: r.id, name: r.name, depths_json: r.depths_json, provenance: r.provenance, metrics_json: r.metrics_json}] AS roots,
              [n IN nodes | {id: n.id, name: n.name, depths_json: n.depths_json, provenance: n.provenance, metrics_json: n.metrics_json}] AS nodes,
              [e IN collect({
                  source_id: src.id, target_id: tgt.id, rel_type: type(r),
                  source_depth: r.source_depth, target_depth: r.target_depth,
                  metadata_json: coalesce(r.metadata_json, "{}"),
                  policies: coalesce(r.policies, "[]"),
                  provenance: r.provenance
              }) WHERE e.rel_type IS NOT NULL] AS edges
            """,
        )

        try:
            with self._driver.session(database=self._database) as session:
                record = session.run(
                    cypher,
                    names=lowered,
                    transitive_types=_TRANSITIVE_RELS,
                ).single()

                if record is None:
                    subgraph = ResolvedSubgraph(roots=[], nodes=[], edges=[])
                else:
                    raw_roots = record["roots"]
                    raw_nodes = record["nodes"]
                    raw_edges = record["edges"]

                    edges_by_source: dict[str, list[dict[str, Any]]] = {}
                    for e in raw_edges:
                        sid = e["source_id"]
                        edges_by_source.setdefault(sid, []).append({
                            "rel_type": e["rel_type"],
                            "target_id": e["target_id"],
                            "rel_props": {
                                "source_depth": e["source_depth"],
                                "target_depth": e["target_depth"],
                                "metadata_json": e.get("metadata_json", "{}"),
                                "policies": e.get("policies", "[]"),
                                "provenance": e.get("provenance"),
                            },
                        })

                    roots = [
                        self._hydrate_primitive(r, edges_by_source.get(r["id"], []))
                        for r in raw_roots
                    ]
                    nodes = [
                        self._hydrate_primitive(n, edges_by_source.get(n["id"], []))
                        for n in raw_nodes
                    ]

                    edges = [
                        EpistemicStep(
                            source_id=UUID(e["source_id"]),
                            target_id=UUID(e["target_id"]),
                            relation_type=RelationType(e["rel_type"]),
                            source_depth=DepthLevel(e["source_depth"]),
                            target_depth=DepthLevel(e["target_depth"]),
                        )
                        for e in raw_edges
                    ]

                    logger.debug(
                        "Subgraph resolved: %d roots, %d nodes, %d edges",
                        len(roots), len(nodes), len(edges),
                    )
                    subgraph = ResolvedSubgraph(roots=roots, nodes=nodes, edges=edges)
        except (GraphError, HydrationError):
            raise
        except Neo4jError as exc:
            logger.error("Neo4j error resolving subgraph for %s: %s", names, exc)
            raise GraphError(f"Failed to resolve subgraph for {names}: {exc}") from exc
        return subgraph
