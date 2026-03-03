# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Neo4j persistence layer for the Volute Reasoning Engine.

Provides PrimitiveRepository — the bridge between Pydantic epistemic
models and the Neo4j graph database. Primitives are stored as nodes
with embedded depth JSON; relata are stored as typed Neo4j relationships.
"""

import json
from typing import Any, LiteralString, cast
from uuid import UUID

from neo4j import GraphDatabase

from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicStep,
    Primitive,
    Relatum,
    RelationType,
    ResolvedSubgraph,
)
from vre.core.policy.models import parse_policy


_TRANSITIVE_RELS = ["REQUIRES", "DEPENDS_ON", "CONSTRAINED_BY"]


class PrimitiveRepository:
    """
    Neo4j persistence layer for epistemic primitives.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        """
        Connect to a Neo4j instance at the given URI with the provided credentials.
        """
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_categories=["UNRECOGNIZED"],
        )
        self._database = database

    def close(self) -> None:
        """
        Close the underlying Neo4j driver and release all connections.
        """
        self._driver.close()

    def __enter__(self) -> "PrimitiveRepository":
        """
        Enter the context manager, returning self.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context manager, closing the driver.
        """
        self.close()

    def ensure_constraints(self) -> None:
        """
        Create uniqueness constraint on Primitive.id.
        """
        with self._driver.session(database=self._database) as session:
            session.run(
                cast(
                    LiteralString,
                    "CREATE CONSTRAINT primitive_id_unique IF NOT EXISTS "
                    "FOR (p:Primitive) REQUIRE p.id IS UNIQUE",
                )
            )

    @staticmethod
    def _depths_to_json(depths: list[Depth]) -> str:
        """
        Serialize depth levels and their properties to a JSON string for Neo4j storage.
        """
        stripped = []
        for depth in depths:
            stripped.append(
                {
                    "level": int(depth.level),
                    "properties": depth.properties,
                }
            )
        return json.dumps(stripped)

    @staticmethod
    def _hydrate_primitive(
        node_data: dict[str, Any],
        relationships: list[dict[str, Any]],
    ) -> Primitive:
        """
        Reconstruct a Primitive from raw Neo4j node data and its relationship records.
        """
        raw_depths = json.loads(node_data["depths_json"])
        depths_by_level: dict[int, Depth] = {}
        for rd in raw_depths:
            depth = Depth(
                level=DepthLevel(rd["level"]),
                properties=rd.get("properties", {}),
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

            relatum = Relatum(
                relation_type=RelationType(rel["rel_type"]),
                target_id=UUID(rel["target_id"]),
                target_depth=DepthLevel(target_depth_val),
                metadata=metadata,
                policies=policies,
            )

            if source_depth is not None and source_depth in depths_by_level:
                depths_by_level[source_depth].relata.append(relatum)

        sorted_depths = sorted(depths_by_level.values(), key=lambda d: int(d.level))

        return Primitive(
            id=UUID(node_data["id"]),
            name=node_data["name"],
            depths=sorted_depths,
        )

    def save_primitive(self, primitive: Primitive) -> None:
        """
        Persist a Primitive — full replace of depths and relata.
        """
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
                    }
                )

        rel_types = [rt.value for rt in RelationType]

        def _tx(tx: Any) -> None:
            tx.run(
                cast(
                    LiteralString,
                    "MERGE (p:Primitive {id: $id}) "
                    "SET p.name = $name, p.depths_json = $depths_json",
                ),
                id=str(primitive.id),
                name=primitive.name,
                depths_json=depths_json,
            )

            for rt in rel_types:
                tx.run(
                    cast(
                        LiteralString,
                        f"MATCH (p:Primitive {{id: $id}})-[r:{rt}]->() DELETE r",
                    ),
                    id=str(primitive.id),
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
                        f"policies: $policies"
                        f"}}]->(t)",
                    ),
                    source_id=str(primitive.id),
                    target_id=rp["target_id"],
                    source_depth=rp["source_depth"],
                    target_depth=rp["target_depth"],
                    metadata_json=rp["metadata_json"],
                    policies=rp["policies"],
                )

        with self._driver.session(database=self._database) as session:
            session.execute_write(_tx)

    def list_names(self) -> list[str]:
        """
        Return the names of all primitives in the graph, sorted alphabetically.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                cast(LiteralString, "MATCH (p:Primitive) RETURN p.name AS name ORDER BY p.name")
            )
            return [record["name"] for record in result]

    def find_by_id(self, id: UUID) -> Primitive | None:
        """
        Look up a primitive by its UUID, returning None if not found.
        """
        cypher = cast(
            LiteralString,
            """
            MATCH (p:Primitive {id: $id})
            OPTIONAL MATCH (p)-[r]->(t:Primitive)
            RETURN
              p.id AS id,
              p.name AS name,
              p.depths_json AS depths_json,
              collect({
                rel_type: type(r),
                target_id: t.id,
                source_depth: r.source_depth,
                target_depth: r.target_depth,
                metadata_json: coalesce(r.metadata_json, "{}"),
                policies: coalesce(r.policies, "[]")
              }) AS rels
            """,
        )

        with self._driver.session(database=self._database) as session:
            record = session.run(cypher, id=str(id)).single()
            if record is None or record["id"] is None:
                return None

            node_data = {
                "id": record["id"],
                "name": record["name"],
                "depths_json": record["depths_json"],
            }
            relationships = [
                {
                    "rel_type": r["rel_type"],
                    "target_id": r["target_id"],
                    "rel_props": {
                        "source_depth": r["source_depth"],
                        "target_depth": r["target_depth"],
                        "metadata_json": r.get("metadata_json") or "{}",
                        "policies": r.get("policies") or "[]",
                    },
                }
                for r in record["rels"]
                if r.get("rel_type") is not None
            ]
            return self._hydrate_primitive(node_data, relationships)

    def find_by_name(self, name: str) -> Primitive | None:
        """
        Look up a primitive by name (case-insensitive), returning None if not found.
        """
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
              collect({
                rel_type: type(r),
                target_id: t.id,
                source_depth: r.source_depth,
                target_depth: r.target_depth,
                metadata_json: coalesce(r.metadata_json, "{}"),
                policies: coalesce(r.policies, "[]")
              }) AS rels
            """,
        )

        with self._driver.session(database=self._database) as session:
            record = session.run(cypher, name=name).single()
            if record is None or record["id"] is None:
                return None

            node_data = {
                "id": record["id"],
                "name": record["name"],
                "depths_json": record["depths_json"],
            }
            relationships = [
                {
                    "rel_type": r["rel_type"],
                    "target_id": r["target_id"],
                    "rel_props": {
                        "source_depth": r["source_depth"],
                        "target_depth": r["target_depth"],
                        "metadata_json": r.get("metadata_json") or "{}",
                        "policies": r.get("policies") or "[]",
                    },
                }
                for r in record["rels"]
                if r.get("rel_type") is not None
            ]
            return self._hydrate_primitive(node_data, relationships)

    def delete_primitive(self, id: UUID) -> bool:
        """
        Delete the primitive with the given UUID and all its relationships. Returns True if deleted.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                cast(
                    LiteralString,
                    "MATCH (p:Primitive {id: $id}) DETACH DELETE p RETURN count(p) AS deleted",
                ),
                id=str(id),
            ).single()
            return result is not None and result["deleted"] > 0

    def resolve_subgraph(
        self,
        names: list[str],
    ) -> ResolvedSubgraph:
        """
        Single-query Cypher traversal that resolves a subgraph for the given names.
        """
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
              [r IN roots | {id: r.id, name: r.name, depths_json: r.depths_json}] AS roots,
              [n IN nodes | {id: n.id, name: n.name, depths_json: n.depths_json}] AS nodes,
              [e IN collect({
                  source_id: src.id, target_id: tgt.id, rel_type: type(r),
                  source_depth: r.source_depth, target_depth: r.target_depth,
                  metadata_json: coalesce(r.metadata_json, "{}"),
                  policies: coalesce(r.policies, "[]")
              }) WHERE e.rel_type IS NOT NULL] AS edges
            """,
        )

        with self._driver.session(database=self._database) as session:
            record = session.run(
                cypher,
                names=lowered,
                transitive_types=_TRANSITIVE_RELS,
            ).single()

            if record is None:
                return ResolvedSubgraph(roots=[], nodes=[], edges=[])

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

            return ResolvedSubgraph(roots=roots, nodes=nodes, edges=edges)
