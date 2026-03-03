# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Grounding engine for the Volute Reasoning Engine.

Provides GroundingEngine — the single entry point for structured epistemic
queries. Always checks at DepthLevel.CONSTRAINTS (D3). The agent submits
concept names; the engine delegates recursive traversal to the repository
(Cypher), then applies depth gating, gap detection, and closure filtering.
"""

from __future__ import annotations

from uuid import UUID

from vre.core.graph import PrimitiveRepository
from vre.core.grounding.models import GroundingResult
from vre.core.grounding.resolver import ConceptResolver
from vre.core.models import (
    Depth,
    DepthGap,
    DepthLevel,
    EpistemicQuery,
    EpistemicResponse,
    EpistemicResult,
    EpistemicStep,
    ExistenceGap,
    Primitive,
    ReachabilityGap,
    RelationalGap,
)

_REQUIRED_DEPTH = DepthLevel.CONSTRAINTS


def _empty_response() -> EpistemicResponse:
    """
    Helper for the empty query case — returns a valid but empty response with no
    EpistemicQuery or EpistemicResult data.
    """
    return EpistemicResponse(
        query=EpistemicQuery(concept_ids=[]),
        result=EpistemicResult(primitives=[]),
    )


class GroundingEngine:
    """
    Structured epistemic query resolution at D3 (CONSTRAINTS).

    Stateless between calls. Accepts concept names, delegates graph
    traversal to the repository, and returns a fully closed epistemic
    response.
    """

    def __init__(self, repository: PrimitiveRepository) -> None:
        """
        Initialize the grounding engine with a primitive repository.
        """
        self._repo = repository

    def list_primitive_names(self) -> list[str]:
        """
        Return a list of all primitive names in the repository.
        """
        return self._repo.list_names()

    @staticmethod
    def _identify_roots(
        names: list[str],
        resolved_roots: list[Primitive],
    ) -> tuple[list[Primitive], list[Primitive]]:
        """
        Identify root primitives for the query based on the input names.
        """
        by_name: dict[str, Primitive] = {r.name.lower(): r for r in resolved_roots}
        all_roots: list[Primitive] = []
        transients: list[Primitive] = []
        for name in names:
            matched = by_name.get(name.lower())
            if matched:
                all_roots.append(matched)
            else:
                t = Primitive(name=name, depths=[])
                all_roots.append(t)
                transients.append(t)
        return all_roots, transients

    @staticmethod
    def _contiguous_max_depth(node: Primitive) -> DepthLevel | None:
        """
        Return the highest DepthLevel forming a contiguous chain from D0, or None if no depths.
        """
        present = {d.level for d in node.depths}
        result: DepthLevel | None = None
        for level in sorted(DepthLevel):
            if level not in present:
                break
            result = level
        return result

    @staticmethod
    def _detect_gaps(
        all_nodes: list[Primitive],
        edges: list[EpistemicStep],
        root_ids: set[UUID],
        transient_ids: set[UUID],
    ) -> list[DepthGap | ExistenceGap | RelationalGap]:
        """
        Detect existence, depth, and relational gaps across the resolved subgraph.
        """
        gaps: list[DepthGap | ExistenceGap | RelationalGap] = []
        id_to_prim = {n.id: n for n in all_nodes}

        # Phase 1 — Existence gaps
        for node in all_nodes:
            if node.id in transient_ids:
                gaps.append(ExistenceGap(primitive=node))

        # Phase 2 — Depth gaps (roots only, always checked at D3)
        for node in all_nodes:
            if node.id in transient_ids or node.id not in root_ids:
                continue
            contiguous = GroundingEngine._contiguous_max_depth(node)
            if contiguous is None or contiguous < _REQUIRED_DEPTH:
                gaps.append(DepthGap(
                    primitive=node,
                    required_depth=_REQUIRED_DEPTH,
                    current_depth=contiguous,
                ))

        # Phase 3 — Relatum-depth relational gaps
        relatum_depth_pairs: dict[tuple[UUID, UUID], DepthLevel] = {}
        for edge in edges:
            if edge.target_id in transient_ids:
                continue
            tgt_prim = id_to_prim.get(edge.target_id)
            if tgt_prim is None:
                continue
            tgt_contiguous = GroundingEngine._contiguous_max_depth(tgt_prim)
            if tgt_contiguous is not None and tgt_contiguous >= edge.target_depth:
                continue
            pair = (edge.source_id, edge.target_id)
            existing = relatum_depth_pairs.get(pair)
            if existing is None or edge.target_depth > existing:
                relatum_depth_pairs[pair] = edge.target_depth
        for (src_id, tgt_id), max_req in relatum_depth_pairs.items():
            src_prim = id_to_prim.get(src_id)
            tgt_prim = id_to_prim.get(tgt_id)
            if src_prim is None or tgt_prim is None:
                continue
            curr = GroundingEngine._contiguous_max_depth(tgt_prim)
            gaps.append(RelationalGap(
                source=src_prim, target=tgt_prim,
                required_depth=max_req, current_depth=curr,
            ))

        return gaps

    @staticmethod
    def _reachable_undirected(root_id: UUID, neighbors: dict[UUID, set[UUID]]) -> set[UUID]:
        """
        BFS from root_id over the undirected neighbor graph; returns all reachable node IDs.
        """
        visited: set[UUID] = {root_id}
        queue: list[UUID] = [root_id]
        while queue:
            current = queue.pop()
            for neighbor in neighbors.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    @staticmethod
    def _filter_depths(all_nodes: list[Primitive]) -> list[Primitive]:
        """
        Return copies of all_nodes with relata filtered to targets present in the collected set.
        """
        collected_ids = {n.id for n in all_nodes}
        return [
            Primitive(
                id=p.id, name=p.name,
                depths=[
                    Depth(
                        level=d.level, properties=d.properties,
                        relata=[r for r in d.relata if r.target_id in collected_ids],
                    )
                    for d in p.depths
                ],
            )
            for p in all_nodes
        ]

    def query(self, concepts: list[str]) -> EpistemicResponse:
        """
        Flat-concept epistemic query at D3 (CONSTRAINTS).

        All submitted concepts are treated symmetrically. Resolves the
        subgraph for all concepts, then checks that every non-transient
        concept is in the same connected component (undirected BFS).
        """
        if not concepts:
            return _empty_response()

        subgraph = self._repo.resolve_subgraph(concepts)

        roots, transients = self._identify_roots(concepts, subgraph.roots)
        transient_ids = {t.id for t in transients}
        root_ids = {r.id for r in roots}
        all_nodes = list(subgraph.nodes) + transients

        gaps: list = self._detect_gaps(
            all_nodes, subgraph.edges, root_ids, transient_ids,
        )

        # Undirected connectivity check across all non-transient roots
        non_transient_roots = [r for r in roots if r.id not in transient_ids]
        if len(non_transient_roots) > 1:
            neighbors: dict[UUID, set[UUID]] = {}
            for edge in subgraph.edges:
                neighbors.setdefault(edge.source_id, set()).add(edge.target_id)
                neighbors.setdefault(edge.target_id, set()).add(edge.source_id)
            anchor = non_transient_roots[0]
            reachable = self._reachable_undirected(anchor.id, neighbors)
            for root in non_transient_roots[1:]:
                if root.id not in reachable:
                    gaps.append(ReachabilityGap(primitive=root))

        filtered = self._filter_depths(all_nodes)

        return EpistemicResponse(
            query=EpistemicQuery(concept_ids=[r.id for r in roots]),
            result=EpistemicResult(primitives=filtered, gaps=gaps, pathway=subgraph.edges),
        )

    def ground(
        self,
        concepts: list[str],
        resolver: ConceptResolver,
    ) -> GroundingResult:
        """
        Resolve and ground concepts in one step.

        Each concept is resolved to its canonical name where possible;
        unknown concepts pass through as-is and become ExistenceGaps in the
        query result. Returns a GroundingResult with grounded=True only when
        all concepts are grounded at D3 with no gaps.
        """
        if not concepts:
            return GroundingResult(grounded=False, resolved=[], gaps=[], trace=None)

        # Resolve to canonical names where possible; unknown names pass through
        # as-is — the engine will surface ExistenceGaps for them.
        name_map = resolver.build_name_map()
        canonical = [
            (resolver.lookup(c, name_map) or c)
            for c in concepts
        ]

        response = self.query(canonical)
        grounded = len(response.result.gaps) == 0
        return GroundingResult(
            grounded=grounded,
            resolved=canonical,
            gaps=response.result.gaps,
            trace=response,
        )
