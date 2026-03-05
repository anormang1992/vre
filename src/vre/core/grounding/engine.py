# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Grounding engine for the Volute Reasoning Engine.

Provides GroundingEngine — the single entry point for structured epistemic
queries. Depth requirements are derived from graph structure: edges carry
source_depth indicating the depth level they live at on the source node.
The engine partitions edges into visible (source grounded deeply enough)
and gated (source too shallow), producing DepthGaps for the latter.

An optional min_depth parameter provides integrators a secondary safety
lever to enforce a stricter floor than the graph alone would require.
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
    Structured epistemic query resolution with graph-derived depth gating.

    Stateless between calls. Accepts concept names, delegates graph
    traversal to the repository, partitions edges by source depth
    visibility, and returns a fully closed epistemic response.
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
    def _partition_edges_by_source_depth(
        edges: list[EpistemicStep],
        id_to_prim: dict[UUID, Primitive],
    ) -> tuple[list[EpistemicStep], list[EpistemicStep]]:
        """
        Split edges into visible and gated based on source node grounding.

        An edge is visible when the source node's contiguous max depth >= the
        edge's source_depth. Otherwise, the edge is gated — the source isn't
        grounded deeply enough to see the relationship.
        """
        visible: list[EpistemicStep] = []
        gated: list[EpistemicStep] = []
        for edge in edges:
            src = id_to_prim.get(edge.source_id)
            if src is None:
                continue
            src_contiguous = GroundingEngine._contiguous_max_depth(src)
            if src_contiguous is not None and src_contiguous >= edge.source_depth:
                visible.append(edge)
            else:
                gated.append(edge)
        return visible, gated

    @staticmethod
    def _detect_gaps(
        all_nodes: list[Primitive],
        visible_edges: list[EpistemicStep],
        gated_edges: list[EpistemicStep],
        root_ids: set[UUID],
        transient_ids: set[UUID],
        min_depth: DepthLevel | None = None,
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

        # Phase 2 — Depth gaps from two sources:
        #   (a) gated edges: source can't see the edge
        #   (b) min_depth override: integrator safety lever
        # Deduplicate per-primitive, keeping the higher required_depth.
        depth_gap_map: dict[UUID, tuple[DepthLevel, DepthLevel | None]] = {}

        # (a) Gated edges → DepthGap on source primitive
        for edge in gated_edges:
            src = id_to_prim.get(edge.source_id)
            if src is None or src.id in transient_ids:
                continue
            src_contiguous = GroundingEngine._contiguous_max_depth(src)
            existing = depth_gap_map.get(src.id)
            if existing is None or edge.source_depth > existing[0]:
                depth_gap_map[src.id] = (edge.source_depth, src_contiguous)

        # (b) min_depth override on roots
        if min_depth is not None:
            for node in all_nodes:
                if node.id in transient_ids or node.id not in root_ids:
                    continue
                contiguous = GroundingEngine._contiguous_max_depth(node)
                if contiguous is None or contiguous < min_depth:
                    existing = depth_gap_map.get(node.id)
                    if existing is None or min_depth > existing[0]:
                        depth_gap_map[node.id] = (min_depth, contiguous)

        for nid, (req, curr) in depth_gap_map.items():
            prim = id_to_prim.get(nid)
            if prim is not None:
                gaps.append(DepthGap(
                    primitive=prim,
                    required_depth=req,
                    current_depth=curr,
                ))

        # Phase 3 — Relatum-depth relational gaps (visible edges only)
        relatum_depth_pairs: dict[tuple[UUID, UUID], DepthLevel] = {}
        for edge in visible_edges:
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

    def query(
        self,
        concepts: list[str],
        min_depth: DepthLevel | None = None,
    ) -> EpistemicResponse:
        """
        Flat-concept epistemic query with graph-derived depth gating.

        All submitted concepts are treated symmetrically. Resolves the
        subgraph for all concepts, partitions edges by source depth
        visibility, then checks that every non-transient concept is in
        the same connected component (undirected BFS over visible edges).

        Parameters
        ----------
        concepts:
            Canonical concept names to query.
        min_depth:
            Optional integrator override — enforces a minimum depth floor
            on all root primitives. Can only raise the floor, never lower
            it below what the graph structure requires.
        """
        if not concepts:
            return _empty_response()

        subgraph = self._repo.resolve_subgraph(concepts)

        roots, transients = self._identify_roots(concepts, subgraph.roots)
        transient_ids = {t.id for t in transients}
        root_ids = {r.id for r in roots}
        all_nodes = list(subgraph.nodes) + transients

        id_to_prim = {n.id: n for n in all_nodes}
        visible_edges, gated_edges = self._partition_edges_by_source_depth(
            subgraph.edges, id_to_prim,
        )

        gaps: list = self._detect_gaps(
            all_nodes, visible_edges, gated_edges, root_ids, transient_ids,
            min_depth=min_depth,
        )

        # Undirected connectivity check across all non-transient roots
        # using only visible edges
        non_transient_roots = [r for r in roots if r.id not in transient_ids]
        if len(non_transient_roots) > 1:
            neighbors: dict[UUID, set[UUID]] = {}
            for edge in visible_edges:
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
            result=EpistemicResult(primitives=filtered, gaps=gaps, pathway=visible_edges),
        )

    def ground(
        self,
        concepts: list[str],
        resolver: ConceptResolver,
        min_depth: DepthLevel | None = None,
    ) -> GroundingResult:
        """
        Resolve and ground concepts in one step.

        Each concept is resolved to its canonical name where possible;
        unknown concepts pass through as-is and become ExistenceGaps in the
        query result. Returns a GroundingResult with grounded=True only when
        all concepts are grounded with no gaps.
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

        response = self.query(canonical, min_depth=min_depth)
        grounded = len(response.result.gaps) == 0
        return GroundingResult(
            grounded=grounded,
            resolved=canonical,
            gaps=response.result.gaps,
            trace=response,
        )
