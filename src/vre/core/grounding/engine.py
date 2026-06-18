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

import logging
from uuid import UUID

from vre.core.backends import Repository
from vre.core.errors import GraphIntegrityError
from vre.core.grounding.models import GroundingResult
from vre.core.models import (
    DepthGap,
    DepthLevel,
    EpistemicQuery,
    EpistemicResponse,
    EpistemicResult,
    EpistemicStep,
    ExistenceGap,
    Primitive,
    Provenance,
    ProvenanceSource,
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


logger = logging.getLogger(__name__)


class GroundingEngine:
    """
    Structured epistemic query resolution with graph-derived depth gating.

    Stateless between calls. Accepts concept names, delegates graph
    traversal to the repository, partitions edges by source depth
    visibility, and returns a fully closed epistemic response.
    """

    def __init__(self, repository: Repository) -> None:
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

        A concept appearing more than once in one query — repeated verbatim or
        differing only by case — collapses to a single root. Folded-name dedup
        runs over the raw input, so one concept yields one root id (and, when
        absent from the graph, one synthetic placeholder and one ExistenceGap)
        regardless of how many times or in what casing it appears. First
        occurrence wins, so order is preserved and the result is deterministic
        (#130).
        """
        by_name: dict[str, Primitive] = {Primitive.fold_name(r.name): r for r in resolved_roots}
        all_roots: list[Primitive] = []
        transients: list[Primitive] = []
        seen: set[str] = set()
        for name in names:
            fold = Primitive.fold_name(name)
            if fold in seen:
                continue
            seen.add(fold)
            matched = by_name.get(fold)
            if matched is not None:
                all_roots.append(matched)
                continue
            # The concept is not in the graph. We manufacture a transient
            # placeholder to anchor the traversal and the ExistenceGap. It
            # carries no knowledge (depths=[]) and is never persisted, but it
            # came from somewhere — the engine, surfacing the *absence* of the
            # concept — so its provenance is SYNTHETIC, not a forged human one.
            transient = Primitive(
                name=name,
                depths=[],
                provenance=Provenance(
                    source=ProvenanceSource.SYNTHETIC,
                    detail=f"transient placeholder for concept '{name}' absent from the graph",
                ),
            )
            transients.append(transient)
            all_roots.append(transient)
        return all_roots, transients

    @staticmethod
    def _validate_edge_endpoints(
        edges: list[EpistemicStep],
        id_to_prim: dict[UUID, Primitive],
    ) -> None:
        """
        Fail loud if any resolved edge references a node absent from the set.

        A conformant Repository returns edges only between nodes it also
        returns. A dangling endpoint is a backend-contract violation, not a
        depth gap, so it is surfaced as GraphIntegrityError rather than being
        silently dropped (source side) or passed through as grounded (target
        side). This keeps both sides of the same violation consistent and
        fail-closed (#94 Finding C1). Running here, before any edge is
        partitioned or interpreted, lets every downstream step index
        id_to_prim directly instead of guarding each lookup.
        """
        for edge in edges:
            for role, node_id in (("source", edge.source_id), ("target", edge.target_id)):
                if node_id not in id_to_prim:
                    raise GraphIntegrityError(
                        f"Resolved {edge.relation_type.value} edge references a "
                        f"{role} node {node_id} absent from the resolved subgraph; "
                        f"the backend returned an edge whose {role} it did not return."
                    )

    @staticmethod
    def _partition_edges_by_source_depth(
        edges: list[EpistemicStep],
        id_to_prim: dict[UUID, Primitive],
    ) -> tuple[list[EpistemicStep], list[EpistemicStep]]:
        """
        Split edges into visible and gated based on source node grounding.

        An edge is visible when the source node's contiguous max depth >= the
        edge's source_depth. Otherwise, the edge is gated — the source isn't
        grounded deeply enough to see the relationship. Endpoints are
        pre-validated by _validate_edge_endpoints, so the source lookup is a
        direct index.
        """
        visible: list[EpistemicStep] = []
        gated: list[EpistemicStep] = []
        for edge in edges:
            src = id_to_prim[edge.source_id]
            src_contiguous = src.contiguous_max_depth
            if src_contiguous is not None and src_contiguous >= edge.source_depth:
                visible.append(edge)
            else:
                gated.append(edge)
        logger.debug("Edge partition: %d visible, %d gated", len(visible), len(gated))
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

        The visible and gated edge lists are pre-restricted to the justified
        frontier (source on a fully-visible path from a root), so no gap is ever
        emitted about a node the agent cannot yet reach (#94). Edge endpoints are
        pre-validated, so every id_to_prim lookup is a direct index.
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
            src = id_to_prim[edge.source_id]
            if src.id in transient_ids:
                continue
            src_contiguous = src.contiguous_max_depth
            existing = depth_gap_map.get(src.id)
            if existing is None or edge.source_depth > existing[0]:
                depth_gap_map[src.id] = (edge.source_depth, src_contiguous)

        # (b) min_depth override on roots
        if min_depth is not None:
            for node in all_nodes:
                if node.id in transient_ids or node.id not in root_ids:
                    continue
                contiguous = node.contiguous_max_depth
                if contiguous is None or contiguous < min_depth:
                    existing = depth_gap_map.get(node.id)
                    if existing is None or min_depth > existing[0]:
                        depth_gap_map[node.id] = (min_depth, contiguous)

        for nid, (req, curr) in depth_gap_map.items():
            gaps.append(DepthGap(
                primitive=id_to_prim[nid],
                required_depth=req,
                current_depth=curr,
            ))

        # Phase 3 — Relatum-depth relational gaps (visible edges only)
        relatum_depth_pairs: dict[tuple[UUID, UUID], DepthLevel] = {}
        for edge in visible_edges:
            if edge.target_id in transient_ids:
                continue
            tgt_prim = id_to_prim[edge.target_id]
            tgt_contiguous = tgt_prim.contiguous_max_depth
            if tgt_contiguous is not None and tgt_contiguous >= edge.target_depth:
                continue
            pair = (edge.source_id, edge.target_id)
            existing = relatum_depth_pairs.get(pair)
            if existing is None or edge.target_depth > existing:
                relatum_depth_pairs[pair] = edge.target_depth
        for (src_id, tgt_id), max_req in relatum_depth_pairs.items():
            src_prim = id_to_prim[src_id]
            tgt_prim = id_to_prim[tgt_id]
            curr = tgt_prim.contiguous_max_depth
            gaps.append(RelationalGap(
                source=src_prim,
                target=tgt_prim,
                required_depth=max_req,
                current_depth=curr,
            ))

        if gaps:
            logger.info("Detected %d gap(s): %s", len(gaps), [g.kind for g in gaps])
        else:
            logger.debug("No gaps detected")
        return gaps

    @staticmethod
    def _reachable_undirected(root_id: UUID, neighbors: dict[UUID, set[UUID]]) -> set[UUID]:
        """
        DFS from root_id over the undirected neighbor graph; returns all reachable node IDs.

        Traversal order does not affect the result — the full connected component
        is returned regardless — so a stack (DFS) is used.
        """
        visited: set[UUID] = {root_id}
        stack: list[UUID] = [root_id]
        while stack:
            current = stack.pop()
            for neighbor in neighbors.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    @staticmethod
    def _reachable_via_visible(
        root_ids: set[UUID],
        visible_edges: list[EpistemicStep],
    ) -> set[UUID]:
        """
        Directed reachability from the query roots over visible edges only.

        Returns every node id on a fully-visible directed path from some root
        (roots included). This is the justified frontier: a node reached only
        through a gated edge is excluded, so it never enters the response. The
        direction matters — undirected reachability would keep more nodes than a
        depth-aware closure in the Repository would, whereas this directed set is
        exactly what such a closure returns, so #87 can later push the same
        boundary into the query layer as a behavior-preserving no-op (#94).
        """
        adjacency: dict[UUID, list[UUID]] = {}
        for edge in visible_edges:
            adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        reachable: set[UUID] = set(root_ids)
        stack: list[UUID] = list(root_ids)
        while stack:
            current = stack.pop()
            for nxt in adjacency.get(current, ()):
                if nxt not in reachable:
                    reachable.add(nxt)
                    stack.append(nxt)
        return reachable

    @staticmethod
    def _filter_depths(nodes: list[Primitive]) -> list[Primitive]:
        """
        Copy each node down to its justified epistemic envelope.

        Two cuts, both honoring "the trace surfaces only what is grounded":
          - depth content above the node's contiguous_max_depth is dropped, so a
            node grounded to D1 never ships the properties or relata of a D3 it
            cannot justify (#94 Finding D1);
          - on the surviving depths, relata whose target was pruned from the
            response are dropped, so a relatum never points outside the returned
            set.

        Returns fresh primitive and depth shells with deep-copied properties; the
        surviving Relatum objects are referenced, not re-copied, so this is not a
        deep clone of the relata graph. Downstream consumers (metrics, tracing)
        treat the trace as read-only. (#87 owns the copy strategy; this states
        what holds today rather than the prior "fully detached" claim, which was
        false — #94 Finding D2.)
        """
        collected_ids = {n.id for n in nodes}
        filtered: list[Primitive] = []
        for p in nodes:
            cmax = p.contiguous_max_depth
            kept_depths = [] if cmax is None else [
                d.model_copy(
                    update={"relata": [r for r in d.relata if r.target_id in collected_ids]},
                    deep=True,
                )
                for d in p.depths
                if d.level <= cmax
            ]
            filtered.append(p.model_copy(update={"depths": kept_depths}, deep=True))
        return filtered

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
            logger.debug("Empty concept list, returning empty response")
            response = _empty_response()
        else:
            subgraph = self._repo.resolve_subgraph(concepts)

            roots, transients = self._identify_roots(concepts, subgraph.roots)
            transient_ids = {t.id for t in transients}
            root_ids = {r.id for r in roots}
            all_nodes = list(subgraph.nodes) + transients
            logger.debug(
                "Query for %s: resolved %d roots (%d transient), %d nodes, %d edges",
                concepts, len(roots), len(transients), len(all_nodes), len(subgraph.edges),
            )

            id_to_prim = {n.id: n for n in all_nodes}
            # Fail loud on a backend that returns an edge dangling outside its own
            # node set, before any edge is partitioned or interpreted (#94 C1).
            self._validate_edge_endpoints(subgraph.edges, id_to_prim)

            visible_edges, gated_edges = self._partition_edges_by_source_depth(
                subgraph.edges, id_to_prim,
            )

            # The justified frontier: nodes on a fully-visible directed path from
            # a root. A node reached only through a gated edge is not grounded
            # enough to be surfaced, so it never enters the response, its gaps, or
            # the pathway (#94 Findings A + D1). Edges are likewise restricted to
            # those whose source is on the frontier, so no gap is emitted about a
            # node the agent cannot yet reach.
            frontier_ids = self._reachable_via_visible(root_ids, visible_edges)
            frontier_visible = [e for e in visible_edges if e.source_id in frontier_ids]
            frontier_gated = [e for e in gated_edges if e.source_id in frontier_ids]

            gaps: list[DepthGap | ExistenceGap | RelationalGap | ReachabilityGap] = self._detect_gaps(
                all_nodes,
                frontier_visible,
                frontier_gated,
                root_ids,
                transient_ids,
                min_depth=min_depth,
            )

            # Undirected connectivity check across all non-transient roots
            # using only frontier-visible edges. Anchor on the root with the
            # largest reachable component so truly isolated nodes get reported.
            non_transient_roots = [r for r in roots if r.id not in transient_ids]
            if len(non_transient_roots) > 1:
                neighbors: dict[UUID, set[UUID]] = {}
                for edge in frontier_visible:
                    neighbors.setdefault(edge.source_id, set()).add(edge.target_id)
                    neighbors.setdefault(edge.target_id, set()).add(edge.source_id)
                anchor = max(
                    non_transient_roots,
                    key=lambda r: len(self._reachable_undirected(r.id, neighbors)),
                )
                reachable = self._reachable_undirected(anchor.id, neighbors)
                for root in non_transient_roots:
                    if root.id != anchor.id and root.id not in reachable:
                        logger.warning("Reachability gap: %r is isolated from other query roots", root.name)
                        gaps.append(ReachabilityGap(primitive=root))

            surviving = [n for n in all_nodes if n.id in frontier_ids]
            filtered = self._filter_depths(surviving)

            response = EpistemicResponse(
                query=EpistemicQuery(concept_ids=[r.id for r in roots]),
                result=EpistemicResult(primitives=filtered, gaps=gaps, pathway=frontier_visible),
            )
        return response

    def ground(
        self,
        concepts: list[str],
        min_depth: DepthLevel | None = None,
    ) -> GroundingResult:
        """
        Ground concepts against the graph in one step.

        Concepts are matched case-insensitively against graph names by the
        repository; unknown concepts surface as ExistenceGaps in the query
        result. Normalization or synonymy of input is the integrator's
        concern, never the engine's.

        Returns a GroundingResult with grounded=True only when there are zero
        gaps across the entire transitive closure of the submitted concepts,
        not only the query roots: prerequisite knowledge reached through
        transitive relata must itself be grounded for the result to pass.
        `min_depth` is root-scoped and can only raise the floor, never lower
        it; the depth required of transitively reached nodes is set by the
        graph's edge annotations. See GroundingResult for the modeled-versus-
        unmodeled ignorance distinction this enforces.
        """
        if not concepts:
            logger.debug("Ground called with empty concepts")
            # Mirror query()'s empty-case trace so a grounded/ungrounded result always
            # carries a (possibly empty) trace — never None. Grounding then has a single
            # signal, `grounded`, and callers never special-case a missing trace.
            result = GroundingResult(grounded=False, resolved=[], gaps=[], trace=_empty_response())
        else:
            logger.info("Grounding %d concept(s)", len(concepts))
            response = self.query(concepts, min_depth=min_depth)

            # Build a name map from the primitives in the response. Real graph
            # nodes carry their stored canonical casing; transient (unknown)
            # nodes echo the raw input — either way the per-input mapping below
            # is correct. Unknown concepts remain ExistenceGaps in the result.
            canonical_by_lower = {
                p.name_lower: p.name for p in response.result.primitives
            }
            resolved = [canonical_by_lower.get(Primitive.fold_name(c), c) for c in concepts]

            grounded = len(response.result.gaps) == 0
            logger.info(
                "Grounding result: grounded=%s, gaps=%d", grounded, len(response.result.gaps)
            )
            result = GroundingResult(
                grounded=grounded,
                resolved=resolved,
                gaps=response.result.gaps,
                trace=response,
            )
        return result
