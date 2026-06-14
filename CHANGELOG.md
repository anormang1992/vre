# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, minor version bumps may introduce breaking changes.

Sections for released versions are backfilled from tagged GitHub releases and
summarize major changes, refactors, and breaking changes — not every commit.

## [Unreleased]

### Added

- Policy callback context types: `ToolCallContext`, `GroundingContext`, and a
  per-edge `TriggeringEdge`. Callback failure reasons now surface in the policy
  violation message. (#58)
- `GapResolvedError` (a sibling of `CandidateValidationError`, exported from `vre`
  and `vre.core`): `learn_gap` raises it when the live graph already satisfies the
  gap — the concept an existence gap names already exists, or a depth/relational
  gap's primitive is already grounded to the required depth. The gap closed
  underneath a stale snapshot, so there is nothing to learn and persisting would
  duplicate or overwrite grounded knowledge. Distinct from a malformed candidate so
  integrators can treat it as "already done" rather than a failure. (#95)
- `DepthGap.missing_levels` / `RelationalGap.missing_levels`: the exact levels a
  fill must author (the holes in `(current, required]` not already present).
  `template_for_gap` now pre-seeds depth and relational candidates with one empty
  slot per missing level, so an integrator fills only the `properties` — VRE
  resolves *which* levels are missing, and a dormant detached level (e.g. a D4 over
  a D1 chain) is never offered for re-authoring. (#95)

### Changed

- **BREAKING (storage):** The serialized `Provenance` payload is now persisted
  under the `provenance_json` property/column on both the Neo4j and SQLite
  backends, replacing the bare `provenance` name, to follow the `*_json` suffix
  convention (`depths_json`, `metrics_json`, `metadata_json`). The Pydantic
  surface (`Primitive`/`Depth`/`Relatum.provenance`) is unchanged, and the nested
  `provenance` key inside `depths_json` stays unsuffixed (already inside a JSON
  blob). (#48)
- **BREAKING:** `vre_guard` now returns a `GuardBlock` when grounding or policy
  blocks a call, and passes the wrapped function's return value through unchanged
  on success (transparent gate). (#88)
- **BREAKING:** Policy evaluation now takes a composed `PolicyCallContext`
  (tool-call context + per-edge `TriggeringEdge`); the gate fails closed when no
  `tool_call` is supplied. (#58)
- Policy `confirmation_message` is used verbatim; the `{action}` interpolation
  was dropped.
- Provenance semantics documented for the knowledge-linter model: `AUTHORED` =
  a human drafted the content directly; `LEARNED` = an agent proposed it and a
  human approved it at the persistence boundary. Both are human-attested by
  construction — provenance is genealogy, not a trust gradient. (#91)
- Learning candidate validation is hardened and now enforced against **live**
  graph state at the persistence gate rather than the gap snapshot (which can go
  stale before `learn_gap` runs). Depth fills must stay within
  `current < level <= required` — no overwriting already-grounded depths, no
  escalating past the depth the gap asked for — and may not name the same level
  twice. Validation now reads the live primitive's full level set: a fill may not
  re-author a level that is already present (even one detached above the contiguous
  max), and contiguity is checked over `existing ∪ proposed`, so the fill supplies
  only the genuine holes and never overwrites grounded knowledge to satisfy the
  chain. `ExistenceCandidate` must match the gapped concept's name and supply a D1
  (IDENTITY) depth, and the existence persist path now checks for the concept
  first (closing the duplicate-node gap). Reachability prerequisites and edge
  placement gate on contiguous depth, not exact level membership, so an edge can
  no longer be placed where grounding would never see it. (#95)

### Removed

- **BREAKING:** Graph-resident policies. Policies are now **code-resident** —
  declared via `@policy_callback` / `register_policy` and never persisted, with
  fail-loud validation at `VRE()` init. The policy wizard, `resolve_callback`,
  `parse_policy`, and `Relatum.policies` were removed. (#81, #93, #97)
- **BREAKING:** `ProvenanceSource.CONVERSATIONAL`. New records use `LEARNED`
  (agent-proposed, human-approved) or `AUTHORED` (human-drafted). (#91)
- The SQLite `_warn_if_legacy_policies()` startup aid — a pre-1.0 crutch for the
  #81 policy removal — for consistency with the clean-break stance.
- **BREAKING:** The `source` parameter of `LearningEngine.learn_gap`. Knowledge
  persisted through the learning path is always stamped `LEARNED` (agent-proposed,
  human-approved); `AUTHORED` provenance can no longer be forged there. (#95)
- The `examples/langchain_ollama` reference agent and the `examples` install
  extra. The demo leaned on a framework that hid where the guard sits and a
  `shell=True` pseudo-sandbox; it is superseded by a single agent-driven showcase.
  The learning-loop pattern it demonstrated is now documented inline in the
  README. (#114)

### Performance

- `resolve_subgraph` hits the `NOCASE` name index, and the transitive cycle check
  is batched into a single query per save (SQLite).

### Migration

These are pre-1.0 breaking changes to on-disk storage and the public API. There
is **no migration script**: rebuild the graph by re-seeding it.

## [0.9.1] - 2026-06-08

First release since 0.8.0. Ships the previously unreleased 0.9.0 work (a breaking
removal of concept resolution) alongside fixes for crashes affecting 0.8.0 users
on SQLite-only installs.

### Removed

- **BREAKING:** Concept resolution removed entirely. Membership is now exact,
  case-insensitive name matching at the repository layer; unknown concepts surface
  as `ExistenceGap`s. Removed `ConceptResolver`, `VRE.resolve()`, `VRE.resolver`,
  and `ResolutionError`. Normalization/lemmatization/synonymy is now the
  integrator's concern.
- **BREAKING:** `spacy` and `click` dependencies dropped; installs no longer
  download a spaCy language model.

### Changed

- **BREAKING:** The policy wizard is now an importable helper (`run_wizard(repo)`),
  not a `python -m vre.core.policy.wizard` CLI, and works with any backend.

### Fixed

- Seeders and the policy wizard crashed on SQLite-only installs due to an
  unconditional `Neo4jRepository` import; the import is now guarded.
- The learning engine silently dropped mismatched gap/candidate pairs (e.g. an
  `ExistenceGap` fed a `DepthCandidate`); it now raises `CandidateValidationError`.

## [0.8.0] - 2026-05-30

### Added

- **SQLite backend** (`SQLiteRepository`): a zero-external-dependency backend
  (file- or memory-backed) with recursive-CTE transitive traversal, cycle
  detection, and full `resolve_subgraph` parity with the Neo4j backend.
- `Repository` ABC defining the persistence contract; `upsert_primitive()` for
  idempotent seeding; `clear()`.
- `seeders/` package and shared `scripts/` backend helpers (`add_backend_args`,
  `make_repository`).

### Changed

- **BREAKING:** `PrimitiveRepository` renamed to `Neo4jRepository` and moved from
  `vre.core.graph` to `vre.core.backends.neo4j`.
- **BREAKING:** Neo4j is now an optional dependency (`pip install vre[neo4j]`).
- **BREAKING:** `ensure_constraints()` is now private (`_ensure_constraints`) and
  called automatically.
- **BREAKING:** `GroundingResult.gaps` narrowed from `list[Any]` to
  `list[KnowledgeGap]`.
- Scripts and examples default to the SQLite backend.

### Fixed

- `GroundingResult.__str__` no longer prints a misleading "verified at D3" message.

## [0.6.0] - 2026-05-08

Reframes VRE as a knowledge linter: identify gaps, validate fills, persist
accepted knowledge. Loop orchestration and decision tracking become the
integrator's responsibility.

### Added

- `learning_engine.learn_gap(gap, candidate)` as the single explicit persistence
  entry point, plus `reachability_prerequisites()` to surface missing depths as
  structured data (no nested callbacks).

### Changed

- **BREAKING:** `ReachabilityCandidate` gains an explicit `source_name`, so edges
  can be placed in either direction. Validation moved onto candidate models
  (`validate_for_gap`); `Primitive.contiguous_max_depth` is now a property; trace
  persistence drops the `learn` operation type.

### Removed

- **BREAKING:** The auto-learning loop and its surface — `VRE.learn_all()`,
  `vre_guard(on_learn=...)`, `LearningCallback`, `CandidateDecision`,
  `LearningResult`, learning metrics, and the nested `_learn_missing_depths` flow.

## [0.4.3] - 2026-04-14

Initial PyPI release, bundling the early epistemic engine and tooling.

### Added

- Depth-gated relata traversal; provenance as a first-class attribute; policy
  callbacks (`PolicyAction` enum, per-violation callbacks); LLM-based concept
  extraction with a two-pass hook protocol; cycle detection on transitive
  relationships; an engine-to-integrator error contract; structured logging;
  agent identity in traces; aggregate usage metrics on `Primitive` nodes;
  grounding-trace persistence to daily JSONL files; PyPI publishing via GitHub
  Actions.

> Note: the auto-learning loop introduced here was removed in 0.6.0.

## [0.2.0] - 2026-03-04

### Added

- Claude Code integration and PR/issue templates.

### Changed

- License changed from MIT to Apache-2.0.

## [0.1.0] - 2026-03-02

- Initial public scaffold: the core epistemic model and project README.

[Unreleased]: https://github.com/anormang1992/vre/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/anormang1992/vre/compare/v0.8.0...v0.9.1
[0.8.0]: https://github.com/anormang1992/vre/compare/v0.6.0...v0.8.0
[0.6.0]: https://github.com/anormang1992/vre/compare/v0.4.3...v0.6.0
[0.4.3]: https://github.com/anormang1992/vre/compare/v0.2.0...v0.4.3
[0.2.0]: https://github.com/anormang1992/vre/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anormang1992/vre/releases/tag/v0.1.0
