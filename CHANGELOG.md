# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, minor version bumps may introduce breaking changes.

## [0.12.0] - 2026-06-12

### Changed

- **BREAKING (storage):** The serialized `Provenance` payload is now persisted
  under the `provenance_json` property/column on both the Neo4j and SQLite
  backends, replacing the bare `provenance` name. This aligns provenance with
  the existing `*_json` suffix convention (`depths_json`, `metrics_json`,
  `metadata_json`), so a property name reliably tells a reader whether the value
  is a native scalar or a JSON blob requiring `json.loads()`. The Pydantic
  surface (`Primitive.provenance`, `Depth.provenance`, `Relatum.provenance`) is
  unchanged — only on-disk names changed. The nested `provenance` key inside
  `depths_json` is intentionally left unsuffixed (it is already inside a JSON
  blob). (#48)
- Provenance semantics are now documented for the knowledge-linter model:
  `AUTHORED` means a human drafted the content directly; `LEARNED` means an agent
  proposed it and a human approved it at the persistence boundary. Both are
  human-attested by construction — provenance is genealogy, not a trust
  gradient. (#91)

### Removed

- **BREAKING:** `ProvenanceSource.CONVERSATIONAL` has been removed. It only had
  meaning when VRE owned the learning loop; as a knowledge linter VRE cannot
  assign or enforce it, so the enum advertised a distinction the system can no
  longer make. New records use `LEARNED` (agent-proposed, human-approved) or
  `AUTHORED` (human-drafted). (#91)
- The SQLite `_warn_if_legacy_policies()` startup aid — a pre-1.0 crutch for the
  #81 policy removal — for consistency with the clean-break stance below.

### Migration

These are pre-1.0 breaking changes to on-disk storage. There is **no migration
path**: rebuild the graph by re-seeding it. No automated migration script is
provided, as no production graphs are expected at this stage.
