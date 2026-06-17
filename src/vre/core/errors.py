# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
VRE exception hierarchy.

All VRE-specific exceptions derive from VREError so integrators can catch
at the desired granularity — from a single error type up to the entire
framework.

VRE's responsibility is to roll back any in-memory mutations and re-raise
errors with clear, typed exceptions. Integrators decide recovery strategy.
"""


class VREError(Exception):
    """Base exception for all VRE errors."""


class GraphError(VREError):
    """A graph backend operation failed (read or write)."""


class PersistenceError(GraphError):
    """A write operation against the graph backend failed."""


class GraphIntegrityError(VREError):
    """A graph operation would violate structural integrity constraints."""


class CyclicRelationshipError(GraphIntegrityError):
    """An edge would create a cycle on transitive relationship types."""


class HydrationError(VREError):
    """Failed to reconstruct a domain object from stored data."""


class ProvenanceError(VREError):
    """Knowledge being persisted is missing required provenance.

    Provenance is required on every primitive, depth, and relatum at the
    persistence boundary (CLAUDE.md §7.2). Raised by validate_provenance,
    which the backends invoke before any save.
    """


class CandidateValidationError(VREError):
    """A learning candidate is missing required fields or references invalid data."""


class GapResolvedError(VREError):
    """A knowledge gap was already resolved by the time learn_gap ran.

    The live primitive is grounded to (or beyond) the depth the gap required, so
    there is nothing left to learn — and persisting would overwrite grounded
    knowledge. This is a benign state divergence, NOT a malformed candidate: the
    candidate was fine, the graph moved underneath it (a concurrent learn round,
    a seeder, a sibling gap that cascaded). Integrators should treat it as
    "already done" — re-ground and proceed — not as a failure to retry.

    Deliberately a sibling of CandidateValidationError, never a subclass, so a
    well-behaved agent is not told it erred.

    Message-only, like the rest of the hierarchy: the message already names the
    concept and depths. Structured fields (primitive id, depths) were dropped as
    unused — re-add them the day a real consumer (telemetry, an integrator API)
    needs to inspect the divergence programmatically.
    """


class RegistryError(VREError):
    """A file-based registry operation failed (read, write, or corruption)."""


class PolicyPlacementError(VREError):
    """A declared policy references an APPLIES_TO edge absent from the graph."""


class SchemaVersionError(VREError):
    """The persisted schema version is newer than this build of VRE can read.

    The on-disk marker records the format the data was written in; the code's
    CURRENT_SCHEMA_VERSION records what this build knows how to read. When the
    disk value is the larger of the two, the store was written by a newer VRE
    and cannot be safely read through this build's assumptions — so we refuse
    loudly rather than risk silent corruption.
    """
