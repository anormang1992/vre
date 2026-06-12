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
    """

    def __init__(
        self,
        message: str,
        *,
        primitive_id=None,
        name=None,
        current_depth=None,
        required_depth=None,
    ) -> None:
        super().__init__(message)
        self.primitive_id = primitive_id
        self.name = name
        self.current_depth = current_depth
        self.required_depth = required_depth


class RegistryError(VREError):
    """A file-based registry operation failed (read, write, or corruption)."""


class PolicyPlacementError(VREError):
    """A declared policy references an APPLIES_TO edge absent from the graph."""
