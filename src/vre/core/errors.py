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


class RegistryError(VREError):
    """A file-based registry operation failed (read, write, or corruption)."""


class PolicyPlacementError(VREError):
    """A declared policy references an APPLIES_TO edge absent from the graph."""
