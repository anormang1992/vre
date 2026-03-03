# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyCallback Protocol and PolicyCallContext — the user-facing callback contract.

Users implementing custom policy logic should type-annotate against
PolicyCallback and accept a PolicyCallContext argument.

Example::

    from vre.core.policy import PolicyCallback, PolicyCallContext

    class AllowTempWrites:
        def __call__(self, context: PolicyCallContext) -> bool:
            # Return False to suppress the violation, True to let it fire.
            path = context.call_kwargs.get("path", "")
            return not path.startswith("/tmp")
"""

from typing import Any, Protocol

from pydantic import BaseModel

from vre.core.grounding import GroundingResult


class PolicyCallContext(BaseModel):
    """
    Context passed to a policy callback at evaluation time.

    Attributes
    ----------
    tool_name:
        Name of the decorated function that triggered the policy check.
    grounding:
        The GroundingResult from Phase 1, including the full epistemic trace.
    call_args:
        Positional arguments the decorated function was called with.
    call_kwargs:
        Keyword arguments the decorated function was called with.
    """

    model_config = {"arbitrary_types_allowed": True}

    tool_name: str
    grounding: GroundingResult
    call_args: tuple
    call_kwargs: dict[str, Any]


class PolicyCallback(Protocol):
    """
    Protocol for policy callback callables.

    A callback receives the full call context and returns a bool:

    - `True`  — the policy violation fires (confirmation required)
    - `False` — the violation is suppressed (callback vetoed it)

    Implement this Protocol to write custom, domain-specific policy logic.
    """

    def __call__(self, context: PolicyCallContext) -> bool:
        """
        Evaluate the policy against the given call context.

        Return False to suppress the violation, True to let it fire.
        """
        ...
