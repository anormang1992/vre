# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
vre_guard — decorator-based epistemic enforcement.

Usage::

    from vre.guard import vre_guard

    @vre_guard(vre, concepts=["write", "file"])
    def write_file(path: str, text: str) -> str:
        ...

Behaviour
---------
Each call runs grounding → policy → execution in a single pass:

1. VRE grounding is checked at D3.
2. `on_trace` is fired (if provided) with the `GroundingResult`.
3. If grounding fails, returns `GroundingResult` immediately — the function
   is *not* called.
4. Policy is evaluated. If PENDING: `on_policy` is consulted (or blocked if
   absent). If BLOCK: returns `PolicyResult`.
5. Otherwise, the original function is called and its return value is returned.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

from vre.core.models import DepthLevel
from vre.core.policy import PolicyResult
from vre.core.policy.callback import PolicyCallContext

if TYPE_CHECKING:
    from vre import VRE
    from vre.core.grounding import GroundingResult

# `concepts` and `cardinality` may be static values or callables that receive
# the same (*args, **kwargs) as the decorated function and return the value
# dynamically at call time.
ConceptsInput = list[str] | Callable[..., list[str]]
CardinalityInput = str | None | Callable[..., str | None]


def vre_guard(
    vre: "VRE",
    concepts: ConceptsInput,
    cardinality: CardinalityInput = None,
    min_depth: DepthLevel | None = None,
    on_trace: Callable[["GroundingResult"], None] | None = None,
    on_policy: Callable[[str], bool] | None = None,
) -> Callable:
    """
    Decorator that gates a function behind VRE grounding and policy checks.

    Parameters
    ----------
    vre:
        VRE instance to use for grounding and policy checks.
    concepts:
        Concept names the function touches. Accepts static list or a callable
        that receives (*args, **kwargs) and returns list[str].
    cardinality:
        Optional cardinality hint for policy evaluation ("single", "multiple").
        Accepts a static string or a callable that receives (*args, **kwargs)
        and returns str | None.
    min_depth:
        Optional integrator override — enforces a minimum depth floor on all
        root primitives. Can only raise the floor, never lower it.
    on_trace:
        Optional callback called with the GroundingResult after grounding
        (both grounded and ungrounded).
    on_policy:
        Optional callback called with the confirmation message when a policy
        requires confirmation. Should return True to proceed, False to block.
    """
    def decorator(fn: Callable) -> Callable:
        """
        Bind the guard to a specific function.
        """
        tool_name = fn.__name__

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            """
            Run grounding → policy → execution on each call.
            """
            resolved_concepts = concepts(*args, **kwargs) if callable(concepts) else concepts
            grounding = vre.check(resolved_concepts, min_depth=min_depth)
            if on_trace:
                on_trace(grounding)

            if not grounding.grounded:
                return grounding

            resolved_cardinality = (
                cardinality(*args, **kwargs) if callable(cardinality) else cardinality
            )
            context = PolicyCallContext(
                tool_name=tool_name,
                grounding=grounding,
                call_args=args,
                call_kwargs=kwargs,
            )

            policy = vre.check_policy(grounding, resolved_cardinality, context)
            if policy.action == "PENDING":
                if on_policy:
                    if not on_policy(policy.confirmation_message or ""):
                        return PolicyResult(action="BLOCK", reason="User declined")
                else:
                    return PolicyResult(
                        action="BLOCK", reason="Confirmation required, no handler"
                    )
            if policy.action == "BLOCK":
                return policy
            return fn(*args, **kwargs)

        wrapped._vre_concepts = concepts  # type: ignore[attr-defined]
        return wrapped

    return decorator
