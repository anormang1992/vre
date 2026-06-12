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
Each call runs grounding -> policy -> execution in a single pass:

1. VRE grounding is checked (depth derived from graph structure).
2. `on_trace` is fired (if provided) with the `GroundingResult`.
3. If grounding fails, returns a `GuardBlock` immediately — the function
   is *not* called.
4. Policy is evaluated. If BLOCK: returns a `GuardBlock`.
5. Otherwise, the original function is called and its return value is
   returned untouched.

The guard returns a `GuardBlock` only when it intervened (grounding failed or
policy blocked); on success it returns the wrapped function's own value. So a
plain-Python caller distinguishes the two outcomes with a single
`isinstance(..., GuardBlock)` check, while an LLM caller can `str()` either.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, ParamSpec, TypeVar

from vre.core.models import DepthLevel
from vre.core.policy import PolicyAction
from vre.core.policy.callback import ToolCallContext
from vre.core.policy.models import PolicyViolation

if TYPE_CHECKING:
    from vre import VRE
    from vre.core.grounding import GroundingResult
    from vre.core.policy.models import PolicyResult

# `concepts` and `cardinality` may be static values or callables that receive
# the same (*args, **kwargs) as the decorated function and return the value
# dynamically at call time.
ConceptsInput = list[str] | Callable[..., list[str]]
CardinalityInput = str | None | Callable[..., str | None]

# The wrapped function's parameter list (P) and return type (R) are preserved
# through the guard: a caller keeps full argument type-checking and sees the
# result as `GuardBlock | R`, where `isinstance(result, GuardBlock)` narrows to R.
P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardBlock:
    """
    Returned by `vre_guard` when the guard blocked execution — grounding failed,
    or a policy returned BLOCK. The wrapped function was *not* called.

    The guard returns this VRE-owned type only when VRE intervened; on success it
    returns the wrapped function's own value untouched. So a plain-Python caller
    distinguishes the two outcomes with a single, unambiguous check::

        result = guarded_fn(...)
        if isinstance(result, GuardBlock):
            ...   # blocked — inspect `.grounding` / `.policy` / `.blocked_by`
        else:
            ...   # ran — `result` is the function's own return value

    `__str__` delegates to the wrapped result, so a block still renders as a full
    explanation for an LLM caller.
    """

    grounding: GroundingResult | None = None
    policy: PolicyResult | None = None

    def __post_init__(self) -> None:
        """
        Enforce the invariant: a block wraps exactly one result. An empty or
        double-wrapped GuardBlock would let `blocked_by` and `__str__` silently
        misreport which layer intervened — precisely the quiet degradation VRE
        exists to prevent.
        """
        if (self.grounding is None) == (self.policy is None):
            raise ValueError("GuardBlock must wrap exactly one of grounding/policy")

    @property
    def blocked_by(self) -> Literal["grounding", "policy"]:
        """
        Which layer blocked — 'grounding' or 'policy'.
        """
        return "grounding" if self.grounding is not None else "policy"

    def __str__(self) -> str:
        """
        Delegate to the wrapped result's explanation — keeps a block LLM-feedable.
        """
        return str(self.grounding if self.grounding is not None else self.policy)


def vre_guard(
    vre: "VRE",
    concepts: ConceptsInput,
    cardinality: CardinalityInput = None,
    min_depth: DepthLevel | None = None,
    on_trace: Callable[["GroundingResult"], None] | None = None,
    on_policy: Callable[[list[PolicyViolation]], bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, GuardBlock | R]]:
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
        (both grounded and ungrounded). Exceptions it raises are logged and
        swallowed — an observability hook must never break enforcement.
    on_policy:
        Optional callback called with the list of PolicyViolation when any
        violation requires confirmation. Should return True to proceed,
        False to block.

    Returns
    -------
    On success — grounded and policy passed — returns the wrapped function's own
    return value, untouched.

    When the guard blocks — grounding failed, or a policy returned BLOCK —
    returns a `GuardBlock` instead, and the wrapped function is *not* called.
    `GuardBlock` renders via `str()` as a full explanation, so an LLM caller can
    feed it straight back to the model. A plain-Python caller distinguishes the
    two outcomes with a single check::

        result = guarded_fn(...)
        if isinstance(result, GuardBlock):
            ...   # blocked — inspect .grounding / .policy / .blocked_by
        else:
            ...   # ran — `result` is the function's own return value
    """
    def decorator(fn: Callable[P, R]) -> Callable[P, GuardBlock | R]:
        """
        Bind the guard to a specific function.
        """
        tool_name = fn.__name__

        @functools.wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> GuardBlock | R:
            """
            Run grounding -> policy -> execution on each call.
            """
            resolved_concepts = concepts(*args, **kwargs) if callable(concepts) else concepts
            logger.info("Guard %r: grounding %d concept(s)", tool_name, len(resolved_concepts))
            logger.debug("Guard %r: concepts %s", tool_name, resolved_concepts)
            grounding = vre.check(resolved_concepts, min_depth=min_depth)
            if on_trace:
                try:
                    on_trace(grounding)
                except Exception:  # noqa: BLE001 — observability must not break enforcement
                    logger.exception("Guard %r: on_trace raised; continuing", tool_name)

            if not grounding.grounded:
                logger.info("Guard %r: not grounded, returning GuardBlock (%d gaps)", tool_name, len(grounding.gaps))
                result = GuardBlock(grounding=grounding)
            else:
                # Policy evaluation
                resolved_cardinality = (
                    cardinality(*args, **kwargs) if callable(cardinality) else cardinality
                )
                tool_call = ToolCallContext(
                    tool_name=tool_name,
                    call_args=args,
                    call_kwargs=kwargs,
                )

                policy = vre.check_policy(
                    grounding, resolved_cardinality, tool_call, on_policy=on_policy,
                )

                if policy.action == PolicyAction.BLOCK:
                    logger.info("Guard %r: policy BLOCKED — %s", tool_name, policy.reason)
                    result = GuardBlock(policy=policy)
                else:
                    logger.debug("Guard %r: grounded and policy passed, executing function", tool_name)
                    result = fn(*args, **kwargs)

            return result

        wrapped._vre_concepts = concepts  # type: ignore[attr-defined]
        return wrapped

    return decorator
