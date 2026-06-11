# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
PolicyRegistry — the in-memory home for code-resident policies.

Policies are declared in the integrator's own (imported) Python via the
`policy_callback` decorator, or the imperative `register_policy` for stateful
callables — never persisted to the graph. Each declaration binds a callable to one
APPLIES_TO edge, identified by source primitive, target primitive, and the source
depth the edge lives at.

    from vre import policy_callback, DepthLevel

    @policy_callback(
        key="protected_file",
        source_primitive="delete",
        target_primitive="file",
        source_depth=DepthLevel.CONSTRAINTS,
        name="Protected file guard",
        confirmation_message="Deleting a protected file. Proceed?",
    )
    def protected_file(ctx): ...

To attach one callable to several edges, stack decorators, each with a distinct key.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from pydantic import BaseModel

from vre.core.errors import VREError
from vre.core.models import DepthLevel
from vre.core.policy.callback import PolicyCallback
from vre.core.policy.models import (
    Cardinality,
    DEFAULT_CONFIRMATION_MESSAGE,
    Policy,
    PolicyCallbackResult,
)


class PolicyPlacement(BaseModel):
    """
    A single code-resident policy bound to one APPLIES_TO edge.

    `policy` carries the declaration's identity and confirmation semantics;
    `callback` is the live callable the gate invokes. The edge is identified by
    `source` -> `target` at `source_depth`.
    """

    model_config = {"frozen": True}

    key: str
    policy: Policy
    callback: Callable[..., PolicyCallbackResult]
    source: str
    target: str
    source_depth: DepthLevel


class OrphanedPlacement(BaseModel):
    """
    A declared placement that resolves to no APPLIES_TO edge in the graph.

    Returned by `VRE.validate_policy_placements()` and enumerated in the
    `PolicyPlacementError` raised at construction — a declared gate that would protect
    nothing.
    """

    model_config = {"frozen": True}

    key: str
    name: str
    source: str
    target: str
    source_depth: DepthLevel
    reason: str


class PolicyRegistry:
    """
    In-memory store of declared policy placements, indexed for gate lookup.

    Registration is import-time and one-way: `freeze()` (called by `VRE.__init__`)
    makes the registry immutable — no policy may be registered after construction. When
    VRE validated the placements first (its default), the freeze also upholds the
    invariant `everything enforced was validated`.
    """

    def __init__(self) -> None:
        """
        Create an empty, unfrozen registry.
        """
        self._by_key: dict[str, PolicyPlacement] = {}
        self._by_edge: dict[tuple[str, str, DepthLevel], list[PolicyPlacement]] = {}
        self._frozen: bool = False

    @staticmethod
    def _edge_key(source: str, target: str, source_depth: DepthLevel) -> tuple[str, str, DepthLevel]:
        """
        Normalize an edge identity to its case-insensitive lookup key.
        """
        return (source.lower(), target.lower(), source_depth)

    def register(
        self,
        callback: PolicyCallback,
        *,
        key: str | None = None,
        source_primitive: str,
        target_primitive: str,
        source_depth: DepthLevel,
        name: str,
        confirmation_message: str = DEFAULT_CONFIRMATION_MESSAGE,
        trigger_cardinality: Cardinality | None = None,
        requires_confirmation: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyCallback:
        """
        Bind `callback` to the `source_primitive` -> `target_primitive` edge at
        `source_depth`, returning the callable unchanged (so decorators stack).

        `key` defaults to the callable's `__name__`. A duplicate key raises, and a
        frozen registry refuses all registration.
        """
        if self._frozen:
            raise VREError(
                "cannot register policies after VRE construction; "
                "import policy modules before constructing VRE"
            )
        # Functions carry __name__; stateful instances (register_policy's reason to exist)
        # do not, so fall back to the class name.
        key = key or getattr(callback, "__name__", None) or type(callback).__name__
        if key in self._by_key:
            raise VREError(
                f"key {key!r} already registered — to attach one callback to multiple "
                f"edges, give each placement an explicit distinct key "
                f"(e.g. key='protected_file', key='protected_dir')."
            )
        placement = PolicyPlacement(
            key=key,
            policy=Policy(
                name=name,
                callback=key,
                requires_confirmation=requires_confirmation,
                trigger_cardinality=trigger_cardinality,
                confirmation_message=confirmation_message,
                metadata=metadata or {},
            ),
            callback=callback,
            source=source_primitive,
            target=target_primitive,
            source_depth=source_depth,
        )
        self._by_key[key] = placement
        self._by_edge.setdefault(
            self._edge_key(source_primitive, target_primitive, source_depth), []
        ).append(placement)
        return callback

    def policy_callback(
        self,
        key: str | None = None,
        *,
        source_primitive: str,
        target_primitive: str,
        source_depth: DepthLevel,
        name: str,
        confirmation_message: str = DEFAULT_CONFIRMATION_MESSAGE,
        trigger_cardinality: Cardinality | None = None,
        requires_confirmation: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[PolicyCallback], PolicyCallback]:
        """
        Decorator form of `register`, binding the decorated callable to THIS registry.

        Returns the callable unchanged, so decorators stack (one per edge, distinct
        keys). Use a dedicated registry per graph for multi-graph / multi-tenant setups
        (`@my_registry.policy_callback(...)`), then pass it to that VRE via
        `policy_registry=`.
        """
        def decorate(callback: PolicyCallback) -> PolicyCallback:
            """Register `callback` on its declared edge and return it unchanged."""
            return self.register(
                callback,
                key=key,
                source_primitive=source_primitive,
                target_primitive=target_primitive,
                source_depth=source_depth,
                name=name,
                confirmation_message=confirmation_message,
                trigger_cardinality=trigger_cardinality,
                requires_confirmation=requires_confirmation,
                metadata=metadata,
            )
        return decorate

    def placements_for(
        self, source_name: str, target_name: str, source_depth: DepthLevel
    ) -> list[PolicyPlacement]:
        """
        Return the placements bound to the `source_name` -> `target_name` edge at
        `source_depth` (case-insensitive on names); empty if none.
        """
        return list(self._by_edge.get(self._edge_key(source_name, target_name, source_depth), ()))

    def iter_placements(self) -> Iterable[PolicyPlacement]:
        """
        Iterate every registered placement, in registration order.
        """
        return tuple(self._by_key.values())

    def keys(self) -> list[str]:
        """
        The registered policy keys, in registration order.
        """
        return list(self._by_key)

    def __len__(self) -> int:
        """
        The number of registered placements.
        """
        return len(self._by_key)

    def freeze(self) -> None:
        """
        Make the registry immutable. Idempotent.
        """
        self._frozen = True

    def clear(self) -> None:
        """
        Drop all placements and lift the frozen flag (test teardown).
        """
        self._by_key.clear()
        self._by_edge.clear()
        self._frozen = False


_DEFAULT_REGISTRY = PolicyRegistry()


# Module-level conveniences over a hidden default registry — the same pattern the
# standard-library `random` module uses (its functions are methods of a default
# instance). `policy_callback` is the decorator; `register_policy` is its imperative
# twin for stateful callables. For multiple graphs in one process, create a
# `PolicyRegistry()` per graph, decorate with its `.policy_callback`, and pass each to
# that VRE via `policy_registry=` — instantiating one VRE freezes only its own registry.
policy_callback = _DEFAULT_REGISTRY.policy_callback
register_policy = _DEFAULT_REGISTRY.register
