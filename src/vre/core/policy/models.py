# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Policy data models.
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from vre.core.errors import VREError

if TYPE_CHECKING:
    from vre.core.policy.callback import PolicyCallback


class PolicyCallbackResult(BaseModel):
    """
    Result returned by a PolicyCallback.

    Attributes
    ----------
    passed:
        True if the action passes the policy (no violation), False if it fails (violation fires).
    message:
        Optional human-readable message providing context about the decision.
    """

    passed: bool
    message: str | None = None


class PolicyAction(str, Enum):
    """
    Outcome of a policy evaluation — PASS or BLOCK.
    """

    PASS = "PASS"
    BLOCK = "BLOCK"


class Cardinality(str, Enum):
    """
    Cardinality hint passed to policy evaluation — "single" or "multiple" target.
    """

    SINGLE = "single"
    MULTIPLE = "multiple"


class Policy(BaseModel):
    name: str
    requires_confirmation: bool = True
    trigger_cardinality: Cardinality | None = None  # None = always fires
    callback: str | None = None  # dotted path to a PolicyCallback callable
    confirmation_message: str = "This action requires confirmation. Proceed?"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_callback(self) -> PolicyCallback | None:
        """
        Resolve the dotted-path callback string to a callable.

        The returned callable must conform to the PolicyCallback Protocol:
        it receives a PolicyCallContext and returns PolicyCallbackResult.
        A result with passed=True suppresses the violation; passed=False
        (or no callback) means the policy fires.
        """
        if self.callback is None:
            return None
        module_path, _, func_name = self.callback.rpartition(".")
        if not module_path or not func_name:
            raise VREError(
                f"Invalid policy callback path '{self.callback}': expected 'module.attr'"
            )
        try:
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise VREError(
                f"Failed to resolve policy callback '{self.callback}': {exc}"
            ) from exc


def parse_policy(data: dict[str, Any]) -> Policy:
    """
    Hydrate a Policy from a dict (e.g. deserialized from JSON).
    """
    return Policy.model_validate(data)


class PolicyViolation(BaseModel):
    """
    A single triggered policy violation produced by PolicyGate.
    """

    policy: Policy
    message: str
    callback_result: PolicyCallbackResult | None = None

    @property
    def requires_confirmation(self) -> bool:
        """
        Whether the originating policy requires human confirmation.
        """
        return self.policy.requires_confirmation


class PolicyResult(BaseModel):
    """
    Result of a VRE policy evaluation.
    """

    action: PolicyAction
    reason: str | None = None
    violations: list[PolicyViolation] = Field(default_factory=list)

    def __str__(self) -> str:
        """
        Render the policy result as a human-readable status string.
        """
        if self.action == PolicyAction.PASS:
            return "[VRE Policy] PASSED"
        return f"[VRE Policy] BLOCKED — {self.reason}"
