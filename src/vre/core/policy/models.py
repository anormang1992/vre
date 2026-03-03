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

if TYPE_CHECKING:
    from vre.core.policy.callback import PolicyCallback


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
        it receives a PolicyCallContext and returns bool. Return False to
        suppress the violation; True (or no callback) means the policy fires.
        """
        if self.callback is None:
            return None
        module_path, _, func_name = self.callback.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, func_name)


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
    requires_confirmation: bool
    message: str


class PolicyResult(BaseModel):
    """
    Result of a VRE policy evaluation.

    `action` is one of "PASS", "PENDING", or "BLOCK".
    """

    action: str
    reason: str | None = None
    confirmation_message: str | None = None

    def __str__(self) -> str:
        """
        Render the policy result as a human-readable status string.
        """
        if self.action == "PASS":
            return "[VRE Policy] PASSED"
        if self.action == "PENDING":
            return f"[VRE Policy] PENDING — {self.confirmation_message}"
        if self.action == "BLOCK":
            return f"[VRE Policy] BLOCKED — {self.reason}"
        return f"[VRE Policy] {self.action}"
