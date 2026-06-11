# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Policy data models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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

    @classmethod
    def unevaluable(cls, message: str) -> "PolicyCallbackResult":
        """
        Result for a callback that could not be evaluated — fails closed, carrying
        the caller-supplied reason it could not run (a missing tool call is one
        such reason, but not the only one).
        """
        return cls(passed=False, message=message)


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


DEFAULT_CONFIRMATION_MESSAGE = "This action requires confirmation. Proceed?"


class Policy(BaseModel):
    name: str
    requires_confirmation: bool = True
    trigger_cardinality: Cardinality | None = None  # None = always fires
    callback: str | None = None  # registry key for the integrator-registered callback (never an import path)
    confirmation_message: str = DEFAULT_CONFIRMATION_MESSAGE
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyViolation(BaseModel):
    """
    A single triggered policy violation produced by PolicyGate.
    """

    policy: Policy
    callback_result: PolicyCallbackResult | None = None

    @property
    def message(self) -> str:
        """
        Human-readable reason the violation fired. The callback's message (when it
        fired or could not be evaluated) leads, followed by the policy's
        confirmation message; the confirmation message alone when there is no
        callback reason.

        Deliberately a derived property — NOT a stored field or `@computed_field`.
        Keeping it out of the serialized form means it can never drift from its
        inputs (`callback_result.message` + `policy.confirmation_message`), which
        are themselves serialized and so fully reconstruct it.
        """
        if self.callback_result is not None and self.callback_result.message:
            message = f"{self.callback_result.message}\n{self.policy.confirmation_message}"
        else:
            message = self.policy.confirmation_message
        return message

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
            msg = "[VRE Policy] PASSED"
        else:
            msg = f"[VRE Policy] BLOCKED — {self.reason}"
        return msg
