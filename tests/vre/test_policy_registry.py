"""
Unit tests for PolicyRegistry and the policy_callback / register_policy declaration API.
"""

import pytest

from vre.core.errors import VREError
from vre.core.models import DepthLevel
from vre.core.policy.callback import PolicyCallContext
from vre.core.policy.models import PolicyCallbackResult
from vre.core.policy.registry import (
    _DEFAULT_REGISTRY,
    PolicyRegistry,
    policy_callback,
    register_policy,
)


def _cb(context: PolicyCallContext) -> PolicyCallbackResult:
    return PolicyCallbackResult(passed=True)


def test_register_then_lookup_returns_policy_and_callback():
    """A registered placement is found by (source, target, source_depth) with its policy + callback."""
    reg = PolicyRegistry()
    reg.register(_cb, key="k", source_primitive="delete", target_primitive="file",
                 source_depth=DepthLevel.CONSTRAINTS, name="Guard")
    placements = reg.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    assert len(placements) == 1
    assert placements[0].callback is _cb
    assert placements[0].policy.name == "Guard"
    assert placements[0].policy.callback == "k"


def test_key_defaults_to_callback_name():
    """When no key is given, the callable's __name__ is used as the policy key."""
    reg = PolicyRegistry()
    reg.register(_cb, source_primitive="delete", target_primitive="file",
                 source_depth=DepthLevel.CONSTRAINTS, name="Guard")
    placements = reg.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    assert placements[0].policy.callback == "_cb"


def test_instance_without_key_falls_back_to_class_name():
    """A stateful instance (no __name__) registered without a key keys off its class name."""
    class RateLimiter:
        def __call__(self, context: PolicyCallContext) -> PolicyCallbackResult:
            return PolicyCallbackResult(passed=True)

    reg = PolicyRegistry()
    reg.register(RateLimiter(), source_primitive="send", target_primitive="email",
                 source_depth=DepthLevel.CONSTRAINTS, name="Rate limit")
    placements = reg.placements_for("send", "email", DepthLevel.CONSTRAINTS)
    assert placements[0].policy.callback == "RateLimiter"


def test_duplicate_key_raises_with_teaching_message():
    """A second registration under the same key raises and teaches the multi-edge idiom."""
    reg = PolicyRegistry()
    reg.register(_cb, key="dup", source_primitive="delete", target_primitive="file",
                 source_depth=DepthLevel.CONSTRAINTS, name="A")
    with pytest.raises(VREError, match="distinct"):
        reg.register(_cb, key="dup", source_primitive="delete", target_primitive="directory",
                     source_depth=DepthLevel.CONSTRAINTS, name="B")


def test_placements_lookup_is_case_insensitive():
    """Source/target names match case-insensitively; source_depth must match exactly."""
    reg = PolicyRegistry()
    reg.register(_cb, key="k", source_primitive="Delete", target_primitive="File",
                 source_depth=DepthLevel.CONSTRAINTS, name="Guard")
    assert reg.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    assert not reg.placements_for("delete", "file", DepthLevel.CAPABILITIES)


def test_placements_lookup_folds_like_primitive_identity():
    """Edge keys fold through Primitive.fold_name (NFC + casefold), not .lower(),
    so a policy resolves 'the same name' identically to the primitive it guards.
    'straße' and 'STRASSE' are one primitive (both fold to 'strasse'); a bare
    .lower() would keep them apart ('straße' vs 'strasse') and miss the edge."""
    reg = PolicyRegistry()
    reg.register(_cb, key="k", source_primitive="straße", target_primitive="File",
                 source_depth=DepthLevel.CONSTRAINTS, name="Guard")
    assert reg.placements_for("STRASSE", "file", DepthLevel.CONSTRAINTS)


def test_freeze_blocks_further_registration():
    """Once frozen, register raises and names the import-order rule."""
    reg = PolicyRegistry()
    reg.freeze()
    with pytest.raises(VREError, match="before constructing VRE"):
        reg.register(_cb, key="k", source_primitive="delete", target_primitive="file",
                     source_depth=DepthLevel.CONSTRAINTS, name="Guard")


def test_clear_resets_contents_and_unfreezes():
    """clear() empties placements and lifts the frozen flag (test teardown)."""
    reg = PolicyRegistry()
    reg.register(_cb, key="k", source_primitive="delete", target_primitive="file",
                 source_depth=DepthLevel.CONSTRAINTS, name="Guard")
    reg.freeze()
    reg.clear()
    assert not reg.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    reg.register(_cb, key="k", source_primitive="delete", target_primitive="file",
                 source_depth=DepthLevel.CONSTRAINTS, name="Guard")
    assert reg.placements_for("delete", "file", DepthLevel.CONSTRAINTS)


def test_decorator_returns_original_and_stacks():
    """Stacked decorators register both placements and return the original callable."""
    # The autouse conftest fixture keeps _DEFAULT_REGISTRY empty around each test.
    @policy_callback(key="a", source_primitive="delete", target_primitive="file",
                     source_depth=DepthLevel.CONSTRAINTS, name="A")
    @policy_callback(key="b", source_primitive="delete", target_primitive="directory",
                     source_depth=DepthLevel.CONSTRAINTS, name="B")
    def guard(context):
        return PolicyCallbackResult(passed=True)

    assert guard(None).passed is True  # returns the original callable
    assert _DEFAULT_REGISTRY.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    assert _DEFAULT_REGISTRY.placements_for("delete", "directory", DepthLevel.CONSTRAINTS)


def test_register_policy_imperative_writes_to_default():
    """register_policy writes a placement to the module-global registry."""
    register_policy(_cb, key="imp", source_primitive="send", target_primitive="email",
                    source_depth=DepthLevel.CONSTRAINTS, name="Imp")
    assert _DEFAULT_REGISTRY.placements_for("send", "email", DepthLevel.CONSTRAINTS)


def test_per_registry_decorator_isolates_graphs():
    """Each registry's `.policy_callback` binds only to that registry — no shared global."""
    reg_a, reg_b = PolicyRegistry(), PolicyRegistry()

    @reg_a.policy_callback(key="a", source_primitive="delete", target_primitive="file",
                           source_depth=DepthLevel.CONSTRAINTS, name="A")
    def guard_a(context):
        return PolicyCallbackResult(passed=True)

    @reg_b.policy_callback(key="b", source_primitive="send", target_primitive="email",
                           source_depth=DepthLevel.CONSTRAINTS, name="B")
    def guard_b(context):
        return PolicyCallbackResult(passed=True)

    assert reg_a.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    assert not reg_a.placements_for("send", "email", DepthLevel.CONSTRAINTS)  # B's policy isn't in A
    assert reg_b.placements_for("send", "email", DepthLevel.CONSTRAINTS)
    assert not reg_b.placements_for("delete", "file", DepthLevel.CONSTRAINTS)
    assert _DEFAULT_REGISTRY.keys() == []  # the global was never touched
