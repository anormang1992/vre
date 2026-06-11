"""
Shared fixtures for the VRE test suite.
"""

import pytest

from vre.core.policy.registry import _DEFAULT_REGISTRY


@pytest.fixture(autouse=True)
def _clean_default_policy_registry():
    """
    Keep the module-global policy registry empty around every test.

    Policies register into the global as an import-time side effect; clearing it
    before and after each test prevents cross-test leakage (and the frozen flag that
    VRE construction sets).
    """
    _DEFAULT_REGISTRY.clear()
    yield
    _DEFAULT_REGISTRY.clear()
