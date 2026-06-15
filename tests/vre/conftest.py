"""
Shared fixtures for the VRE test suite.
"""

import os

import pytest

from vre.core.backends.sqlite import SQLiteRepository
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


# ---------------------------------------------------------------------------
# Backend-conformance fixtures (#83)
#
# The `repository` fixture is parameterized over every Repository backend so a
# single test asserts parity across them. SQLite runs in-memory; Neo4j connects
# to a live instance via VRE_TEST_NEO4J_URI and is skipped cleanly when absent,
# so CI exercises it where Neo4j exists and skips visibly where it does not.
# ---------------------------------------------------------------------------


def _make_neo4j_repo():
    """Construct a live Neo4jRepository from env, or pytest.skip the test."""
    uri = os.environ.get("VRE_TEST_NEO4J_URI")
    password = os.environ.get("VRE_TEST_NEO4J_PASSWORD")
    if not uri or not password:
        pytest.skip(
            "VRE_TEST_NEO4J_URI / VRE_TEST_NEO4J_PASSWORD not set "
            "— Neo4j conformance skipped"
        )
    try:
        from vre.core.backends.neo4j import Neo4jRepository
    except ImportError:
        pytest.skip("neo4j driver not installed")
    # Username/database default to Neo4j's conventional values — not secrets;
    # the password is required from the env and never defaulted in code.
    user = os.environ.get("VRE_TEST_NEO4J_USER", "neo4j")
    database = os.environ.get("VRE_TEST_NEO4J_DATABASE", "neo4j")
    try:
        return Neo4jRepository(uri, user, password, database=database)
    except Exception as exc:  # noqa: BLE001 — any connect/auth failure = skip
        pytest.skip(f"Neo4j unavailable at {uri}: {exc}")


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param("neo4j", id="neo4j", marks=pytest.mark.requires_neo4j),
    ]
)
def repository(request):
    """A fresh, empty Repository per test, yielded once for each backend."""
    if request.param == "sqlite":
        repo = SQLiteRepository(":memory:")
        try:
            yield repo
        finally:
            repo.close()
    else:
        repo = _make_neo4j_repo()
        repo.clear()
        try:
            yield repo
        finally:
            repo.clear()
            repo.close()


@pytest.fixture
def neo4j_repository():
    """A live Neo4jRepository for backend-specific tests (skips if unavailable)."""
    repo = _make_neo4j_repo()
    repo.clear()
    try:
        yield repo
    finally:
        repo.clear()
        repo.close()
