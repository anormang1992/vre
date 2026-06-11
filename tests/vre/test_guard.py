"""
Unit tests for vre.guard — vre_guard decorator.
"""

from unittest.mock import MagicMock
from uuid import uuid4

from vre.core.grounding import GroundingResult
from vre.core.models import ExistenceGap, Primitive
from vre.core.policy import PolicyAction, PolicyResult
from vre.core.policy.models import PolicyViolation, Policy
from vre.guard import vre_guard


# ── helpers ──────────────────────────────────────────────────────────────────

def _grounding(grounded=True, resolved=None, gaps=None, agent_id=None):
    return GroundingResult(
        grounded=grounded,
        resolved=resolved or ["file"],
        gaps=gaps or [],
        agent_id=agent_id,
    )


def _mock_vre(grounding: GroundingResult, policy: PolicyResult | None = None):
    """Return a MagicMock VRE wired with the given grounding and policy."""
    mock = MagicMock()
    mock.check.return_value = grounding
    mock.check_policy.return_value = policy or PolicyResult(action=PolicyAction.PASS)
    return mock


def _violation(message="Confirm?", requires_confirmation=True) -> PolicyViolation:
    return PolicyViolation(
        policy=Policy(name="Test", requires_confirmation=requires_confirmation,
                      confirmation_message=message),
    )


# ── ungrounded path ───────────────────────────────────────────────────────────

def test_vre_guard_returns_grounding_result_when_not_grounded():
    """When grounding fails, vre_guard returns GroundingResult without calling fn."""
    from vre.guard import vre_guard

    gap = ExistenceGap(primitive=Primitive(name="unknown", depths=[]))
    mock_vre = _mock_vre(_grounding(grounded=False, gaps=[gap]))

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        """Write a file."""
        return "executed"

    result = my_fn()
    assert isinstance(result, GroundingResult)
    assert result.grounded is False
    assert "[VRE] Not grounded" in str(result)


def test_vre_guard_blocks_on_existence_gap():
    """Existence gap → grounded=False → returns GroundingResult without calling fn."""
    from vre.guard import vre_guard
    from vre.core.models import ExistenceGap, Primitive

    gap = ExistenceGap(primitive=Primitive(name="api", depths=[]))
    mock_vre = _mock_vre(_grounding(grounded=False, gaps=[gap]))

    @vre_guard(mock_vre, concepts=["file", "api"])
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, GroundingResult)
    assert result.grounded is False
    assert len(result.gaps) == 1
    assert "[VRE] Not grounded" in str(result)


def test_vre_guard_exposes_concepts():
    """Decorated function has _vre_concepts set to the declared concepts."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["write", "file"])
    def my_fn():
        """Write content to a file."""
        pass

    assert hasattr(my_fn, "_vre_concepts")
    assert my_fn._vre_concepts == ["write", "file"]


# ── single-phase: default execution ──────────────────────────────────────────

def test_vre_guard_executes_fn_on_first_call():
    """When grounded, the function is called and its result returned immediately."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    result = my_fn()
    assert result == "executed"


def test_vre_guard_fires_on_trace_when_grounded():
    """on_trace is called once on a single-phase call."""
    from vre.guard import vre_guard

    traces = []
    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"], on_trace=traces.append)
    def my_fn():
        return "executed"

    my_fn()
    assert len(traces) == 1
    assert isinstance(traces[0], GroundingResult)


def test_vre_guard_on_trace_raising_is_swallowed_and_execution_continues():
    """A raising on_trace is logged, not propagated — observability must not break enforcement (#97)."""
    mock_vre = _mock_vre(_grounding())

    def boom(result):
        raise RuntimeError("trace boom")

    @vre_guard(mock_vre, concepts=["file"], on_trace=boom)
    def my_fn():
        return "executed"

    assert my_fn() == "executed"


def test_vre_guard_on_trace_receives_agent_id():
    """on_trace receives GroundingResult with agent_id when VRE stamps it."""
    agent_id = uuid4()
    traces = []
    mock_vre = _mock_vre(_grounding(agent_id=agent_id))

    @vre_guard(mock_vre, concepts=["file"], on_trace=traces.append)
    def my_fn():
        return "executed"

    my_fn()
    assert traces[0].agent_id == agent_id


def test_vre_guard_grounding_called_once():
    """VRE grounding is called exactly once per single-phase call."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    my_fn()
    assert mock_vre.check.call_count == 1


# ── single-phase: policy gates ────────────────────────────────────────────────

def test_vre_guard_blocks_when_no_handler():
    """BLOCK policy with no on_policy handler → returns PolicyResult(BLOCK)."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(
            action=PolicyAction.BLOCK,
            reason="Confirmation required, no handler",
            violations=[_violation()],
        ),
    )

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, PolicyResult)
    assert result.action == PolicyAction.BLOCK


def test_vre_guard_calls_fn_when_policy_passes():
    """PASS policy (on_policy handled inside check_policy) → function executes."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(action=PolicyAction.PASS, violations=[_violation()]),
    )

    @vre_guard(mock_vre, concepts=["file"], on_policy=lambda violations: True)
    def my_fn():
        return "executed"

    result = my_fn()
    assert result == "executed"


def test_vre_guard_blocks_on_block_policy():
    """BLOCK policy → returns the PolicyResult."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(action=PolicyAction.BLOCK, reason="Forbidden"),
    )

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, PolicyResult)
    assert result.action == PolicyAction.BLOCK
    assert result.reason == "Forbidden"


def test_vre_guard_passes_on_policy_through_to_check_policy():
    """on_policy is forwarded to vre.check_policy()."""
    from vre.guard import vre_guard

    handler = lambda violations: True  # noqa: E731
    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"], on_policy=handler)
    def my_fn():
        return "executed"

    my_fn()
    mock_vre.check_policy.assert_called_once()
    call_kwargs = mock_vre.check_policy.call_args
    assert call_kwargs.kwargs.get("on_policy") is handler


# ── single-phase: different fns are independent ───────────────────────────────

def test_vre_guard_same_args_different_fns_are_independent():
    """Two decorated functions with same args execute independently."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"])
    def fn_a():
        return "a"

    @vre_guard(mock_vre, concepts=["file"])
    def fn_b():
        return "b"

    assert fn_a() == "a"
    assert fn_b() == "b"


# ── callable concepts ─────────────────────────────────────────────────────────

def test_vre_guard_callable_concepts_called_with_fn_args():
    """When concepts is callable, it receives (*args, **kwargs) at call time."""
    from vre.guard import vre_guard

    received = []

    def concept_fn(*args, **kwargs):
        received.append((args, kwargs))
        return ["file"]

    mock_vre = _mock_vre(_grounding(resolved=["file"]))

    @vre_guard(mock_vre, concepts=concept_fn)
    def my_fn(path, mode="r"):
        return "executed"

    my_fn("a.txt", mode="w")
    assert received == [(("a.txt",), {"mode": "w"})]


def test_vre_guard_callable_concepts_result_is_grounded():
    """Concepts returned by the callable are passed to vre.check()."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding(resolved=["directory"]))

    @vre_guard(mock_vre, concepts=lambda path: ["directory"] if path.endswith("/") else ["file"])
    def my_fn(path):
        return "executed"

    my_fn("logs/")
    mock_vre.check.assert_called_once_with(["directory"], min_depth=None)


def test_vre_guard_callable_concepts_stored_on_attribute():
    """
    _vre_concepts stores the callable itself for introspection.
    """
    from vre.guard import vre_guard

    def fn():
        return ["file"]
    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=fn)
    def my_fn():
        return "executed"

    assert my_fn._vre_concepts is fn


# ── callable cardinality ──────────────────────────────────────────────────────

def test_vre_guard_callable_cardinality_evaluated_on_call():
    """
    When cardinality is callable, it is evaluated on the single-phase call.
    """
    from vre.guard import vre_guard

    received = []

    def card_fn(*args, **kwargs):
        received.append((args, kwargs))
        return "multiple"

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"], cardinality=card_fn)
    def my_fn(x):
        return "executed"

    my_fn(42)
    assert received == [((42,), {})]


def test_vre_guard_callable_cardinality_receives_fn_args():
    """Cardinality callable receives the same (*args, **kwargs) as the decorated fn."""
    from vre.guard import vre_guard

    received = []

    def card_fn(*args, **kwargs):
        received.append((args, kwargs))
        return "single" if len(args) == 1 else "multiple"

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"], cardinality=card_fn)
    def my_fn(*paths):
        return "executed"

    my_fn("a.txt", "b.txt")
    assert received == [(("a.txt", "b.txt"), {})]


def test_vre_guard_callable_cardinality_passed_to_check_policy():
    """Resolved cardinality from callable is forwarded to vre.check_policy()."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"], cardinality=lambda *a, **kw: "multiple")
    def my_fn():
        return "executed"

    my_fn()
    mock_vre.check_policy.assert_called_once()
    call_args = mock_vre.check_policy.call_args
    assert call_args[0][1] == "multiple"  # second positional arg is cardinality


# ── min_depth passthrough ────────────────────────────────────────────────────

def test_vre_guard_min_depth_passed_to_check():
    """
    min_depth parameter is forwarded to vre.check().
    """
    from vre.guard import vre_guard
    from vre.core.models import DepthLevel

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"], min_depth=DepthLevel.CONSTRAINTS)
    def my_fn():
        return "executed"

    my_fn()
    mock_vre.check.assert_called_once_with(["file"], min_depth=DepthLevel.CONSTRAINTS)


def test_vre_guard_no_min_depth_passes_none():
    """
    Without min_depth, vre.check() is called with min_depth=None.
    """
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding())

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    my_fn()
    mock_vre.check.assert_called_once_with(["file"], min_depth=None)


def test_vre_guard_on_policy_decline_returns_block():
    """When on_policy returns False (inside check_policy), returns PolicyResult(BLOCK)."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(
            action=PolicyAction.BLOCK,
            reason="User declined",
            violations=[_violation()],
        ),
    )

    @vre_guard(mock_vre, concepts=["file"], on_policy=lambda violations: False)
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, PolicyResult)
    assert result.action == PolicyAction.BLOCK
    assert result.reason == "User declined"
