"""
Unit tests for vre.guard — vre_guard decorator.
"""

from unittest.mock import MagicMock

from vre.core.grounding import GroundingResult
from vre.core.policy import PolicyResult
from vre.learning.callback import LearningCallback
from vre.learning.models import CandidateDecision


# ── helpers ──────────────────────────────────────────────────────────────────

def _grounding(grounded=True, resolved=None, gaps=None):
    return GroundingResult(
        grounded=grounded,
        resolved=resolved or ["file"],
        gaps=gaps or [],
    )


def _mock_vre(grounding: GroundingResult, policy: PolicyResult | None = None):
    """Return a MagicMock VRE wired with the given grounding and policy."""
    mock = MagicMock()
    mock.check.return_value = grounding
    mock.check_policy.return_value = policy or PolicyResult(action="PASS")
    return mock


# ── ungrounded path ───────────────────────────────────────────────────────────

def test_vre_guard_returns_grounding_result_when_not_grounded():
    """When grounding fails, vre_guard returns GroundingResult without calling fn."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(_grounding(grounded=False, gaps=[MagicMock()]))

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

def test_vre_guard_blocks_when_pending_and_no_handler():
    """PENDING policy with no on_policy handler → returns PolicyResult(BLOCK)."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(action="PENDING", confirmation_message="Confirm this action?"),
    )

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, PolicyResult)
    assert result.action == "BLOCK"


def test_vre_guard_calls_fn_when_policy_confirmed():
    """PENDING policy with on_policy=True handler → function executes."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(action="PENDING", confirmation_message="Confirm?"),
    )

    @vre_guard(mock_vre, concepts=["file"], on_policy=lambda msg: True)
    def my_fn():
        return "executed"

    result = my_fn()
    assert result == "executed"


def test_vre_guard_blocks_on_block_policy():
    """BLOCK policy → returns the PolicyResult."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(action="BLOCK", reason="Forbidden"),
    )

    @vre_guard(mock_vre, concepts=["file"])
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, PolicyResult)
    assert result.action == "BLOCK"
    assert result.reason == "Forbidden"


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
    _, call_args, _ = mock_vre.check_policy.mock_calls[0]
    assert call_args[1] == "multiple"  # second positional arg is cardinality


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


# ── on_learn path ────────────────────────────────────────────────────────────

class _StubLearner(LearningCallback):
    """Minimal LearningCallback for guard tests."""
    def __call__(self, candidate, grounding, gap):
        return None, CandidateDecision.REJECTED


def test_vre_guard_on_learn_invokes_learn_all_when_not_grounded():
    """When grounding fails and on_learn is provided, learn_all is called."""
    from vre.guard import vre_guard

    grounded_result = _grounding(grounded=True)
    mock_vre = _mock_vre(_grounding(grounded=False, gaps=[MagicMock()]))
    mock_vre.learn_all.return_value = grounded_result
    mock_vre.check_policy.return_value = PolicyResult(action="PASS")

    learner = _StubLearner()

    @vre_guard(mock_vre, concepts=["file"], on_learn=learner)
    def my_fn():
        return "executed"

    result = my_fn()
    assert result == "executed"
    mock_vre.learn_all.assert_called_once()


def test_vre_guard_on_learn_fires_on_trace_after_learning():
    """on_trace fires twice: once after initial grounding, once after learning."""
    from vre.guard import vre_guard

    traces = []
    grounded_result = _grounding(grounded=True)
    mock_vre = _mock_vre(_grounding(grounded=False, gaps=[MagicMock()]))
    mock_vre.learn_all.return_value = grounded_result
    mock_vre.check_policy.return_value = PolicyResult(action="PASS")

    learner = _StubLearner()

    @vre_guard(mock_vre, concepts=["file"], on_trace=traces.append, on_learn=learner)
    def my_fn():
        return "executed"

    my_fn()
    assert len(traces) == 2
    assert traces[0].grounded is False
    assert traces[1].grounded is True


def test_vre_guard_on_learn_returns_grounding_when_still_not_grounded():
    """When learn_all doesn't fully resolve gaps, returns the GroundingResult."""
    from vre.guard import vre_guard

    still_ungrounded = _grounding(grounded=False, gaps=[MagicMock()])
    mock_vre = _mock_vre(_grounding(grounded=False, gaps=[MagicMock()]))
    mock_vre.learn_all.return_value = still_ungrounded

    learner = _StubLearner()

    @vre_guard(mock_vre, concepts=["file"], on_learn=learner)
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, GroundingResult)
    assert result.grounded is False


def test_vre_guard_on_policy_decline_returns_block():
    """When on_policy returns False, returns PolicyResult(BLOCK, 'User declined')."""
    from vre.guard import vre_guard

    mock_vre = _mock_vre(
        _grounding(),
        PolicyResult(action="PENDING", confirmation_message="Are you sure?"),
    )

    @vre_guard(mock_vre, concepts=["file"], on_policy=lambda msg: False)
    def my_fn():
        return "executed"

    result = my_fn()
    assert isinstance(result, PolicyResult)
    assert result.action == "BLOCK"
    assert result.reason == "User declined"
