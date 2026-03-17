# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
LearningCallback — the integrator-facing contract for the learning loop.

The callback receives a structured candidate template, the full GroundingResult,
and the specific gap being addressed. It returns the (possibly modified) candidate
and a decision. Provenance is derived from the decision by the engine, not set by
the callback.

Integrators decide:
- Whether to prompt "enter learning mode?" before invoking the callback
- What UI to present for each gap type
- Whether learning is available at all (organizational policy)

Example::

    from vre.learning.callback import LearningCallback
    from vre.learning.models import LearningCandidate, CandidateDecision

    class InteractiveLearner(LearningCallback):
        def __call__(
            self,
            candidate: LearningCandidate,
            grounding: GroundingResult,
            gap: KnowledgeGap,
        ) -> tuple[LearningCandidate | None, CandidateDecision]:
            # Present candidate to user, collect decision
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vre.core.grounding.models import GroundingResult
from vre.core.models import KnowledgeGap
from vre.learning.models import CandidateDecision, LearningCandidate


class LearningCallback(ABC):
    """
    Abstract base class for learning loop callbacks.

    A callback receives a candidate template, the full GroundingResult for context,
    and the specific gap being addressed.

    Lifecycle:
      - `__enter__` is called at the start of a learning session (one
        `learn_all` invocation). Use it to acquire resources or set state.
      - `__exit__` is called when the session ends. Use it to clean up.
      - The callback is used as a context manager by `learn_all`::

            with on_learn:
                # engine invokes on_learn(...) for each gap

      Default implementations are no-ops — override only if your callback
      needs session lifecycle management.

    Returns:
      - (candidate, ACCEPTED) — persist as proposed, provenance: learned
      - (modified_candidate, MODIFIED) — persist modified version, provenance: conversational
      - (None, SKIPPED) — intentionally dismissed (e.g. edge absence is enforcement), loop continues
      - (None, REJECTED) — discard, stops the learning loop entirely
    """

    @abstractmethod
    def __call__(
        self,
        candidate: LearningCandidate,
        grounding: GroundingResult,
        gap: KnowledgeGap,
    ) -> tuple[LearningCandidate | None, CandidateDecision]:
        ...

    def __enter__(self) -> LearningCallback:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
