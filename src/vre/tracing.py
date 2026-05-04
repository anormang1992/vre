# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Trace persistence — serializes grounding results to daily JSONL files.

Enabled by default on every VRE instance. Traces are written to
``~/.vre/traces/YYYY-MM-DD.jsonl``. Disable with ``persist_traces=False``.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from vre.core.grounding.models import GroundingResult
from vre.learning.models import LearningResult


logger = logging.getLogger(__name__)


class TraceEntry(BaseModel):
    """
    A single JSONL line representing one grounding or learning operation.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operation: Literal["check", "learn"]
    concepts: list[str]
    resolved: list[str]
    grounded: bool
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    agent_id: str | None = None
    learning_outcomes: list[dict[str, Any]] | None = None


def build_trace_entry(
    operation: Literal["check", "learn"],
    concepts: list[str],
    result: GroundingResult,
    learning_outcomes: list[LearningResult] | None = None,
) -> TraceEntry:
    """
    Construct a `TraceEntry` from a `GroundingResult` and optional learning outcomes.
    """
    gaps = [gap.model_dump(mode="json") for gap in result.gaps]

    steps = [step.model_dump(mode="json") for step in result.get_pathway_steps()]

    serialized_outcomes: list[dict[str, Any]] | None = None
    if learning_outcomes is not None:
        serialized_outcomes = [lr.model_dump(mode="json") for lr in learning_outcomes]

    return TraceEntry(
        operation=operation,
        concepts=concepts,
        resolved=result.resolved,
        grounded=result.grounded,
        gaps=gaps,
        steps=steps,
        agent_id=str(result.agent_id) if result.agent_id is not None else None,
        learning_outcomes=serialized_outcomes,
    )


DEFAULT_TRACE_DIR = Path.home() / ".vre" / "traces"


class TraceWriter:
    """
    Appends `TraceEntry` objects to daily JSONL files.

    Files are named `YYYY-MM-DD.jsonl` under the trace directory
    (defaults to ``~/.vre/traces/``). Each line is independently valid JSON.
    """

    def __init__(self, trace_dir: Path | None = None) -> None:
        self._trace_dir = trace_dir or DEFAULT_TRACE_DIR

    def _path_for_today(self) -> Path:
        return self._trace_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

    def write(self, entry: TraceEntry) -> None:
        path = self._path_for_today()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = entry.model_dump_json() + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


class TraceManager:
    """
    Internal coordinator owning the TraceWriter and the in-flight suppression
    state used by `learn_all` to skip per-iteration `check()` traces in favor
    of a single summary trace at the end of the loop.

    All write paths are best-effort: persistence failures are logged and never
    raise to the caller.
    """

    def __init__(self, writer: TraceWriter | None) -> None:
        self._writer = writer
        self._suppressed = False

    @contextmanager
    def suppress(self) -> Iterator[None]:
        """
        Suppress writes via `write_check` for the duration of the block.

        Re-entrant: nested suppress() blocks restore the prior state on exit.
        """
        was, self._suppressed = self._suppressed, True
        try:
            yield
        finally:
            self._suppressed = was

    def _safe_write(self, entry: TraceEntry, *, label: str) -> None:
        """
        Persist a trace entry, swallowing and logging any writer failures.
        """
        try:
            self._writer.write(entry)  # type: ignore[union-attr]
        except Exception:
            logger.warning("Failed to persist trace for %s()", label, exc_info=True)

    def write_check(self, concepts: list[str], result: GroundingResult) -> None:
        """
        Persist a 'check' trace entry. No-op when no writer is configured or
        when called inside a `suppress()` block.
        """
        if self._writer is not None and not self._suppressed:
            self._safe_write(build_trace_entry("check", concepts, result), label="check")

    def write_learn(
        self,
        concepts: list[str],
        result: GroundingResult,
        outcomes: list[LearningResult],
    ) -> None:
        """
        Persist a 'learn' trace entry summarizing a learning loop. No-op when
        no writer is configured. Not affected by `suppress()` — learning
        summaries are the reason `learn_all` suppresses its inner check writes.
        """
        if self._writer is not None:
            self._safe_write(
                build_trace_entry("learn", concepts, result, outcomes),
                label="learn_all",
            )
