# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Trace persistence — serializes grounding results to daily JSONL files.

Enabled by default on every VRE instance. Traces are written to
``~/.vre/traces/YYYY-MM-DD.jsonl``. Disable with ``persist_traces=False``.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from vre.core.grounding.models import GroundingResult
from vre.learning.models import LearningResult


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
