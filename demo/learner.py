"""
Demo auto-learning module — meta-epistemic dialogue with the agent.

Uses ChatOllama structured output to fill candidate templates, then
presents proposals to the user via Rich for review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from langchain_ollama import ChatOllama
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from demo.repl import console
from vre.learning.callback import LearningCallback
from vre.learning.models import (
    CandidateDecision,
    DepthCandidate,
    ExistenceCandidate,
    LearningCandidate,
    ReachabilityCandidate,
    RelationalCandidate,
)

if TYPE_CHECKING:
    from vre.core.grounding.models import GroundingResult
    from vre.core.models import KnowledgeGap


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_LEARN_SYSTEM = """\
You are a knowledge engineer for the Volute Reasoning Engine (VRE).
VRE is an epistemic safety framework where knowledge is organized as
conceptual primitives with depth levels:

  D0 EXISTENCE  — does this concept exist?
  D1 IDENTITY   — what is it, in principle?
  D2 CAPABILITIES — what can it do / what can happen to it?
  D3 CONSTRAINTS  — under what conditions does that hold?
  D4+ IMPLICATIONS — what follows if it happens?

Each depth carries `properties` — descriptive attributes intrinsic to
the concept at that level. Properties describe what something IS, what
it CAN DO, or what CONDITIONS apply to it. 

You are given an epistemic trace showing the current state of the
knowledge graph and a gap that needs to be filled. Propose knowledge
to fill the gap based on the trace context and your understanding
of the domain.

CRITICAL:
Relationships between concepts (edges) are separate graph structures managed by the engine;
they are not properties.
"""

# ---------------------------------------------------------------------------
# Per-gap-type prompts
# ---------------------------------------------------------------------------

_EXISTENCE_PROMPT = """\
The concept '{name}' does not exist in the knowledge graph.

Propose a D1 (IDENTITY) depth for this concept. D0 (EXISTENCE) will
be created automatically. Fill in the `d1` field with level=1 and
properties that describe what this concept is in principle — its
intrinsic attributes as an entity.

Epistemic trace:
{trace}
"""

_DEPTH_PROMPT = """\
The concept '{name}' exists but is only grounded to D{current} ({current_name}).
It needs to be grounded to at least D{required} ({required_name}).

Propose the missing depth levels. Fill in `new_depths` for each missing
level between D{next_level} and D{required} inclusive. Each entry needs
`level` (int) and `properties` — descriptive attributes intrinsic to the
concept at that depth level.

Existing depths:
{existing}

Epistemic trace:
{trace}
"""

_RELATIONAL_PROMPT = """\
The relationship from '{source}' to '{target}' requires '{target}' to be
grounded to D{required} ({required_name}), but it is only at D{current} ({current_name}).

Propose the missing depth levels for '{target}'. Fill in `new_depths` for
each missing level. Each entry needs `level` (int) and `properties` —
descriptive attributes intrinsic to '{target}' at that depth level.

Existing depths on '{target}':
{existing}

Epistemic trace:
{trace}
"""

_REACHABILITY_PROMPT = """\
The concept '{source}' is not connected to other concepts in the subgraph.

Available targets:
{targets}

Existing depths on '{source}':
{source_depths}

Propose an edge connecting '{source}' to one of the available targets.
You MUST fill in ALL of the following fields:

1. target_name — name of the target concept
2. relation_type — one of: APPLIES_TO, REQUIRES, CONSTRAINED_BY, DEPENDS_ON, INCLUDES
3. source_depth_level — int, which depth on '{source}' to place the edge at
4. target_depth_level — int, the minimum depth required on the target for the edge to resolve

Do NOT propose new depths. Focus only on edge placement. Any missing depths
will be handled separately after the edge is placed.

Epistemic trace:
{trace}
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_existing_depths(primitive) -> str:
    lines = []
    for d in sorted(primitive.depths, key=lambda d: d.level):
        props = ", ".join(f"{k}={v}" for k, v in d.properties.items()) if d.properties else "none"
        lines.append(f"  D{d.level.value} {d.level.name}: {props}")
    return "\n".join(lines) or "  (none)"


def _fmt_available_targets(grounding: GroundingResult, source_id) -> str:
    lines = []
    if grounding.trace:
        for p in grounding.trace.result.primitives:
            if p.id == source_id:
                continue
            lines.append(f"  {p.name}:")
            if p.depths:
                for d in sorted(p.depths, key=lambda d: d.level):
                    props = ", ".join(f"{k}={v}" for k, v in d.properties.items()) if d.properties else "none"
                    lines.append(f"    D{d.level.value} {d.level.name}: {props}")
            else:
                lines.append("    (no depths)")
    return "\n".join(lines) or "  (none)"


def build_prompt(candidate: LearningCandidate, grounding: GroundingResult, gap) -> str:
    """
    Build the LLM prompt for a given gap type and candidate template.
    """
    trace = str(grounding)
    if isinstance(candidate, ExistenceCandidate):
        return _EXISTENCE_PROMPT.format(name=candidate.name, trace=trace)
    if isinstance(candidate, DepthCandidate):
        current = gap.current_depth.value if gap.current_depth is not None else -1
        current_name = gap.current_depth.name if gap.current_depth is not None else "none"
        next_level = current + 1
        return _DEPTH_PROMPT.format(
            name=gap.primitive.name,
            current=current,
            current_name=current_name,
            required=gap.required_depth.value,
            required_name=gap.required_depth.name,
            next_level=next_level,
            existing=_fmt_existing_depths(gap.primitive),
            trace=trace,
        )
    if isinstance(candidate, RelationalCandidate):
        current = gap.current_depth.value if gap.current_depth is not None else -1
        current_name = gap.current_depth.name if gap.current_depth is not None else "none"
        return _RELATIONAL_PROMPT.format(
            source=gap.source.name,
            target=gap.target.name,
            required=gap.required_depth.value,
            required_name=gap.required_depth.name,
            current=current,
            current_name=current_name,
            existing=_fmt_existing_depths(gap.target),
            trace=trace,
        )
    if isinstance(candidate, ReachabilityCandidate):
        return _REACHABILITY_PROMPT.format(
            source=gap.primitive.name,
            targets=_fmt_available_targets(grounding, gap.primitive.id),
            source_depths=_fmt_existing_depths(gap.primitive),
            trace=trace,
        )
    return ""


# ---------------------------------------------------------------------------
# Candidate rendering
# ---------------------------------------------------------------------------

def render_candidate(candidate: LearningCandidate, gap) -> None:
    """
    Render a filled candidate to the console via Rich.
    """
    if isinstance(candidate, ExistenceCandidate):
        console.print(Panel(
            f"[bold]Concept:[/] {candidate.name}\n"
            f"[bold]D1 Identity:[/] {candidate.d1.properties if candidate.d1 else '(empty)'}",
            title="[bold yellow]Existence Proposal[/]",
            border_style="yellow",
        ))
    elif isinstance(candidate, DepthCandidate):
        depths_str = "\n".join(
            f"  D{d.level.value} {d.level.name}: {d.properties}" for d in candidate.new_depths
        ) or "  (none)"
        console.print(Panel(
            f"[bold]Concept:[/] {gap.primitive.name}\n"
            f"[bold]New depths:[/]\n{depths_str}",
            title="[bold yellow]Depth Proposal[/]",
            border_style="yellow",
        ))
    elif isinstance(candidate, RelationalCandidate):
        depths_str = "\n".join(
            f"  D{d.level.value} {d.level.name}: {d.properties}" for d in candidate.new_depths
        ) or "  (none)"
        console.print(Panel(
            f"[bold]Source:[/] {gap.source.name} → [bold]Target:[/] {gap.target.name}\n"
            f"[bold]New depths on target:[/]\n{depths_str}",
            title="[bold yellow]Relational Proposal[/]",
            border_style="yellow",
        ))
    elif isinstance(candidate, ReachabilityCandidate):
        console.print(Panel(
            f"[bold]Source:[/] {gap.primitive.name}\n"
            f"[bold]Target:[/] {candidate.target_name or '(none)'}\n"
            f"[bold]Relation:[/] {candidate.relation_type.value if candidate.relation_type else '(none)'}\n"
            f"[bold]Source depth:[/] D{candidate.source_depth_level.value if candidate.source_depth_level else '?'}\n"
            f"[bold]Target depth:[/] D{candidate.target_depth_level.value if candidate.target_depth_level else '?'}\n\n"
            f"[dim italic]Edge placement is a safety-critical decision. The depth at which an edge\n"
            f"is placed determines when an agent can reason about this relationship.\n"
            f"If this concept should NOT be connected — if the absence of this edge is\n"
            f"itself an enforcement mechanism — choose Skip.[/]",
            title="[bold yellow]Reachability Proposal[/]",
            border_style="yellow",
        ))


# ---------------------------------------------------------------------------
# DemoLearner — the callback implementation
# ---------------------------------------------------------------------------

class DemoLearner(LearningCallback):
    """
    Learning callback that uses ChatOllama to fill candidate templates
    and presents them to the user for review via Rich.

    Flow: agent proposes → user reviews → accept/modify/skip/reject.
    If the user requests modifications, the agent re-proposes with feedback
    until the user is satisfied.
    """

    def __init__(self, model: str = "qwen3:8b") -> None:
        self._model = model
        self._active = False

    def __enter__(self) -> DemoLearner:
        self._active = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._active = False

    def _invoke_llm(
        self,
        candidate: LearningCandidate,
        messages: list[dict[str, str]],
    ) -> LearningCandidate:
        llm = ChatOllama(model=self._model).with_structured_output(type(candidate))
        return llm.invoke(messages)

    def __call__(
        self,
        candidate: LearningCandidate,
        grounding: GroundingResult,
        gap: KnowledgeGap,
    ) -> tuple[LearningCandidate | None, CandidateDecision]:

        if not self._active:
            if not Confirm.ask("\n[bold cyan]Knowledge gap detected.[/] Enter learning mode?"):
                return None, CandidateDecision.REJECTED
            self._active = True

        messages = [
            {"role": "system", "content": _LEARN_SYSTEM},
            {"role": "user", "content": build_prompt(candidate, grounding, gap)},
        ]

        console.print("\n[dim]Agent is proposing knowledge...[/]")
        filled = self._invoke_llm(candidate, messages)
        render_candidate(filled, gap)

        was_modified = False
        while True:
            choice = questionary.select(
                "Decision:",
                choices=["accept", "modify", "skip", "reject"],
                default="accept",
            ).ask()

            if choice is None:  # Ctrl-C
                return None, CandidateDecision.REJECTED
            if choice == "accept":
                decision = CandidateDecision.MODIFIED if was_modified else CandidateDecision.ACCEPTED
                return filled, decision
            if choice == "skip":
                return None, CandidateDecision.SKIPPED
            if choice == "reject":
                return None, CandidateDecision.REJECTED
            if choice == "modify":
                was_modified = True
                messages.append({"role": "assistant", "content": filled.model_dump_json()})
                feedback = Prompt.ask("[bold]What should be changed?[/]")
                messages.append({"role": "user", "content": feedback})
                console.print("\n[dim]Agent is revising...[/]")
                filled = self._invoke_llm(candidate, messages)
                render_candidate(filled, gap)
