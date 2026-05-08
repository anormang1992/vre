"""
VRE-guarded shell tool and learn_gaps tool for the demo agent.
"""

from __future__ import annotations

import subprocess
from typing import Callable

from vre.core.models import ReachabilityGap
from vre.guard import vre_guard
from vre.learning.models import ReachabilityCandidate
from vre.learning.templates import template_for_gap


def init_tools(
    vre,
    sandbox: str,
    concepts: Callable,
    cardinality: Callable,
    on_trace: Callable,
    on_policy: Callable,
):
    @vre_guard(
        vre,
        concepts=concepts,
        cardinality=cardinality,
        on_trace=on_trace,
        on_policy=on_policy,
    )
    def shell_tool(command: str, cwd: str = sandbox) -> str:
        """
        Execute a shell command inside the workspace directory. The
        workspace is fully writable — files can be created, modified,
        deleted, and executed here. Use relative paths to stay within
        the workspace.
        """
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.stdout + result.stderr

    return shell_tool


def init_learn_tool(vre, learner):
    """
    Create a learn_gaps tool that the agent can invoke to fill knowledge gaps.

    The loop lives here — not in VRE core. VRE provides check(),
    template_for_gap(), and learn_gap(). The integrator (this tool)
    decides how to orchestrate them.
    """
    def learn_gaps(concepts: str) -> str:
        """
        Identify and resolve knowledge gaps for the given concepts.
        Pass a comma-separated list of concept names.
        """
        concept_list = [c.strip() for c in concepts.split(",")]
        grounding = vre.check(concept_list)
        if grounding.grounded:
            return str(grounding)

        skipped: set[int] = set()
        with learner:
            while not grounding.grounded and grounding.gaps:
                gap_index = next(
                    (i for i, _ in enumerate(grounding.gaps) if i not in skipped),
                    None,
                )
                if gap_index is None:
                    break

                gap = grounding.gaps[gap_index]
                template = template_for_gap(gap)
                filled = learner(template, grounding, gap)

                if filled is None:
                    skipped.add(gap_index)
                    continue

                if isinstance(gap, ReachabilityGap) and isinstance(filled, ReachabilityCandidate):
                    prereqs = vre.learning_engine.reachability_prerequisites(gap, filled)
                    for depth_gap in prereqs:
                        depth_template = template_for_gap(depth_gap)
                        depth_filled = learner(depth_template, grounding, depth_gap)
                        if depth_filled is None:
                            break
                        vre.learning_engine.learn_gap(depth_gap, depth_filled)
                    else:
                        vre.learning_engine.learn_gap(gap, filled)
                        vre.resolver.invalidate()
                        grounding = vre.check(concept_list)
                        skipped.clear()
                        continue
                    skipped.add(gap_index)
                    continue

                vre.learning_engine.learn_gap(gap, filled)
                vre.resolver.invalidate()
                grounding = vre.check(concept_list)
                skipped.clear()

        return str(grounding)

    return learn_gaps
