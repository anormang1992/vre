"""
Rich REPL with streaming agent output and <think> block rendering.
"""

from __future__ import annotations

import re

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

console = Console()

_THINK_FULL = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_output(raw: str) -> tuple[str, str]:
    """
    Split the streaming buffer into (thinking, answer).

    Handles three states:
    - Completed <think>...</think> blocks → extracted into thinking
    - An open <think> with no closing tag yet → routed to thinking (dim)
    - Everything outside think tags → answer
    """
    completed = "\n".join(_THINK_FULL.findall(raw)).strip()
    remainder = _THINK_FULL.sub("", raw)

    # Unclosed <think> block — model is still generating thinking content
    in_progress = ""
    if "<think>" in remainder:
        pre, _, in_progress = remainder.partition("<think>")
        remainder = pre

    thinking = "\n".join(filter(None, [completed, in_progress])).strip()
    return thinking, remainder.strip()


def _render(thinking: str, answer: str):
    parts = []
    if thinking:
        parts.append(Text(thinking, style="dim"))
    if answer:
        parts.append(Markdown(answer))
    return Group(*parts) if parts else Text("")


def run(agent_executor) -> None:
    console.print("[bold cyan]VRE Demo REPL[/]  (Ctrl+C to exit)\n")
    while True:
        try:
            user_input = console.input("[bold]> [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/]")
            break
        if not user_input:
            continue

        thinking_buf = ""
        answer_buf = ""
        live: Live | None = None

        for chunk in agent_executor.stream({"input": user_input}):
            if "_live_pause" in chunk:
                if live is not None:
                    live.stop()
                    live = None
                continue
            if "_live_resume" in chunk:
                thinking_buf = ""
                answer_buf = ""
                continue
            if "thinking" in chunk:
                thinking_buf += chunk["thinking"]
            if "output" in chunk:
                answer_buf += chunk["output"]
            # Fallback: extract inline <think> tags when reasoning=False
            if not thinking_buf:
                thinking_buf, answer_buf = _split_output(answer_buf)
            if live is None:
                live = Live(console=console, refresh_per_second=15)
                live.start()
            live.update(_render(thinking_buf, answer_buf))

        if live is not None:
            live.stop()
