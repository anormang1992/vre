"""
Minimal tool-calling agent loop using ChatOllama and langchain-core only.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

SYSTEM = """\
You are a filesystem assistant. You have two tools:

1. shell_tool — runs a shell command in the workspace directory. Use relative paths.
2. learn_gaps — resolves knowledge gaps so blocked commands can proceed.

Every shell command is checked by VRE (Volute Reasoning Engine) before execution.
If VRE does not have enough knowledge about the concepts involved, the command is
blocked and the tool returns a grounding result listing the gaps.

When a shell command is blocked:
- Read the gaps in the response (DEPTH, REACHABILITY, RELATIONAL, EXISTENCE).
- Call learn_gaps with ALL the concept names from the blocked command (comma-separated).
- After learn_gaps resolves the gaps, retry the original shell command.

Do not assume a tool call succeeded unless you see its result. Do not narrate actions
you have not taken. Call tools, read results, then respond.
"""


class ToolAgent:
    """
    Agent loop: stream the model response, execute any tool calls, repeat
    until the model produces a final answer with no tool calls.

    Text chunks are yielded as they arrive so the REPL can display them
    incrementally. Tool-call rounds are also streamed — the model's thinking
    content (including <think> blocks) appears before the tool fires.
    """

    def __init__(self, tools: list, model: str = "gemma4:26b") -> None:
        self._llm = ChatOllama(
            model=model,
            reasoning=True,
            top_p=0.95,
            top_k=64,
            temperature=1.0
        ).bind_tools(tools)
        self._tools = {t.name: t for t in tools}

    def stream(self, inputs: dict):
        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=inputs["input"]),
        ]

        while True:
            # Stream chunks so the REPL can display them as they arrive.
            # AIMessageChunk supports + to accumulate tool_call_chunks into
            # complete tool_calls on the final aggregated message.
            chunks = []
            for chunk in self._llm.stream(messages):
                chunks.append(chunk)
                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content", "")
                if reasoning:
                    yield {"thinking": reasoning}
                if chunk.content:
                    yield {"output": chunk.content}

            # Reconstruct a full AIMessage from the streamed chunks.
            full = chunks[0]
            for c in chunks[1:]:
                full = full + c

            history_msg = AIMessage(
                content=full.content,
                tool_calls=full.tool_calls,
            )
            messages.append(history_msg)

            if not full.tool_calls:
                break

            yield {"_live_pause": True}
            for tc in full.tool_calls:
                tool = self._tools.get(tc["name"])
                result = (
                    tool.invoke(tc["args"])
                    if tool is not None
                    else f"Unknown tool: {tc['name']}"
                )
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
            yield {"_live_resume": True}


def make_agent(tools: list, model: str = "qwen3:8b") -> ToolAgent:
    return ToolAgent(tools, model)
