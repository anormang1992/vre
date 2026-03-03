"""
Minimal tool-calling agent loop using ChatOllama and langchain-core only.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

SYSTEM = "You are a helpful assistant with access to tools for filesystem management."


class ToolAgent:
    """
    Agent loop: stream the model response, execute any tool calls, repeat
    until the model produces a final answer with no tool calls.

    Text chunks are yielded as they arrive so the REPL can display them
    incrementally. Tool-call rounds are also streamed — the model's thinking
    content (including <think> blocks) appears before the tool fires.
    """

    def __init__(self, tools: list, model: str = "qwen3:8b") -> None:
        self._llm = ChatOllama(model=model, reasoning=True).bind_tools(tools)
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

            messages.append(full)

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
