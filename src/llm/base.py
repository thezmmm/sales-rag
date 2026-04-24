"""
Abstract base classes and shared data structures for LLM providers.

All concrete providers live in sibling modules (ollama.py, openai.py) and
must subclass LLMProvider, implementing:
  - chat()             — single-shot, returns ChatResponse (supports tool calls)
  - stream_chat()      — streaming, yields str chunks     (no tool calls)
  - make_tool_message() — formats tool-result messages
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class ToolCall:
    id: str          # OpenAI requires the ID when posting tool results back
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall]
    # Provider-specific assistant message ready to append to messages list
    raw_message: Any = field(repr=False)


class LLMProvider(ABC):
    """Minimal interface every LLM provider must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """Single-shot completion. Supports tool/function calls."""

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        """Streaming completion — yields text chunks as they arrive.
        Tool calls are not supported in stream mode."""

    @abstractmethod
    def make_tool_message(self, tool_call: ToolCall, content: str) -> dict:
        """Build the tool-result message dict to append after a tool call."""