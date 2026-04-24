"""
Ollama provider — wraps the `ollama` Python SDK for local model inference.

Supported models: llama3.2, mistral, qwen2.5, phi3, …
The Ollama daemon must be running (default: http://localhost:11434).
"""

from __future__ import annotations

from typing import Generator

from llm.base import ChatResponse, LLMProvider, ToolCall


class OllamaProvider(LLMProvider):
    """
    Parameters
    ----------
    model : Ollama model tag, e.g. 'llama3.2:3b', 'mistral'
    host  : Ollama server URL; None uses the SDK default (localhost:11434)
    """

    def __init__(self, model: str = "llama3.2:3b", host: str | None = None):
        import ollama as _ollama  # lazy — not required when using other providers
        self._ollama = _ollama
        self.model = model
        self._client = _ollama.Client(host=host) if host else None

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        resp = (self._client or self._ollama).chat(**kwargs)
        msg = resp.message

        tcs: list[ToolCall] = []
        if msg.tool_calls:
            for i, tc in enumerate(msg.tool_calls):
                tcs.append(ToolCall(
                    id=str(i),  # Ollama doesn't expose call IDs; use index
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        return ChatResponse(content=msg.content, tool_calls=tcs, raw_message=msg)

    def stream_chat(self, messages: list[dict]) -> Generator[str, None, None]:
        resp_iter = (self._client or self._ollama).chat(
            model=self.model, messages=messages, stream=True,
        )
        for chunk in resp_iter:
            text = chunk.message.content
            if text:
                yield text

    def make_tool_message(self, tool_call: ToolCall, content: str) -> dict:
        # Ollama does not require tool_call_id in tool-result messages
        return {"role": "tool", "content": content}