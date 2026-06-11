from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .types import ModelResponse


class ModelAdapter(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        raise NotImplementedError


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        temperature: float = 0,
        max_tokens: int = 4096,
        retries: int = 2,
        extra_headers: dict[str, str] | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.extra_headers = extra_headers or {}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in tools],
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = await asyncio.to_thread(self._post, payload)
        choice = data["choices"][0]["message"]
        calls = choice.get("tool_calls") or []
        tool_call = None
        if calls:
            function = calls[0]["function"]
            tool_call = {
                "id": calls[0].get("id"),
                "name": function["name"],
                "arguments": json.loads(function.get("arguments") or "{}"),
            }
        content = choice.get("content") or ""
        return ModelResponse(
            content=content,
            tool_call=tool_call,
            final_answer=None if tool_call else content,
            usage=data.get("usage", {}),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                **self.extra_headers,
            },
        )
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
        raise RuntimeError(f"model request failed: {last_error}")


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter chat-completions adapter with optional app attribution."""

    DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0,
        max_tokens: int = 4096,
        retries: int = 2,
        referer: str = "",
        app_title: str = "ToolMem Bench",
        endpoint: str = DEFAULT_ENDPOINT,
    ):
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if app_title:
            headers["X-OpenRouter-Title"] = app_title
        super().__init__(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            extra_headers=headers,
        )


class DeterministicFakeModel(ModelAdapter):
    """Scripted adapter used for deterministic end-to-end tests."""

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        policy: Callable[[list[dict[str, Any]]], ModelResponse] | None = None,
    ):
        self.responses = list(responses or [])
        self.policy = policy

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        assert [tool["name"] for tool in tools] == ["create_tool", "find_tool", "update_tool"]
        if self.policy:
            return self.policy(messages)
        if not self.responses:
            return ModelResponse(final_answer="", content="")
        return self.responses.pop(0)
