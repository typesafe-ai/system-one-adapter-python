"""Native Anthropic Messages API providers.

Native mode uses ``output_config.format``, Claude's schema-constrained output, which is
only available on this native API and not through an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from typing import Any

import anthropic
from typesafe_sdk import TypeSafeError

from system_one_adapter._utils.error_handling import map_provider_error
from system_one_adapter.providers.base import (
    Message,
    ProviderResult,
    record_request,
    record_response,
    translating,
)

_DEFAULT_MAX_TOKENS = 4096


class _AnthropicErrors:
    """Shared Anthropic-SDK error translation for the sync and async providers."""

    @staticmethod
    def translate_error(error: Exception) -> TypeSafeError:
        """Map an Anthropic SDK exception to an SDK error."""
        return map_provider_error(
            error,
            status_errors=(anthropic.APIStatusError,),
            timeout_errors=(anthropic.APITimeoutError,),
            connection_errors=(anthropic.APIConnectionError,),
        )


def _request_kwargs(
    model_name: str,
    messages: list[Message],
    schema: dict[str, Any],
    *,
    structured: bool,
    max_tokens: int,
) -> dict[str, Any]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    conversation = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    kwargs: dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system,
        "messages": conversation,
    }
    if structured:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    return kwargs


def _result(response: Any) -> ProviderResult:
    record_response(response, finish_reason=response.stop_reason)
    if response.stop_reason == "max_tokens":
        raise TypeSafeError(
            "Anthropic response was truncated at the output token limit. "
            "Increase max_tokens on AnthropicProvider or AsyncAnthropicProvider, or request fewer questions."
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    return ProviderResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


class AnthropicProvider(_AnthropicErrors):
    """Synchronously call the Anthropic Messages API."""

    def __init__(self, model_name: str, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        """Build the provider with a configurable output token limit."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(max_retries=0)

    def close(self) -> None:
        """Release the SDK client's connection pool."""
        self._client.close()

    def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one Messages request and return its raw payload and usage."""
        with translating(self.translate_error):
            kwargs = _request_kwargs(self.model_name, messages, schema, structured=structured, max_tokens=self.max_tokens)
            record_request(kwargs, api="messages")
            response = self._client.messages.create(**kwargs)
        return _result(response)


class AsyncAnthropicProvider(_AnthropicErrors):
    """Asynchronously call the Anthropic Messages API."""

    def __init__(self, model_name: str, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        """Build the provider with a configurable output token limit."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(max_retries=0)

    async def aclose(self) -> None:
        """Release the SDK client's connection pool."""
        await self._client.close()

    async def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one Messages request and return its raw payload and usage."""
        with translating(self.translate_error):
            kwargs = _request_kwargs(self.model_name, messages, schema, structured=structured, max_tokens=self.max_tokens)
            record_request(kwargs, api="messages")
            response = await self._client.messages.create(**kwargs)
        return _result(response)
