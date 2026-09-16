"""OpenAI Responses and OpenAI-compatible Chat Completions providers.

``base_url`` and ``api_key`` target any OpenAI-compatible vendor (xAI, Groq, Together,
DeepSeek, Mistral, a local server, ...); both default to the OpenAI SDK's own resolution
when omitted. OpenAI's endpoint defaults to Responses; other endpoints default to
Chat Completions. ``api`` explicitly selects either transport, including for proxies.
"""

from __future__ import annotations

from typing import Any, Literal

import openai
from typesafe_sdk import TypeSafeError

from system_one_adapter._utils.error_handling import map_provider_error
from system_one_adapter.providers.base import (
    Message,
    ProviderResult,
    record_request,
    record_response,
    render_messages,
    translating,
)


class _OpenAIErrors:
    """Shared OpenAI-SDK error translation for the sync and async providers."""

    @staticmethod
    def translate_error(error: Exception) -> TypeSafeError:
        """Map an OpenAI SDK exception to an SDK error."""
        return map_provider_error(
            error,
            status_errors=(openai.APIStatusError,),
            timeout_errors=(openai.APITimeoutError,),
            connection_errors=(openai.APIConnectionError,),
        )


def _response_format(schema: dict[str, Any], *, structured: bool) -> Any:
    if not structured:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": "evaluation", "schema": schema, "strict": True},
    }


def _result(response: Any) -> ProviderResult:
    record_response(response, finish_reason=response.choices[0].finish_reason)
    return ProviderResult(
        text=response.choices[0].message.content or "",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )


def _responses_request_kwargs(model_name: str, messages: list[Message], schema: dict[str, Any], *, structured: bool) -> dict[str, Any]:
    output_format = {"type": "json_schema", "name": "evaluation", "schema": schema, "strict": True} if structured else {"type": "json_object"}
    kwargs: dict[str, Any] = {
        "model": model_name,
        "input": render_messages(messages),
        "text": {"format": output_format},
        "store": False,
    }
    if structured:
        kwargs["instructions"] = "\n\n".join(message.content for message in messages if message.role == "system")
        kwargs["input"] = render_messages([message for message in messages if message.role != "system"])
    # JSON mode requires a JSON instruction in `input`; the separate `instructions`
    # field does not satisfy the API's check, so prompted mode keeps system messages.
    return kwargs


def _responses_result(response: Any) -> ProviderResult:
    record_response(response, finish_reason=response.status)
    if response.status != "completed":
        reason = response.status
        if response.error is not None:
            reason = response.error.message
        elif response.incomplete_details is not None:
            reason = response.incomplete_details.reason
        raise TypeSafeError(f"OpenAI response did not complete: {reason}.")
    return ProviderResult(
        text=response.output_text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


class OpenAIProvider(_OpenAIErrors):
    """Synchronously call Responses or an OpenAI-compatible chat API."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api: Literal["responses", "chat_completions"] | None = None,
    ) -> None:
        """Build the provider for ``model_name`` on the given endpoint."""
        if api not in (None, "responses", "chat_completions"):
            raise ValueError("api must be 'responses' or 'chat_completions'")
        self.model_name = model_name
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.api = api if api is not None else ("responses" if self._client.base_url.host == "api.openai.com" else "chat_completions")

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
        """Perform one request and return its raw payload and usage."""
        with translating(self.translate_error):
            if self.api == "responses":
                kwargs = _responses_request_kwargs(self.model_name, messages, schema, structured=structured)
                record_request(kwargs, api=self.api)
                return _responses_result(self._client.responses.create(**kwargs))
            # The plain role/content dicts and response_format dict are valid runtime
            # inputs that the SDK's strict TypedDict overloads do not recognize.
            kwargs = {
                "model": self.model_name,
                "messages": render_messages(messages),
                "response_format": _response_format(schema, structured=structured),
            }
            record_request(kwargs, api=self.api)
            response = self._client.chat.completions.create(**kwargs)  # pyrefly: ignore[no-matching-overload]
        return _result(response)


class AsyncOpenAIProvider(_OpenAIErrors):
    """Asynchronously call Responses or an OpenAI-compatible chat API."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api: Literal["responses", "chat_completions"] | None = None,
    ) -> None:
        """Build the provider for ``model_name`` on the given endpoint."""
        if api not in (None, "responses", "chat_completions"):
            raise ValueError("api must be 'responses' or 'chat_completions'")
        self.model_name = model_name
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.api = api if api is not None else ("responses" if self._client.base_url.host == "api.openai.com" else "chat_completions")

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
        """Perform one request and return its raw payload and usage."""
        with translating(self.translate_error):
            if self.api == "responses":
                kwargs = _responses_request_kwargs(self.model_name, messages, schema, structured=structured)
                record_request(kwargs, api=self.api)
                return _responses_result(await self._client.responses.create(**kwargs))
            # The plain role/content dicts and response_format dict are valid runtime
            # inputs that the SDK's strict TypedDict overloads do not recognize.
            kwargs = {
                "model": self.model_name,
                "messages": render_messages(messages),
                "response_format": _response_format(schema, structured=structured),
            }
            record_request(kwargs, api=self.api)
            response = await self._client.chat.completions.create(**kwargs)  # pyrefly: ignore[no-matching-overload]
        return _result(response)
