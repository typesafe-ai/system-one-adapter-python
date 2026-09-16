"""Provider-neutral request and result types.

A provider turns a prepared message list into one raw JSON payload plus token usage.
The client owns schema construction, decoding, and retries; a provider only performs a
single model request. Sync and async providers are separate classes so each constructs
only the SDK client it uses. This is also the seam tests substitute to drive the client
without a network call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from typesafe_sdk import TypeSafeError

ProviderName = Literal["openai", "anthropic"]

# Per-call context lets providers enrich the trace without changing the provider
# protocol or sharing mutable state between concurrent evaluations.
_active_attempt: ContextVar[dict[str, Any] | None] = ContextVar("llm_attempt", default=None)


@contextmanager
def translating(translate: Callable[[Exception], TypeSafeError]) -> Iterator[None]:
    """Re-raise any provider SDK exception from the block as an SDK error.

    Raising the translated error at the provider boundary lets the retry loop stay a
    thin wrapper over the SDK's tenacity policy, whose predicate classifies SDK errors.
    """
    try:
        yield
    except Exception as error:
        translated = translate(error)
        if translated is error:
            raise
        raise translated from error


@dataclass(frozen=True)
class Message:
    """One chat message in provider-neutral form."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ProviderResult:
    """The raw JSON payload a model returned and the tokens it cost."""

    text: str
    input_tokens: int
    output_tokens: int


@runtime_checkable
class SyncProvider(Protocol):
    """Perform one synchronous model request in native or prompted output mode."""

    model_name: str

    def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one model request and return its raw payload and usage."""
        ...

    def translate_error(self, error: Exception) -> TypeSafeError:
        """Map an exception from this provider's SDK to an SDK error."""
        ...


@runtime_checkable
class AsyncProvider(Protocol):
    """Perform one asynchronous model request in native or prompted output mode."""

    model_name: str

    async def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one model request and return its raw payload and usage."""
        ...

    def translate_error(self, error: Exception) -> TypeSafeError:
        """Map an exception from this provider's SDK to an SDK error."""
        ...


@runtime_checkable
class SupportsClose(Protocol):
    """Optional lifecycle capability, separate from the request protocol."""

    def close(self) -> None: ...


@runtime_checkable
class SupportsAsyncClose(Protocol):
    """Optional async lifecycle capability, separate from the request protocol."""

    async def aclose(self) -> None: ...


def render_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Render messages into the role/content dictionaries the chat APIs expect."""
    return [{"role": m.role, "content": m.content} for m in messages]


@contextmanager
def capture_attempt(
    attempts: list[dict[str, Any]],
    provider: SyncProvider | AsyncProvider,
    messages: list[Message],
    *,
    schema: dict[str, Any],
    structured: bool,
) -> Iterator[dict[str, Any]]:
    """Snapshot one provider call, including calls that raise before returning."""
    attempt: dict[str, Any] = {
        "messages": render_messages(messages),
        "model_request_parameters": {"schema": deepcopy(schema), "structured": structured},
        "llm_response": None,
        "debug_info": {
            "model_name": provider.model_name,
            "provider": f"{type(provider).__module__}.{type(provider).__qualname__}",
        },
    }
    attempts.append(attempt)
    token = _active_attempt.set(attempt)
    try:
        yield attempt
    except Exception as error:
        attempt["debug_info"].update(error=str(error), error_type=type(error).__name__)
        raise
    finally:
        _active_attempt.reset(token)


def record_request(request: dict[str, Any], *, api: str) -> None:
    """Capture the built-in provider's SDK arguments before sending the request."""
    if (attempt := _active_attempt.get()) is not None:
        attempt["request"] = deepcopy(request)
        attempt["debug_info"]["api"] = api


def record_response(response: Any, *, finish_reason: str | None) -> None:
    """Capture the SDK response before parsing can reject incomplete output."""
    if (attempt := _active_attempt.get()) is not None:
        attempt["llm_response"] = response.model_dump(mode="json", by_alias=True)
        attempt["debug_info"]["finish_reason"] = finish_reason
