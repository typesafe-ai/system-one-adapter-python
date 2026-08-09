"""Provider error mapping and retry handling."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic_ai import ModelAPIError, ModelHTTPError
from typesafe_client import RetryConfig
from typesafe_client.api.api_client import (
    TypeSafeApiError,
    TypeSafeAuthError,
    TypeSafeTimeoutError,
    TypeSafeTokensExceededError,
    TypeSafeUnknownError,
)

ResultT = TypeVar("ResultT")

# Provider error codes and prose fragments signalling the input exceeded the model's
# context window. Providers disagree on shape, so both are checked. The codes are also
# scanned for as text, covering bodies that did not parse as JSON.
# Only codes that mean the context window specifically belong here. OpenAI's
# ``string_above_max_length`` is deliberately excluded: it is a generic field-length
# validation error, also raised for oversized tool names and similar fields.
CONTEXT_WINDOW_ERROR_CODES = frozenset(
    {"context_length_exceeded", "request_too_large"}
)
CONTEXT_WINDOW_MESSAGE_MARKERS = (
    "context window",
    "maximum context",
    "context limit",
    "prompt is too long",
    "request too large",
)


def run_with_retries(
    function: Callable[[], ResultT],
    retry: RetryConfig,
) -> tuple[ResultT, int]:
    """Run a synchronous provider call with mapped retries.

    :param function: Provider call.
    :param retry: Transient-failure retry policy.
    :return: Result and retry count.
    """
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return function(), attempt - 1
        except Exception as error:
            mapped_error, retryable = _map_error_and_decide_whether_to_retry(error)
            if attempt == retry.max_attempts or not retryable:
                raise mapped_error from error
            if delay := retry.next_delay(attempt=attempt):
                time.sleep(delay)

    raise AssertionError("retry loop did not return or raise")


async def run_with_retries_async(
    function: Callable[[], Awaitable[ResultT]],
    retry: RetryConfig,
) -> tuple[ResultT, int]:
    """Run an asynchronous provider call with mapped retries.

    :param function: Asynchronous provider call.
    :param retry: Transient-failure retry policy.
    :return: Result and retry count.
    """
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return await function(), attempt - 1
        except Exception as error:
            mapped_error, retryable = _map_error_and_decide_whether_to_retry(error)
            if attempt == retry.max_attempts or not retryable:
                raise mapped_error from error
            if delay := retry.next_delay(attempt=attempt):
                await asyncio.sleep(delay)

    raise AssertionError("retry loop did not return or raise")


def _map_error_and_decide_whether_to_retry(
    error: Exception,
) -> tuple[TypeSafeApiError, bool]:
    """Map a provider exception and decide whether the attempt may be retried.

    Timeouts are retried even though ``TypeSafeTimeoutError.is_retryable()`` reports
    False. The reference client retries every ``httpx.RequestError`` before it ever
    becomes a ``TypeSafeTimeoutError``, so consulting ``is_retryable`` alone would make
    transient connection failures non-retryable here. This is deliberately slightly
    broader than the reference: HTTP 408 and 504 responses are retried too.

    :param error: Exception raised by the provider call.
    :return: Mapped error and whether it is retryable.
    """
    mapped_error = _map_provider_exception_to_typesafe_api_error(error)
    retryable = isinstance(mapped_error, TypeSafeTimeoutError) or (
        mapped_error.is_retryable()
    )
    return mapped_error, retryable


def _map_provider_exception_to_typesafe_api_error(
    error: Exception,
) -> TypeSafeApiError:
    if isinstance(error, TypeSafeApiError):
        return error

    detail = {"message": str(error)}
    status_code = error.status_code if isinstance(error, ModelHTTPError) else None
    if status_code in (401, 403):
        return TypeSafeAuthError(detail)
    is_connection_error = isinstance(error, (httpx.RequestError, TimeoutError)) or (
        isinstance(error, ModelAPIError) and not isinstance(error, ModelHTTPError)
    )
    if is_connection_error or status_code in (408, 504):
        return TypeSafeTimeoutError(detail)

    # ``body`` is only a dict when the provider returned parseable JSON; it can also
    # be raw text or None, so fall back to scanning the stringified error.
    body = error.body if isinstance(error, ModelHTTPError) else None
    provider_error = body.get("error", body) if isinstance(body, dict) else None
    # Providers put the machine-readable code under "code" (OpenAI) or "type"
    # (Anthropic); either may be absent, hence the message-marker fallback.
    error_codes = (
        {provider_error.get("code"), provider_error.get("type")}
        if isinstance(provider_error, dict)
        else set()
    )
    error_text = str(error).lower()
    context_window_exceeded = bool(error_codes & CONTEXT_WINDOW_ERROR_CODES) or any(
        marker in error_text for marker in CONTEXT_WINDOW_MESSAGE_MARKERS
    )
    if status_code in (400, 413) and context_window_exceeded:
        return TypeSafeTokensExceededError(detail)
    return TypeSafeUnknownError(detail, status_code)
