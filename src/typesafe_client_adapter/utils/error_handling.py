"""Provider error mapping and retry handling."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

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
ErrorMode = Literal["pydantic_ai", "typesafe"]
RETRYABLE_HTTP_STATUS_CODES = frozenset((408, 429, 500, 502, 503, 504, 521, 522, 524))

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
    error_mode: ErrorMode = "pydantic_ai",
) -> tuple[ResultT, int]:
    """Run a synchronous provider call with optional TypeSafe translation.

    :param function: Provider call.
    :param retry: Transient-failure retry policy.
    :param error_mode: Error interface exposed after retries are exhausted.
    :return: Result and retry count.
    """
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return function(), attempt - 1
        except Exception as error:
            retryable = _is_retryable_provider_error(error)
            if attempt == retry.max_attempts or not retryable:
                if error_mode == "typesafe":
                    raise _translate_provider_error_to_typesafe(error) from error
                raise
            if delay := retry.next_delay(attempt=attempt):
                time.sleep(delay)

    raise AssertionError("retry loop did not return or raise")


async def run_with_retries_async(
    function: Callable[[], Awaitable[ResultT]],
    retry: RetryConfig,
    error_mode: ErrorMode = "pydantic_ai",
) -> tuple[ResultT, int]:
    """Run an asynchronous provider call with optional TypeSafe translation.

    :param function: Asynchronous provider call.
    :param retry: Transient-failure retry policy.
    :param error_mode: Error interface exposed after retries are exhausted.
    :return: Result and retry count.
    """
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return await function(), attempt - 1
        except Exception as error:
            retryable = _is_retryable_provider_error(error)
            if attempt == retry.max_attempts or not retryable:
                if error_mode == "typesafe":
                    raise _translate_provider_error_to_typesafe(error) from error
                raise
            if delay := retry.next_delay(attempt=attempt):
                await asyncio.sleep(delay)

    raise AssertionError("retry loop did not return or raise")


def _is_retryable_provider_error(error: Exception) -> bool:
    """Decide whether a provider failure may be retried.

    Retry classification uses the original provider error, independent of the public
    error interface selected by ``error_mode``. Timeouts are retryable even though
    ``TypeSafeTimeoutError.is_retryable()`` reports false because the reference client
    retries transport failures before translating them.

    :param error: Exception raised by the provider call.
    :return: Whether the call may be retried.
    """
    if isinstance(error, TypeSafeApiError):
        return isinstance(error, TypeSafeTimeoutError) or error.is_retryable()
    if isinstance(error, (httpx.RequestError, TimeoutError)):
        return True
    if isinstance(error, ModelAPIError) and not isinstance(error, ModelHTTPError):
        return True
    return isinstance(error, ModelHTTPError) and (
        error.status_code in RETRYABLE_HTTP_STATUS_CODES
    )


def _translate_provider_error_to_typesafe(
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
