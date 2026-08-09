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
            mapped_error = _map_provider_exception_to_typesafe_api_error(error)
            retryable = isinstance(mapped_error, TypeSafeTimeoutError) or (
                mapped_error.is_retryable()
            )
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
            mapped_error = _map_provider_exception_to_typesafe_api_error(error)
            retryable = isinstance(mapped_error, TypeSafeTimeoutError) or (
                mapped_error.is_retryable()
            )
            if attempt == retry.max_attempts or not retryable:
                raise mapped_error from error
            if delay := retry.next_delay(attempt=attempt):
                await asyncio.sleep(delay)

    raise AssertionError("retry loop did not return or raise")

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

    body = error.body if isinstance(error, ModelHTTPError) else None
    provider_error = body.get("error", body) if isinstance(body, dict) else {}
    context_window_exceeded = isinstance(provider_error, dict) and (
        provider_error.get("code") == "context_length_exceeded"
        or str(provider_error.get("message", "")).lower().startswith(
            "prompt is too long"
        )
    )
    if status_code == 400 and context_window_exceeded:
        return TypeSafeTokensExceededError(detail)
    return TypeSafeUnknownError(detail, status_code)
