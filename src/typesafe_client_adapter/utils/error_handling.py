"""Provider error mapping and retry handling."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar, cast

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
from typesafe_client.api.retry import RetryLoop, TypeSafeMaxRetriesExceededError

_TOKEN_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window",
    "maximum context",
    "prompt is too long",
    "request too large",
    "too many tokens",
)

ResultT = TypeVar("ResultT")


class _RetryableTypeSafeTimeoutError(TypeSafeTimeoutError):
    """Adapter timeout error compatible with the reference retry loop."""

    def is_retryable(self) -> bool:
        return True


def run_with_retries(
    function: Callable[[], ResultT],
    retry: RetryConfig,
) -> tuple[ResultT, int]:
    """Run a synchronous provider call with mapped retries.

    :param function: Provider call.
    :param retry: Transient-failure retry policy.
    :return: Result and retry count.
    """
    retry_loop = RetryLoop(retry)

    def call_with_mapped_provider_errors() -> ResultT:
        with _map_provider_errors():
            return function()

    try:
        result = retry_loop.retry(call_with_mapped_provider_errors)
    except TypeSafeMaxRetriesExceededError as retry_error:
        mapped_error = retry_error.__cause__
        assert isinstance(mapped_error, TypeSafeApiError)
        raise mapped_error from mapped_error.__cause__
    return cast(ResultT, result), retry_loop.attempt - 1


async def run_with_retries_async(
    function: Callable[[], Awaitable[ResultT]],
    retry: RetryConfig,
) -> tuple[ResultT, int]:
    """Run an asynchronous provider call with mapped retries.

    :param function: Asynchronous provider call.
    :param retry: Transient-failure retry policy.
    :return: Result and retry count.
    """
    retry_loop = RetryLoop(retry)

    async def call_with_mapped_provider_errors() -> ResultT:
        with _map_provider_errors():
            return await function()

    try:
        result = await retry_loop.async_retry(call_with_mapped_provider_errors)
    except TypeSafeMaxRetriesExceededError as retry_error:
        mapped_error = retry_error.__cause__
        assert isinstance(mapped_error, TypeSafeApiError)
        raise mapped_error from mapped_error.__cause__
    return cast(ResultT, result), retry_loop.attempt - 1


@contextmanager
def _map_provider_errors() -> Iterator[None]:
    """Map errors raised inside one PydanticAI provider attempt."""
    try:
        yield
    except TypeSafeApiError:
        raise
    except Exception as error:
        raise _map_provider_exception_to_typesafe_api_error(error) from error


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
        return _RetryableTypeSafeTimeoutError(detail)

    error_text = str(error).lower()
    if any(marker in error_text for marker in _TOKEN_ERROR_MARKERS):
        return TypeSafeTokensExceededError(detail)
    return TypeSafeUnknownError(detail, status_code)
