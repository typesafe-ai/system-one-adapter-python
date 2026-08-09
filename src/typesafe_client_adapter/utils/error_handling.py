"""Provider error mapping and retry classification."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anthropic
import httpx
import openai
from typesafe_client import RetryConfig
from typesafe_client.api.api_client import (
    TypeSafeApiError,
    TypeSafeAuthError,
    TypeSafeTimeoutError,
    TypeSafeTokensExceededError,
    TypeSafeUnknownError,
)

_AUTH_ERRORS = (openai.AuthenticationError, anthropic.AuthenticationError)
_CONNECTION_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    httpx.RequestError,
    TimeoutError,
)
_TOKEN_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window",
    "maximum context",
    "prompt is too long",
    "request too large",
    "too many tokens",
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
        except Exception as error:  # noqa: BLE001 - unknown providers vary
            _raise_for_nonretryable(error, attempt, retry.max_attempts)
            delay = retry.next_delay(attempt=attempt)
            if delay:
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
        except Exception as error:  # noqa: BLE001 - unknown providers vary
            _raise_for_nonretryable(error, attempt, retry.max_attempts)
            delay = retry.next_delay(attempt=attempt)
            if delay:
                await asyncio.sleep(delay)

    raise AssertionError("retry loop did not return or raise")


def _raise_for_nonretryable(
    error: Exception,
    attempt: int,
    max_attempts: int,
) -> None:
    """Raise a mapped error when another provider attempt is not allowed.

    :param error: Original provider or PydanticAI error.
    :param attempt: Current one-based attempt number.
    :param max_attempts: Maximum provider attempts.
    """
    mapped_error = _map_error(error)
    if attempt >= max_attempts or not _retryable(error, mapped_error):
        raise mapped_error from error


def _map_error(error: Exception) -> TypeSafeApiError:
    if isinstance(error, TypeSafeApiError):
        return error

    chain = _error_chain(error)
    detail = {"message": str(error)}
    if any(isinstance(item, _AUTH_ERRORS) for item in chain):
        return TypeSafeAuthError(detail)

    status_code = _status_code(chain)
    if status_code in (401, 403):
        return TypeSafeAuthError(detail)
    if any(isinstance(item, _CONNECTION_ERRORS) for item in chain) or status_code in (
        408,
        504,
    ):
        return TypeSafeTimeoutError(detail)

    error_text = " ".join(str(item).lower() for item in chain)
    if any(marker in error_text for marker in _TOKEN_ERROR_MARKERS):
        return TypeSafeTokensExceededError(detail)
    return TypeSafeUnknownError(detail, status_code)


def _retryable(error: Exception, mapped_error: TypeSafeApiError) -> bool:
    chain = _error_chain(error)
    if any(isinstance(item, _CONNECTION_ERRORS) for item in chain):
        return True
    if _status_code(chain) in TypeSafeUnknownError.retryable_status_codes:
        return True
    return (
        isinstance(mapped_error, TypeSafeUnknownError) and mapped_error.is_retryable()
    )


def _error_chain(error: Exception) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _status_code(chain: list[BaseException]) -> int | None:
    return next(
        (item.status_code for item in chain if hasattr(item, "status_code")),
        None,
    )
