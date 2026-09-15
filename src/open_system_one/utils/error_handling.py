"""Provider error mapping and retry handling."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

import httpx
import httpx2
from pydantic_ai import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)

# Keep retry budgets, status overrides, predicates, and Retry-After aligned with 0.5.x.
from typesafe_sdk._core.errors import api_error
from typesafe_sdk._core.retry import build_tenacity, build_tenacity_async

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RetryReasons:
    """Reason for performing one retry.

    :param category: Retry mechanism that requested another attempt.
    :param msg: Detailed retry cause.
    """

    category: Literal["provider_error", "malformed_structure"]
    msg: str


def run_with_retries(
    function: Callable[[], ResultT],
    retry: RetryPolicy,
    retry_reasons: list[RetryReasons] | None = None,
) -> tuple[ResultT, int]:
    """Apply the SDK retry policy to translated provider failures."""
    for attempt in build_tenacity(retry):
        with attempt:
            try:
                return function(), attempt.retry_state.attempt_number - 1
            except Exception as error:
                translated = _translate_provider_error_to_typesafe(error)
                if retry_reasons is not None:
                    # Record only failures that actually lead to another attempt.
                    attempt.retry_state.retry_object.before_sleep = (
                        lambda _, message=str(error): retry_reasons.append(
                            RetryReasons(category="provider_error", msg=message)
                        )
                    )
                if translated is error:
                    raise
                raise translated from error
    raise AssertionError("retry loop did not return or raise")


async def run_with_retries_async(
    function: Callable[[], Awaitable[ResultT]],
    retry: RetryPolicy,
    retry_reasons: list[RetryReasons] | None = None,
) -> tuple[ResultT, int]:
    """Apply the same SDK retry policy to asynchronous provider calls."""
    async for attempt in build_tenacity_async(retry):
        with attempt:
            try:
                return await function(), attempt.retry_state.attempt_number - 1
            except Exception as error:
                translated = _translate_provider_error_to_typesafe(error)
                if retry_reasons is not None:
                    attempt.retry_state.retry_object.before_sleep = (
                        lambda _, message=str(error): retry_reasons.append(
                            RetryReasons(category="provider_error", msg=message)
                        )
                    )
                if translated is error:
                    raise
                raise translated from error
    raise AssertionError("retry loop did not return or raise")


def _translate_provider_error_to_typesafe(error: Exception) -> TypeSafeError:
    """Preserve HTTP status and metadata for SDK error handling and retry policies."""
    if isinstance(error, TypeSafeError):
        return error
    if isinstance(error, ModelHTTPError):
        response = getattr(error.__cause__, "response", None)
        headers = httpx2.Headers(response.headers if response is not None else {})
        return api_error(error.status_code, error.body, headers)
    # Provider SDKs wrap transport timeouts; keep timeout-specific retry controls.
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, (httpx.TimeoutException, TimeoutError)):
            request = getattr(cause, "_request", None)
            timeouts = (
                request.extensions.get("timeout") if request is not None else None
            )
            timeout = httpx2.Timeout(**timeouts) if timeouts else httpx2.Timeout(None)
            return TypeSafeAPITimeoutError(timeout=timeout)
        cause = cause.__cause__
    if isinstance(error, (httpx.RequestError, ModelAPIError)):
        return TypeSafeAPIConnectionError(str(error))
    if isinstance(error, UnexpectedModelBehavior):
        return TypeSafeAPIResponseValidationError(
            200, str(error), httpx2.Headers(), "answers"
        )
    return TypeSafeError(str(error))
