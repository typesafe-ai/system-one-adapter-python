"""Open TypeSafe public API."""

from typesafe_client import RetryConfig
from typesafe_client.api.api_client import (
    TypeSafeApiError,
    TypeSafeAuthError,
    TypeSafeTimeoutError,
    TypeSafeTokensExceededError,
    TypeSafeUnknownError,
)
from typesafe_client.api.retry import NoRetries

from .client import TypeSafeClientAdapter
from .utils.error_handling import ErrorMode

__all__ = [
    "ErrorMode",
    "NoRetries",
    "RetryConfig",
    "TypeSafeApiError",
    "TypeSafeAuthError",
    "TypeSafeClientAdapter",
    "TypeSafeTimeoutError",
    "TypeSafeTokensExceededError",
    "TypeSafeUnknownError",
]
