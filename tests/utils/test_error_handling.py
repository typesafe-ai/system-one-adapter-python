"""Error handling tests."""

import pytest
from pydantic_ai import ModelHTTPError
from typesafe_client import RetryConfig

from typesafe_client_adapter.utils.error_handling import run_with_retries


@pytest.mark.parametrize("status_code", [408, 504])
def test_status_errors_are_retried(status_code):
    calls = 0

    def fail_first_status_attempt():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelHTTPError(status_code, "test-model")
        return "success"

    result, n_retries = run_with_retries(
        fail_first_status_attempt,
        RetryConfig(max_attempts=2, initial_backoff=0, jitter=False),
    )

    assert result == "success"
    assert calls == 2
    assert n_retries == 1
