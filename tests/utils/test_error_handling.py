"""Error handling tests."""

import asyncio

import httpx
import pytest
from pydantic_ai import ModelAPIError, ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from typesafe_client import RetryConfig
from typesafe_client.api.api_client import (
    TypeSafeApiError,
    TypeSafeAuthError,
    TypeSafeTimeoutError,
    TypeSafeTokensExceededError,
    TypeSafeUnknownError,
)
from typesafe_client.api.models import NoulQuestion

from typesafe_client_adapter import TypeSafeClientAdapter
from typesafe_client_adapter.utils.error_handling import run_with_retries

QUESTION = NoulQuestion(instructions="The review is positive.")


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


@pytest.mark.parametrize(
    ("make_error", "error_type", "expected_status_code"),
    [
        (
            lambda: ModelHTTPError(401, "test-model", {"message": "bad key"}),
            TypeSafeAuthError,
            None,
        ),
        (
            lambda: ModelHTTPError(403, "test-model", {"message": "forbidden"}),
            TypeSafeAuthError,
            None,
        ),
        (
            lambda: ModelHTTPError(
                400,
                "test-model",
                {"error": {"code": "context_length_exceeded"}},
            ),
            TypeSafeTokensExceededError,
            None,
        ),
        (
            lambda: ModelHTTPError(408, "test-model", {"message": "request timeout"}),
            TypeSafeTimeoutError,
            None,
        ),
        (
            lambda: ModelHTTPError(504, "test-model", {"message": "gateway timeout"}),
            TypeSafeTimeoutError,
            None,
        ),
        (
            lambda: ModelAPIError("test-model", "connection failed"),
            TypeSafeTimeoutError,
            None,
        ),
        (lambda: httpx.ConnectError("connection refused"), TypeSafeTimeoutError, None),
        (lambda: TimeoutError("timed out"), TypeSafeTimeoutError, None),
        (
            lambda: ModelHTTPError(500, "test-model", {"message": "boom"}),
            TypeSafeUnknownError,
            500,
        ),
        (
            lambda: ModelHTTPError(418, "test-model", {"message": "teapot"}),
            TypeSafeUnknownError,
            418,
        ),
        (
            lambda: ModelHTTPError(
                413,
                "test-model",
                {"error": {"type": "request_too_large"}},
            ),
            TypeSafeTokensExceededError,
            None,
        ),
        (
            lambda: ModelHTTPError(
                429,
                "test-model",
                {"error": {"message": "Too many tokens per minute"}},
            ),
            TypeSafeUnknownError,
            429,
        ),
        # A 400 unrelated to the context window must not be over-matched.
        (
            lambda: ModelHTTPError(
                400,
                "test-model",
                {"error": {"message": "Invalid schema for tool 'final_result'."}},
            ),
            TypeSafeUnknownError,
            400,
        ),
        (lambda: RuntimeError("something else"), TypeSafeUnknownError, None),
    ],
)
def test_provider_errors(make_error, error_type, expected_status_code):
    def raise_configured_provider_error(messages, agent_info):
        raise make_error()

    model = FunctionModel(raise_configured_provider_error, model_name="test-model")

    with pytest.raises(error_type) as raised:
        TypeSafeClientAdapter(
            structured_outputs=True,
            llm_answer_mode="probabilities",
        ).system_one(model, "document", {"answer": QUESTION})

    assert isinstance(raised.value, TypeSafeApiError)
    if error_type is TypeSafeUnknownError:
        assert raised.value.status_code == expected_status_code


@pytest.fixture
def mock_provider_model():
    """Build provider models backed by a mock transport, closing their clients after.

    :return: Factory taking the provider name and an httpx request handler.
    """
    http_clients = []

    def create_model_backed_by_mock_transport(provider, handle_request):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
        http_clients.append(http_client)
        if provider == "openai":
            return OpenAIResponsesModel(
                "gpt-4o-mini",
                provider=OpenAIProvider(api_key="test", http_client=http_client),
            )
        return AnthropicModel(
            "claude-haiku-4-5",
            provider=AnthropicProvider(api_key="test", http_client=http_client),
        )

    yield create_model_backed_by_mock_transport

    for http_client in http_clients:
        asyncio.run(http_client.aclose())


@pytest.mark.parametrize(
    ("provider", "status_code", "error_body", "expected_error"),
    [
        pytest.param(
            "openai",
            400,
            {
                "error": {
                    "message": "This model's maximum context length is 128000 tokens.",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded",
                }
            },
            TypeSafeTokensExceededError,
            id="openai-chat-completions-code",
        ),
        pytest.param(
            "openai",
            400,
            # Responses API leaves ``code`` null, so only the message identifies it.
            {
                "error": {
                    "message": (
                        "Your input exceeds the context window of this model. "
                        "Please adjust your input."
                    ),
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": None,
                }
            },
            TypeSafeTokensExceededError,
            id="openai-responses-no-code",
        ),
        pytest.param(
            "openai",
            413,
            {
                "error": {
                    "message": "Request too large for gpt-4o-mini.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "request_too_large",
                }
            },
            TypeSafeTokensExceededError,
            id="openai-request-too-large",
        ),
        pytest.param(
            "anthropic",
            400,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 220256 tokens > 200000 maximum",
                },
                "request_id": "req_test",
            },
            TypeSafeTokensExceededError,
            id="anthropic-prompt-too-long",
        ),
        pytest.param(
            "anthropic",
            400,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "input length and `max_tokens` exceed context limit: "
                        "210000 + 8192 > 200000"
                    ),
                },
                "request_id": "req_test",
            },
            TypeSafeTokensExceededError,
            id="anthropic-max-tokens-exceed-limit",
        ),
        pytest.param(
            "anthropic",
            400,
            # Unparseable body: detection must fall back to the stringified error.
            "prompt is too long: 220256 tokens > 200000 maximum",
            TypeSafeTokensExceededError,
            id="non-json-body",
        ),
        pytest.param(
            "openai",
            400,
            # Generic field-length validation, raised for oversized tool names and
            # similar fields, so it must not be read as a context-window error.
            {
                "error": {
                    "message": (
                        "Invalid 'tools[0].name': string too long. Expected a string "
                        "with maximum length 64."
                    ),
                    "type": "invalid_request_error",
                    "param": "tools[0].name",
                    "code": "string_above_max_length",
                }
            },
            TypeSafeUnknownError,
            id="openai-string-above-max-length",
        ),
        pytest.param(
            "openai",
            400,
            {
                "error": {
                    "message": "Invalid schema for response_format.",
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": None,
                }
            },
            TypeSafeUnknownError,
            id="openai-unrelated-bad-request",
        ),
    ],
)
def test_provider_context_window_errors_are_mapped(
    mock_provider_model,
    provider,
    status_code,
    error_body,
    expected_error,
):
    def return_provider_error(request):
        if isinstance(error_body, str):
            return httpx.Response(status_code, text=error_body, request=request)
        return httpx.Response(status_code, json=error_body, request=request)

    model = mock_provider_model(provider, return_provider_error)

    with pytest.raises(expected_error) as raised:
        TypeSafeClientAdapter(
            structured_outputs=True,
            llm_answer_mode="probabilities",
        ).system_one(model, "document", {"answer": QUESTION})

    if expected_error is TypeSafeUnknownError:
        assert raised.value.status_code == status_code
