"""End-to-end client tests through PydanticAI's model interface."""

import asyncio
import json

import httpx
import pytest
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage
from typesafe_client import RetryConfig, TypeSafeClient
from typesafe_client.api.api_client import (
    TypeSafeApiError,
    TypeSafeAuthError,
    TypeSafeTimeoutError,
    TypeSafeTokensExceededError,
    TypeSafeUnknownError,
)
from typesafe_client.api.models import ChoiceQuestion, NoulQuestion, ScoreQuestion

from typesafe_client_adapter import TypeSafeClientAdapter, deserialize_llm_attempt
from typesafe_client_adapter.utils.error_handling import run_with_retries

QUESTIONS = {
    "positive": NoulQuestion(instructions="The review is positive."),
    "stars": ScoreQuestion(instructions="Rating.", criteria=["Bad.", "Good."]),
    "genre": ChoiceQuestion(
        instructions="Genre.",
        criteria={"fiction": "A story.", "nonfiction": "Facts."},
    ),
}


def test_client_is_typesafe_client():
    assert issubclass(TypeSafeClientAdapter, TypeSafeClient)


def model_response(response_data, expected_output_mode, expected_descriptions=()):
    """Return a local model producing ``response_data`` in the requested output mode."""

    def return_configured_model_response(messages, agent_info):
        parameters = agent_info.model_request_parameters
        assert parameters.output_mode == expected_output_mode
        prompted_output_instructions = parameters.prompted_output_instructions
        if expected_output_mode == "prompted":
            assert prompted_output_instructions, (
                "Prompted output mode requires output instructions"
            )
            assert (
                sum(
                    instruction_part.content == prompted_output_instructions
                    for instruction_part in parameters.instruction_parts or []
                )
                == 1
            )
        else:
            assert prompted_output_instructions is None
        output_schema = parameters.output_object.json_schema
        for expected_description in expected_descriptions:
            assert expected_description in json.dumps(output_schema)
        if not agent_info.output_tools:
            parts = [TextPart(json.dumps(response_data))]
        else:
            parts = [ToolCallPart(agent_info.output_tools[0].name, response_data)]
        return ModelResponse(
            parts,
            usage=RequestUsage(input_tokens=11, output_tokens=7),
        )

    return FunctionModel(
        return_configured_model_response,
        model_name="test-model",
        profile={"supports_json_schema_output": True},
    )


@pytest.mark.parametrize("structured_outputs", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("answer_mode", "response_data"),
    [
        (
            "probabilities",
            {
                "answers": {
                    "positive": 0.8,
                    "stars": {"0": 0.1, "1": 0.9},
                    "genre": {"fiction": 0.5, "nonfiction": 0.5},
                }
            },
        ),
        (
            "discrete",
            {"answers": {"positive": True, "stars": 1, "genre": "fiction"}},
        ),
    ],
)
def test_system_one(answer_mode, response_data, async_call, structured_outputs):
    client = TypeSafeClientAdapter(
        structured_outputs=structured_outputs,
        llm_answer_mode=answer_mode,
    )
    expected_output_mode = "native" if structured_outputs else "prompted"
    expected_descriptions = (
        (
            "Score levels, answer with the integer:\\n0 = Bad.\\n1 = Good.",
            "Choice labels, answer with one label:\\nfiction = A story.\\n"
            "nonfiction = Facts.",
        )
        if answer_mode == "discrete"
        else ()
    )
    model = model_response(
        response_data,
        expected_output_mode,
        expected_descriptions,
    )

    if async_call:
        response = asyncio.run(
            client.system_one_async(model, "A delightful novel.", QUESTIONS)
        )
    else:
        response = client.system_one(model, "A delightful novel.", QUESTIONS)

    assert response.model == "test-model"
    assert response.answers["positive"].type == "noul"
    assert response.answers["positive"].noul == pytest.approx(
        0.8 if answer_mode == "probabilities" else 1.0
    )
    assert response.answers["stars"].score == pytest.approx(
        0.9 if answer_mode == "probabilities" else 1.0
    )
    assert response.answers["stars"].confidence == pytest.approx(
        0.8 if answer_mode == "probabilities" else 1.0
    )
    assert response.answers["genre"].choice == "fiction"
    assert response.answers["genre"].confidence == pytest.approx(
        0.0 if answer_mode == "probabilities" else 1.0
    )
    assert response.answers["stars"].probabilities == pytest.approx(
        {"0": 0.1, "1": 0.9} if answer_mode == "probabilities" else {"0": 0.0, "1": 1.0}
    )
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.n_retries == 0
    assert response.usage.n_retries_malformed_structure == 0
    assert response.usage.latency >= 0
    assert not hasattr(response.usage, "max_error")
    assert not hasattr(response.usage, "invalid_probs")
    assert not hasattr(response.usage, "probability_errors")
    assert response.debug["max_error"] == 0
    assert response.debug["invalid_probs"] == 0
    assert response.debug["probability_errors"] == {}

    llm_query = response.debug["llm_attempts"][0]
    llm_response = llm_query["llm_response"]
    serialized_query = json.dumps(llm_query)
    assert "Evaluate every question" in serialized_query
    assert "A delightful novel." in serialized_query
    assert llm_query["messages"][-1]["kind"] == "request"
    model_request_parameters = llm_query["model_request_parameters"]
    assert model_request_parameters["output_mode"] == expected_output_mode
    assert "positive" in json.dumps(model_request_parameters["output_object"])
    assert llm_response["kind"] == "response"
    restored_messages = ModelMessagesTypeAdapter.validate_python(
        [*llm_query["messages"], llm_response]
    )
    assert len(restored_messages) == 2
    assert llm_query["debug_info"]["model_name"] == "test-model"

    model_request_arguments = deserialize_llm_attempt(llm_query)
    replayed_response = asyncio.run(
        model.request(*model_request_arguments)
    )
    assert replayed_response.parts == restored_messages[-1].parts


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
                400, "test-model", {"message": "context_length_exceeded"}
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
        (lambda: RuntimeError("something else"), TypeSafeUnknownError, None),
    ],
)
def test_provider_errors(make_error, error_type, expected_status_code):
    def raise_configured_provider_error(messages, agent_info):
        raise make_error()

    model = FunctionModel(raise_configured_provider_error, model_name="test-model")

    with pytest.raises(error_type) as raised:
        TypeSafeClientAdapter().system_one(
            model, "document", {"answer": QUESTIONS["positive"]}
        )

    assert isinstance(raised.value, TypeSafeApiError)
    if error_type is TypeSafeUnknownError:
        assert raised.value.status_code == expected_status_code


@pytest.mark.parametrize("async_call", [False, True])
def test_transient_errors_are_retried(async_call):
    calls = 0
    success_model = model_response(
        {"answers": {"answer": 0.75}}, expected_output_mode="prompted"
    )

    def fail_first_provider_attempt(messages, agent_info):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelHTTPError(503, "test-model", {"message": "unavailable"})
        return success_model.function(messages, agent_info)

    model = FunctionModel(fail_first_provider_attempt, model_name="test-model")
    retry = RetryConfig(max_attempts=2, initial_backoff=0, jitter=False)
    client = TypeSafeClientAdapter(retry=retry)
    questions = {"answer": QUESTIONS["positive"]}

    if async_call:
        response = asyncio.run(client.system_one_async(model, "document", questions))
    else:
        response = client.system_one(model, "document", questions)

    assert calls == 2
    assert response.usage.n_retries == 1
    assert response.usage.n_retries_malformed_structure == 0
    assert len(response.debug["llm_attempts"]) == 2
    assert response.debug["llm_attempts"][0]["llm_response"] is None
    assert (
        response.debug["llm_attempts"][0]["debug_info"]["error_type"]
        == "ModelHTTPError"
    )
    assert response.debug["llm_attempts"][1]["llm_response"]["kind"] == "response"


@pytest.mark.parametrize("async_call", [False, True])
def test_retries_are_exhausted(async_call):
    calls = 0

    def raise_retryable_provider_error(messages, agent_info):
        nonlocal calls
        calls += 1
        raise ModelHTTPError(503, "test-model", {"message": "unavailable"})

    model = FunctionModel(raise_retryable_provider_error, model_name="test-model")
    retry = RetryConfig(max_attempts=3, initial_backoff=0, jitter=False)
    client = TypeSafeClientAdapter(retry=retry)
    questions = {"answer": QUESTIONS["positive"]}

    with pytest.raises(TypeSafeUnknownError) as raised:
        if async_call:
            asyncio.run(client.system_one_async(model, "document", questions))
        else:
            client.system_one(model, "document", questions)

    assert calls == 3
    assert raised.value.status_code == 503


@pytest.mark.parametrize("async_call", [False, True])
def test_usage_includes_tokens_spent_on_failed_attempts(async_call):
    """Tokens burned by an attempt that later failed are still billed to the caller."""
    calls = 0

    def simulate_malformed_transient_then_successful_attempts(messages, agent_info):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Burns tokens, then fails structural validation.
            return ModelResponse(
                [TextPart(json.dumps({"answers": "not-an-object"}))],
                usage=RequestUsage(input_tokens=100, output_tokens=50),
            )
        if calls == 2:
            # The corrective retry dies transiently, failing the whole attempt.
            raise ModelHTTPError(503, "test-model", {"message": "unavailable"})
        return ModelResponse(
            [TextPart(json.dumps({"answers": {"answer": 0.75}}))],
            usage=RequestUsage(input_tokens=100, output_tokens=50),
        )

    model = FunctionModel(
        simulate_malformed_transient_then_successful_attempts,
        model_name="test-model",
    )
    retry = RetryConfig(max_attempts=2, initial_backoff=0, jitter=False)
    client = TypeSafeClientAdapter(retry=retry, n_retry_malformed_structure=1)
    questions = {"answer": QUESTIONS["positive"]}

    if async_call:
        response = asyncio.run(client.system_one_async(model, "document", questions))
    else:
        response = client.system_one(model, "document", questions)

    assert calls == 3
    assert response.usage.input_tokens == 200
    assert response.usage.output_tokens == 100
    assert response.usage.n_retries == 1
    assert response.usage.n_retries_malformed_structure == 1
    assert len(response.debug["llm_attempts"]) == 3


@pytest.mark.parametrize(
    "questions",
    [
        pytest.param({}, id="no-questions"),
        pytest.param(
            {"stars": ScoreQuestion(instructions="Rating.", criteria=[])},
            id="empty-score-criteria",
        ),
        pytest.param(
            {"genre": ChoiceQuestion(instructions="Genre.", criteria={})},
            id="empty-choice-criteria",
        ),
    ],
)
def test_invalid_questions_are_rejected(questions):
    model = model_response({"answers": {}}, expected_output_mode="prompted")

    with pytest.raises(ValueError):
        TypeSafeClientAdapter().system_one(model, "document", questions)


@pytest.mark.parametrize("status_code", [408, 504])
def test_status_errors_are_retried(status_code):
    class ProviderStatusError(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    calls = 0

    def fail_first_status_attempt():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderStatusError(status_code)
        return "success"

    result, n_retries = run_with_retries(
        fail_first_status_attempt,
        RetryConfig(max_attempts=2, initial_backoff=0, jitter=False),
    )

    assert result == "success"
    assert calls == 2
    assert n_retries == 1


def test_malformed_structure_is_retried():
    calls = 0

    def return_malformed_then_valid_response(messages, agent_info):
        nonlocal calls
        calls += 1
        answers = {} if calls == 1 else {"answer": 0.75}
        return ModelResponse(
            [TextPart(json.dumps({"answers": answers}))],
            usage=RequestUsage(input_tokens=11, output_tokens=7),
        )

    model = FunctionModel(
        return_malformed_then_valid_response,
        model_name="test-model",
    )
    response = TypeSafeClientAdapter(n_retry_malformed_structure=1).system_one(
        model,
        "document",
        {"answer": QUESTIONS["positive"]},
    )

    assert calls == 2
    assert response.usage.n_retries == 0
    assert response.usage.n_retries_malformed_structure == 1
    assert len(response.debug["llm_attempts"]) == 2
