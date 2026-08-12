"""Provider compatibility tests replayed from recorded HTTP cassettes.

VCR (``vcrpy`` through ``pytest-recording``, configured in ``conftest.py``) intercepts
these tests at the HTTP layer: the first run records each provider exchange to a
cassette, and every run after that replays it in place of the real call, so the request
this client builds and the response it parses are both exercised for real.

Cassettes live in ``tests/cassettes`` and are replayed by default, so the whole client
stack runs against recorded provider traffic without credentials or network access.
Complete responses, including serialized PydanticAI request contexts, live in
``tests/expected_responses``.
Re-record after changing prompts, schemas, or providers::

    uv run pytest tests/test_client_with_live_apis.py --record-mode=rewrite

Recording makes real, billable API calls and needs ``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, and ``TYPESAFE_API_KEY``.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from pydantic_ai import ModelMessagesTypeAdapter
from pytest import param
from typesafe_client import TypeSafeClient
from typesafe_client.api.models import ChoiceQuestion, NoulQuestion, ScoreQuestion

from typesafe_client_adapter import TypeSafeClientAdapter

DOCUMENT = (
    "The reviewer calls this entirely invented novel about dragons and wizards a "
    "flawless masterpiece and the best book they have ever read. They say it has no "
    "weaknesses, offer only unreserved praise, and urge everyone to read it."
)
QUESTIONS = {
    "positive": NoulQuestion(instructions="The book review is positive."),
    "rating": ScoreQuestion(
        instructions="How favorable the reviewer's overall assessment is.",
        criteria=[
            "The reviewer condemns the book and urges readers to avoid it.",
            "The reviewer is mostly critical and does not recommend the book.",
            "The reviewer expresses mixed or neutral feelings about the book.",
            "The reviewer praises the book overall while noting meaningful flaws.",
            "The reviewer offers unreserved praise and an emphatic recommendation.",
        ],
    ),
    "genre": ChoiceQuestion(
        instructions="Which genre this review is about.",
        criteria={
            "fiction": "A novel or short story.",
            "nonfiction": "A book based on facts, real events, or ideas.",
        },
    ),
}


def _remove_generated_message_metadata(value):
    """Remove PydanticAI values generated independently on every replay.

    :param value: Nested response value.
    :return: Copy without timestamps, run IDs, or conversation IDs.
    """
    if isinstance(value, dict):
        return {
            key: _remove_generated_message_metadata(item)
            for key, item in value.items()
            if key not in {"timestamp", "run_id", "conversation_id"}
        }
    if isinstance(value, list):
        return [_remove_generated_message_metadata(item) for item in value]
    return value


def assert_live_response_matches_reference(response, request):
    """Validate expected answers, stable response data, and debug messages.

    :param response: Live or cassette-replayed TypeSafe response.
    :param request: Pytest request identifying the matching expected response.
    """
    response_data = response.model_dump(mode="json")
    expected_answer_probabilities = {
        "positive": response.answers["positive"].noul,
        "rating": response.answers["rating"].probabilities["4"],
        "genre": response.answers["genre"].probabilities["fiction"],
    }
    for question_id, probability in expected_answer_probabilities.items():
        assert probability > 0.9, f"Low confidence for expected {question_id} answer"

    # Latency is wall-clock and so never reproducible; assert it is plausible and drop
    # it rather than pinning a recorded value that the next run cannot match.
    latency = response_data["usage"].pop("latency", None)
    if latency is not None:
        assert 0 < latency < 120

    expected_response_path = (
        Path(__file__).with_name("expected_responses")
        / f"{request.node.name}.json"
    )
    if request.config.getoption("--record-mode") == "none":
        expected_response_data = json.loads(expected_response_path.read_text())
        assert _remove_generated_message_metadata(
            response_data
        ) == _remove_generated_message_metadata(expected_response_data)
    else:
        expected_response_path.write_text(json.dumps(response_data, indent=2) + "\n")
    if "llm_attempts" in response_data.get("debug", {}):
        for llm_query in response_data["debug"]["llm_attempts"]:
            llm_response = llm_query["llm_response"]
            ModelMessagesTypeAdapter.validate_python(
                [
                    *llm_query["messages"],
                    *([llm_response] if llm_response is not None else []),
                ]
            )


@pytest.mark.vcr
@pytest.mark.parametrize(
    "model",
    [
        param("gpt-4o-mini", id="openai"),
        param("claude-haiku-4-5", id="anthropic"),
    ],
)
@pytest.mark.parametrize(
    "structured_outputs",
    [param(False, id="prompted"), param(True, id="native")],
)
@pytest.mark.parametrize("answer_mode", ["probabilities", "discrete"])
def test_live_responses_match_reference_shape(
    model,
    structured_outputs,
    answer_mode,
    request,
):
    client = TypeSafeClientAdapter(
        structured_outputs=structured_outputs,
        llm_answer_mode=answer_mode,
    )
    if structured_outputs:
        response = asyncio.run(
            client.system_one_async(model, DOCUMENT, QUESTIONS)
        )
    else:
        response = client.system_one(model, DOCUMENT, QUESTIONS)
    assert_live_response_matches_reference(response, request)


# TypeSafe is outside the live test's provider x output-mode x answer-mode param grid.
@pytest.mark.vcr
def test_live_typesafe_response_matches_reference_shape(request):
    client = TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"])

    response = client.system_one("speed_latest", DOCUMENT, QUESTIONS)

    assert_live_response_matches_reference(response, request)
