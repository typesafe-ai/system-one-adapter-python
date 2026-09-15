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

import msgspec
import pytest
from pydantic_ai import ModelMessagesTypeAdapter
from pytest import param
from typesafe_sdk import (
    Choice,
    Noul,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
)

from open_system_one import AsyncOpenSystemOne, OpenSystemOne

DOCUMENT = (
    "The reviewer calls this entirely invented novel about dragons and wizards a "
    "flawless masterpiece and the best book they have ever read. They say it has no "
    "weaknesses, offer only unreserved praise, and urge everyone to read it."
)
QUESTIONS = {
    "positive": Noul(instructions="The book review is positive."),
    "rating": Score(
        instructions="How favorable the reviewer's overall assessment is.",
        criteria=[
            "The reviewer condemns the book and urges readers to avoid it.",
            "The reviewer is mostly critical and does not recommend the book.",
            "The reviewer expresses mixed or neutral feelings about the book.",
            "The reviewer praises the book overall while noting meaningful flaws.",
            "The reviewer offers unreserved praise and an emphatic recommendation.",
        ],
    ),
    "genre": Choice(
        instructions="Which genre this review is about.",
        criteria={
            "fiction": "A novel or short story.",
            "nonfiction": "A book based on facts, real events, or ideas.",
        },
    ),
}
CONTEXT_PROBE_DOCUMENT = """Catalog facts:
- marker_fen has state DORMANT.
- marker_tor has state ACTIVE.

Shipping facts:
- The parcel's handling class is CLASS_CRYSTAL.
"""
CONTEXT_PROBE_QUESTIONS = {
    "instruction_probe": Choice(
        instructions="Return the only marker whose state is ACTIVE.",
        criteria={
            "marker_fen": "The marker_fen catalog entry.",
            "marker_tor": "The marker_tor catalog entry.",
        },
    ),
    "criteria_probe": Choice(
        instructions="Return the correct opaque handling route for the parcel.",
        criteria={
            "route_7q": "Use when the handling class is CLASS_CRYSTAL.",
            "route_2m": "Use when the handling class is CLASS_STEEL.",
        },
    ),
}
MODEL_PARAMETERS = [
    param("gpt-4o-mini", id="openai"),
    param("claude-haiku-4-5", id="anthropic"),
]
STRUCTURED_OUTPUT_PARAMETERS = [
    param(False, id="prompted"),
    param(True, id="native"),
]
ANSWER_MODE_PARAMETERS = ["probabilities", "discrete"]


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
    response_data = (
        response.model_dump(mode="json")
        if hasattr(response, "model_dump")
        else {
            "model": response.model,
            "answers": msgspec.to_builtins(response.answers, str_keys=True),
            "usage": msgspec.to_builtins(response.usage),
        }
    )
    expected_answer_probabilities = {
        "positive": response.answers["positive"].noul,
        "rating": response.answers["rating"].probabilities[4],
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
        Path(__file__).with_name("expected_responses") / f"{request.node.name}.json"
    )
    if request.config.getoption("--record-mode") in (None, "none"):
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
@pytest.mark.parametrize("model", MODEL_PARAMETERS)
@pytest.mark.parametrize("structured_outputs", STRUCTURED_OUTPUT_PARAMETERS)
@pytest.mark.parametrize("answer_mode", ANSWER_MODE_PARAMETERS)
def test_live_responses_match_reference_shape(
    model,
    structured_outputs,
    answer_mode,
    request,
    vcr,
):
    client_class = AsyncOpenSystemOne if structured_outputs else OpenSystemOne
    client = client_class(
        structured_outputs=structured_outputs,
        llm_answer_mode=answer_mode,
        model=model,
    )
    if structured_outputs:
        # Raw SDK inputs and unordered rubric keys must preserve the provider request.
        questions = {
            name: msgspec.to_builtins(question) for name, question in QUESTIONS.items()
        }
        questions["rating"]["criteria"] = dict(
            reversed(list(enumerate(QUESTIONS["rating"].criteria)))
        )
        response = asyncio.run(client.system_one(state=DOCUMENT, questions=questions))
    else:
        response = client.system_one(DOCUMENT, QUESTIONS, model=model)
    assert_live_response_matches_reference(response, request)

    # Shared provider runs protect the SDK's typed views and integer score keys.
    assert isinstance(response, SystemOneResponse)
    assert isinstance(response.scores["rating"], ScoreAnswer)
    assert response.scores["rating"].legend == dict(
        enumerate(QUESTIONS["rating"].criteria)
    )
    assert set(response.scores["rating"].probabilities) == set(range(5))
    assert response.nouls["positive"] is response.answers["positive"]
    assert response.choices["genre"] is response.answers["genre"]

    # Probability-mode Choice answers are nested schemas reached through `$ref`.
    # Anthropic's native transformer previously kept the option keys but silently
    # dropped a sibling description, leaving the model without the question or
    # criteria. Inspect the final provider request so this test covers the transformed
    # schema the model actually receives rather than only Pydantic's source schema.
    if structured_outputs and answer_mode == "probabilities":
        request_body = json.loads(vcr.requests[0].body)
        # OpenAI and Anthropic place their native schema in different envelopes.
        provider_schema = (
            request_body["output_config"]["format"]["schema"]
            if "output_config" in request_body
            else request_body["text"]["format"]["schema"]
        )
        # Follow the Choice field's reference to the concrete probability-map schema.
        definitions = provider_schema["$defs"]
        choice_reference = definitions["TypeSafeAnswers"]["properties"]["genre"]["$ref"]
        choice_schema = definitions[choice_reference.rsplit("/", maxsplit=1)[-1]]

        # Require both the question and every option criterion at their final locations.
        choice_question = QUESTIONS["genre"]
        assert isinstance(choice_question, Choice)
        assert choice_question.instructions in choice_schema["description"]
        for answer, criterion in choice_question.criteria.items():
            assert criterion in choice_schema["properties"][answer]["description"]


@pytest.mark.vcr
@pytest.mark.parametrize("model", MODEL_PARAMETERS)
@pytest.mark.parametrize("structured_outputs", STRUCTURED_OUTPUT_PARAMETERS)
@pytest.mark.parametrize("answer_mode", ANSWER_MODE_PARAMETERS)
def test_live_models_follow_question_instructions_and_criteria(
    model,
    structured_outputs,
    answer_mode,
):
    client = OpenSystemOne(
        structured_outputs=structured_outputs,
        llm_answer_mode=answer_mode,
    )
    if structured_outputs:
        response = asyncio.run(
            client.system_one_async(
                CONTEXT_PROBE_DOCUMENT,
                CONTEXT_PROBE_QUESTIONS,
                model=model,
            )
        )
    else:
        response = client.system_one(
            CONTEXT_PROBE_DOCUMENT,
            CONTEXT_PROBE_QUESTIONS,
            model=model,
        )

    # Each answer is unambiguous only when its model-visible context is available, so
    # require both the expected choice and a high probability for that choice.
    expected_choices = {
        "instruction_probe": "marker_tor",
        "criteria_probe": "route_7q",
    }
    for question_id, expected_choice in expected_choices.items():
        answer = response.answers[question_id]
        assert answer.choice == expected_choice
        assert answer.probabilities[expected_choice] > 0.9


# TypeSafe is outside the live test's provider x output-mode x answer-mode param grid.
@pytest.mark.vcr
def test_live_typesafe_response_matches_reference_shape(request):
    with TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as client:
        response = client.system_one(DOCUMENT, QUESTIONS, model="speed_latest")

    assert_live_response_matches_reference(response, request)
