"""Provider compatibility tests replayed from recorded HTTP cassettes.

VCR (``vcrpy`` through ``pytest-recording``, configured in ``conftest.py``) intercepts
these tests at the HTTP layer: the first run records each provider exchange to a
cassette, and every run after that replays it in place of the real call, so the request
this client builds and the response it parses are both exercised for real.

Cassettes live in ``tests/cassettes`` and are replayed by default, so the whole client
stack runs against recorded provider traffic without credentials or network access.
Complete normalized responses live in ``tests/expected_responses``. Only volatile
debug timestamps, run IDs, and conversation IDs use ``<dynamic>`` placeholders.
Re-record after changing prompts, schemas, or providers::

    uv run pytest tests/test_live_apis.py --record-mode=rewrite

Recording makes real, billable API calls and needs ``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, and ``TYPESAFE_API_KEY``.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from typesafe_client import TypeSafeClient
from typesafe_client.api.models import ChoiceQuestion, NoulQuestion, ScoreQuestion

from typesafe_client_adapter import TypeSafeClientAdapter

DOCUMENT = "This book was a delight to read."
QUESTIONS = {
    "positive": NoulQuestion(instructions="The book review is positive."),
    "stars": ScoreQuestion(
        instructions="Star rating for the book based on the review.",
        criteria=[
            "Horrendous. Unreadable garbage.",
            "Pretty bad, but theoretically readable.",
            "Acceptable, but just barely.",
            "Pretty good. Worth reading but not perfect.",
            "Transcendent and impactful. A must read.",
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

_DYNAMIC_DEBUG_FIELDS = frozenset({"timestamp", "run_id", "conversation_id"})


def _normalize_response_data(value):
    """Replace volatile provider-call values while retaining the complete response."""
    if isinstance(value, dict):
        return {
            key: (
                "<dynamic>"
                if key in _DYNAMIC_DEBUG_FIELDS and item is not None
                else _normalize_response_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_response_data(item) for item in value]
    return value


@pytest.mark.vcr
@pytest.mark.parametrize(
    ("client", "model"),
    [
        pytest.param(
            TypeSafeClientAdapter(),
            "gpt-4o-mini",
            id="openai-probabilities",
        ),
        pytest.param(
            TypeSafeClientAdapter(llm_answer_mode="discrete"),
            "gpt-4o-mini",
            id="openai-discrete",
        ),
        pytest.param(
            TypeSafeClientAdapter(),
            "claude-haiku-4-5",
            id="anthropic-probabilities",
        ),
        pytest.param(
            TypeSafeClientAdapter(llm_answer_mode="discrete"),
            "claude-haiku-4-5",
            id="anthropic-discrete",
        ),
        pytest.param(
            TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]),
            "speed_latest",
            id="typesafe",
        ),
    ],
)
def test_live_responses_match_reference_shape(
    client,
    model,
    request,
):
    response = client.system_one(model, DOCUMENT, QUESTIONS)
    response_data = response.model_dump(mode="json")
    normalized_response_data = _normalize_response_data(deepcopy(response_data))

    # Latency is wall-clock and so never reproducible; assert it is plausible and drop
    # it rather than pinning a recorded value that the next run cannot match.
    latency = normalized_response_data["usage"].pop("latency", None)
    if latency is not None:
        assert 0 < latency < 120

    expected_response_path = (
        Path(__file__).with_name("expected_responses")
        / f"{request.node.callspec.id}.json"
    )
    expected_response_data = json.loads(expected_response_path.read_text())
    assert normalized_response_data == expected_response_data
    assert json.dumps(normalized_response_data, sort_keys=True) == json.dumps(
        expected_response_data,
        sort_keys=True,
    )
