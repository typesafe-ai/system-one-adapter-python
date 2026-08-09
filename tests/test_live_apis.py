"""Provider compatibility tests replayed from recorded HTTP cassettes.

VCR (``vcrpy`` through ``pytest-recording``, configured in ``conftest.py``) intercepts
these tests at the HTTP layer: the first run records each provider exchange to a
cassette, and every run after that replays it in place of the real call, so the request
this client builds and the response it parses are both exercised for real.

Cassettes live in ``tests/cassettes`` and are replayed by default, so the whole client
stack runs against recorded provider traffic without credentials or network access.
Complete normalized debug responses live in ``tests/expected_debug``. Only volatile
timestamps, run IDs, and conversation IDs use ``<dynamic>`` placeholders.
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


def _normalize_debug_data(value):
    """Replace volatile provider-call values while retaining the complete payload."""
    if isinstance(value, dict):
        return {
            key: (
                "<dynamic>"
                if key in _DYNAMIC_DEBUG_FIELDS and item is not None
                else _normalize_debug_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_debug_data(item) for item in value]
    return value


@pytest.mark.vcr
@pytest.mark.parametrize(
    ("client", "model", "expected_response_data"),
    [
        pytest.param(
            TypeSafeClientAdapter(),
            "gpt-4o-mini",
            {
                "model": "gpt-4o-mini",
                "answers": {
                    "positive": {
                        "type": "noul",
                        "noul": 1.0
                    },
                    "stars": {
                        "type": "score",
                        "score": 4.0,
                        "confidence": 1.0,
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.0,
                            "3": 0.0,
                            "4": 1.0
                        }
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 1.0,
                        "probabilities": {
                            "fiction": 1.0,
                            "nonfiction": 0.0
                        }
                    }
                },
                "usage": {
                    "input_tokens": 621,
                    "output_tokens": 42,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0
                }
            },
            id="openai-probabilities",
        ),
        pytest.param(
            TypeSafeClientAdapter(llm_answer_mode="discrete"),
            "gpt-4o-mini",
            {
                "model": "gpt-4o-mini",
                "answers": {
                    "positive": {
                        "type": "noul",
                        "noul": 1.0
                    },
                    "stars": {
                        "type": "score",
                        "score": 4.0,
                        "confidence": 1.0,
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.0,
                            "3": 0.0,
                            "4": 1.0
                        }
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 1.0,
                        "probabilities": {
                            "fiction": 1.0,
                            "nonfiction": 0.0
                        }
                    }
                },
                "usage": {
                    "input_tokens": 365,
                    "output_tokens": 16,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0
                }
            },
            id="openai-discrete",
        ),
        pytest.param(
            TypeSafeClientAdapter(),
            "claude-haiku-4-5",
            {
                "model": "claude-haiku-4-5",
                "answers": {
                    "positive": {
                        "type": "noul",
                        "noul": 0.95
                    },
                    "stars": {
                        "type": "score",
                        "score": 3.1500000000000004,
                        "confidence": 0.675,
                        "probabilities": {
                            "0": 0.01,
                            "1": 0.02,
                            "2": 0.05,
                            "3": 0.65,
                            "4": 0.27
                        }
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 0.0,
                        "probabilities": {
                            "fiction": 0.5,
                            "nonfiction": 0.5
                        }
                    }
                },
                "usage": {
                    "input_tokens": 665,
                    "output_tokens": 118,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0
                }
            },
            id="anthropic-probabilities",
        ),
        pytest.param(
            TypeSafeClientAdapter(llm_answer_mode="discrete"),
            "claude-haiku-4-5",
            {
                "model": "claude-haiku-4-5",
                "answers": {
                    "positive": {
                        "type": "noul",
                        "noul": 1.0
                    },
                    "stars": {
                        "type": "score",
                        "score": 3.0,
                        "confidence": 1.0,
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.0,
                            "3": 1.0,
                            "4": 0.0
                        }
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 1.0,
                        "probabilities": {
                            "fiction": 1.0,
                            "nonfiction": 0.0
                        }
                    }
                },
                "usage": {
                    "input_tokens": 405,
                    "output_tokens": 29,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0
                }
            },
            id="anthropic-discrete",
        ),
        pytest.param(
            TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]),
            "speed_latest",
            {
                "model": "speed_v10_mango_loco",
                "answers": {
                    "positive": {
                        "type": "noul",
                        "noul": 0.99
                    },
                    "stars": {
                        "type": "score",
                        "score": 3.15,
                        "confidence": 0.86,
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.01,
                            "3": 0.83,
                            "4": 0.16
                        },
                        "legend": {
                            "0": "Horrendous. Unreadable garbage.",
                            "1": "Pretty bad, but theoretically readable.",
                            "2": "Acceptable, but just barely.",
                            "3": "Pretty good. Worth reading but not perfect.",
                            "4": "Transcendent and impactful. A must read."
                        }
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 0.79,
                        "probabilities": {
                            "fiction": 0.89,
                            "nonfiction": 0.11
                        }
                    }
                },
                "usage": {
                    "input_tokens": 396,
                    "output_tokens": 55
                }
            },
            id="typesafe",
        ),
    ],
)
def test_live_responses_match_reference_shape(
    client,
    model,
    expected_response_data,
    request,
):
    response = client.system_one(model, DOCUMENT, QUESTIONS)
    response_data = response.model_dump(mode="json")
    normalized_response_data = deepcopy(response_data)

    # Latency is wall-clock and so never reproducible; assert it is plausible and drop
    # it rather than pinning a recorded value that the next run cannot match.
    latency = normalized_response_data["usage"].pop("latency", None)
    if latency is not None:
        assert 0 < latency < 120

    debug = response_data.get("debug")
    if isinstance(client, TypeSafeClientAdapter):
        assert debug["max_error"] == 0
        assert debug["invalid_probs"] == 0
        assert debug["probability_errors"] == {}
        assert len(debug["query"]) == 1
        query_entry = debug["query"][0]
        serialized_messages = json.dumps(query_entry["llm_query"]["messages"])
        assert "Evaluate every question" in serialized_messages
        assert DOCUMENT in serialized_messages
        assert "json_schema" in query_entry["llm_query"]["model_request_parameters"][
            "output_object"
        ]
        assert query_entry["llm_response"]["kind"] == "response"
        assert query_entry["debug_info"]["model_name"] == model
        debug_snapshot_path = (
            Path(__file__).with_name("expected_debug")
            / f"{request.node.callspec.id}.json"
        )
        expected_response_data = {
            **expected_response_data,
            "debug": json.loads(debug_snapshot_path.read_text()),
        }
        normalized_response_data["debug"] = _normalize_debug_data(debug)
    else:
        assert debug is None

    assert normalized_response_data == expected_response_data
