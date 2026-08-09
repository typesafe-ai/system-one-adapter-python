"""Cached live-provider compatibility test."""

import os
from pathlib import Path

import pytest
from typesafe_client import TypeSafeClient
from typesafe_client.api.models import ChoiceQuestion, NoulQuestion, ScoreQuestion

from open_typesafe_client import OpenTypeSafeClient
from open_typesafe_client.utils.json_cache import JsonCache

JSON_CACHE_PATH = Path(__file__).with_name("json_cache.json")
if "REGENERATE_JSON_CACHE" in os.environ:
    JSON_CACHE_PATH.write_text("{}\n")
JSON_CACHE = JsonCache(JSON_CACHE_PATH)
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


@pytest.mark.parametrize(
    ("client", "model", "expected_response_data"),
    [
        pytest.param(
            OpenTypeSafeClient(),
            "gpt-4o-mini",
            {
                "model": "gpt-4o-mini",
                "answers": {
                    "positive": {"type": "noul", "noul": 0.96},
                    "stars": {
                        "type": "score",
                        "score": 3.46,
                        "confidence": 0.55,
                        "probabilities": {
                            "0": 0.01,
                            "1": 0.02,
                            "2": 0.07,
                            "3": 0.3,
                            "4": 0.6,
                        },
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 0.64,
                        "probabilities": {"fiction": 0.82, "nonfiction": 0.18},
                    },
                },
                "usage": {
                    "input_tokens": 356,
                    "output_tokens": 91,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0,
                    "latency": 0.74,
                    "max_error": 0.0,
                    "invalid_probs": 0,
                    "probability_errors": {},
                },
            },
            id="openai-probabilities",
        ),
        pytest.param(
            OpenTypeSafeClient(llm_answer_mode="discrete"),
            "gpt-4o-mini",
            {
                "model": "gpt-4o-mini",
                "answers": {
                    "positive": {"type": "noul", "noul": 1.0},
                    "stars": {
                        "type": "score",
                        "score": 4.0,
                        "confidence": 1.0,
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.0,
                            "3": 0.0,
                            "4": 1.0,
                        },
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 1.0,
                        "probabilities": {"fiction": 1.0, "nonfiction": 0.0},
                    },
                },
                "usage": {
                    "input_tokens": 331,
                    "output_tokens": 29,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0,
                    "latency": 0.51,
                    "max_error": 0.0,
                    "invalid_probs": 0,
                    "probability_errors": {},
                },
            },
            id="openai-discrete",
        ),
        pytest.param(
            OpenTypeSafeClient(),
            "claude-haiku-4-5",
            {
                "model": "claude-haiku-4-5",
                "answers": {
                    "positive": {"type": "noul", "noul": 0.94},
                    "stars": {
                        "type": "score",
                        "score": 3.23,
                        "confidence": 0.3583333333333333,
                        "probabilities": {
                            "0": 0.02,
                            "1": 0.03,
                            "2": 0.1,
                            "3": 0.4,
                            "4": 0.45,
                        },
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 0.52,
                        "probabilities": {"fiction": 0.76, "nonfiction": 0.24},
                    },
                },
                "usage": {
                    "input_tokens": 411,
                    "output_tokens": 104,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0,
                    "latency": 1.08,
                    "max_error": 0.0,
                    "invalid_probs": 0,
                    "probability_errors": {},
                },
            },
            id="anthropic-probabilities",
        ),
        pytest.param(
            OpenTypeSafeClient(llm_answer_mode="discrete"),
            "claude-haiku-4-5",
            {
                "model": "claude-haiku-4-5",
                "answers": {
                    "positive": {"type": "noul", "noul": 1.0},
                    "stars": {
                        "type": "score",
                        "score": 4.0,
                        "confidence": 1.0,
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.0,
                            "3": 0.0,
                            "4": 1.0,
                        },
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 1.0,
                        "probabilities": {"fiction": 1.0, "nonfiction": 0.0},
                    },
                },
                "usage": {
                    "input_tokens": 382,
                    "output_tokens": 31,
                    "n_retries": 0,
                    "n_retries_malformed_structure": 0,
                    "latency": 0.79,
                    "max_error": 0.0,
                    "invalid_probs": 0,
                    "probability_errors": {},
                },
            },
            id="anthropic-discrete",
        ),
        pytest.param(
            TypeSafeClient(),
            "speed_latest",
            {
                "model": "speed_latest",
                "answers": {
                    "positive": {"type": "noul", "noul": 0.98},
                    "stars": {
                        "type": "score",
                        "score": 3.635,
                        "confidence": 0.6958333333333333,
                        "probabilities": {
                            "0": 0.005,
                            "1": 0.005,
                            "2": 0.04,
                            "3": 0.25,
                            "4": 0.7,
                        },
                    },
                    "genre": {
                        "type": "choice",
                        "choice": "fiction",
                        "confidence": 0.76,
                        "probabilities": {"fiction": 0.88, "nonfiction": 0.12},
                    },
                },
                "usage": {"input_tokens": 287, "output_tokens": 47},
            },
            id="typesafe",
        ),
    ],
)
def test_live_responses_match_reference_shape(client, model, expected_response_data):
    response = JSON_CACHE(client.system_one)(model, DOCUMENT, QUESTIONS)

    assert response == expected_response_data
