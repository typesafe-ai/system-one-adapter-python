"""Probability normalization tests."""

import pytest

from open_system_one.utils.probability_normalization import (
    normalize_probabilities_of_all_answers,
    probability_debug_data,
)


@pytest.mark.parametrize(
    (
        "normalization_enabled",
        "raw_probability",
        "expected_probability",
        "expected_originals",
        "expected_max_error",
        "expected_probability_errors",
    ),
    [
        (
            False,
            0.2,
            0.2,
            None,
            0.6,
            {"stars": 0.6, "genre": 0.6},
        ),
        (
            True,
            0.2,
            0.5,
            {
                "stars": {"0": 0.2, "1": 0.2},
                "genre": {"fiction": 0.2, "nonfiction": 0.2},
            },
            0.6,
            {"stars": 0.6, "genre": 0.6},
        ),
        (False, 0.50000025, 0.50000025, None, 5e-7, {}),
    ],
)
def test_probability_normalization_and_debug_data(
    normalization_enabled,
    raw_probability,
    expected_probability,
    expected_originals,
    expected_max_error,
    expected_probability_errors,
):
    score_probabilities = {"0": raw_probability, "1": raw_probability}
    choice_probabilities = {
        "fiction": raw_probability,
        "nonfiction": raw_probability,
    }
    probability_normalizations = {
        "positive": None,
        "stars": normalize_probabilities_of_all_answers(
            ["0", "1"],
            score_probabilities,
            "probabilities",
            normalization_enabled,
        ),
        "genre": normalize_probabilities_of_all_answers(
            ["fiction", "nonfiction"],
            choice_probabilities,
            "probabilities",
            normalization_enabled,
        ),
    }

    assert probability_normalizations["stars"].probabilities == pytest.approx(
        {"0": expected_probability, "1": expected_probability}
    )
    assert probability_normalizations["genre"].probabilities == pytest.approx(
        {"fiction": expected_probability, "nonfiction": expected_probability}
    )

    debug_data = probability_debug_data(probability_normalizations)
    assert debug_data["max_error"] == pytest.approx(expected_max_error)
    assert debug_data["invalid_probs"] == len(expected_probability_errors)
    assert debug_data["probability_errors"] == pytest.approx(
        expected_probability_errors
    )
    assert debug_data.get("original_probabilities") == expected_originals
