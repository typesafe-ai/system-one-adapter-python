"""Probability distribution normalization."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

AnswerMode: TypeAlias = Literal["probabilities", "discrete"]

PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ProbabilityNormalization:
    """Result of processing one probability distribution."""

    probabilities: dict[str, float]
    error: float = 0.0
    original_probabilities: dict[str, float] | None = None


def probability_debug_data(
    probability_normalizations: Mapping[
        str,
        ProbabilityNormalization | None,
    ],
) -> dict[str, Any]:
    """Build probability diagnostics from question normalization results.

    :param probability_normalizations: Results keyed by question identifier.
    :return: Probability-related debug fields.
    """
    errors = {
        question_id: probability_normalization.error
        for question_id, probability_normalization in probability_normalizations.items()
        if probability_normalization is not None
    }
    probability_errors = {
        question_id: error
        for question_id, error in errors.items()
        if error > PROBABILITY_TOLERANCE
    }
    original_probabilities = {
        question_id: probability_normalization.original_probabilities
        for question_id, probability_normalization in probability_normalizations.items()
        if probability_normalization is not None
        and probability_normalization.original_probabilities is not None
    }
    debug_data: dict[str, Any] = {
        "max_error": max(errors.values(), default=0.0),
        "invalid_probs": len(probability_errors),
        "probability_errors": probability_errors,
    }
    if original_probabilities:
        debug_data["original_probabilities"] = original_probabilities
    return debug_data


def rescale_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Rescale probabilities to sum to 1, falling back to uniform for a zero total.

    Used wherever a value only has meaning over a true distribution (such as a score
    expected value), independent of whether the reported probabilities are normalized.

    :param probabilities: Probabilities keyed by answer.
    :return: Probabilities summing to 1.
    """
    total = sum(probabilities.values())
    if total == 0:
        uniform_probability = 1.0 / len(probabilities)
        return dict.fromkeys(probabilities, uniform_probability)
    return {
        answer: probability / total for answer, probability in probabilities.items()
    }


def normalize_probabilities_of_all_answers(
    answers: list[str],
    value: Any,
    answer_mode: AnswerMode,
    enabled: bool,
) -> ProbabilityNormalization:
    """Build and optionally normalize a probability distribution.

    :param answers: Ordered possible answers.
    :param value: LLM answer value.
    :param answer_mode: Probability or discrete LLM answer mode.
    :param enabled: Whether to normalize invalid distributions.
    :return: Normalization result.
    """
    if answer_mode == "discrete":
        selected = str(value)
        probabilities = {answer: float(answer == selected) for answer in answers}
        return ProbabilityNormalization(probabilities)

    if isinstance(value, Mapping):
        answer_probabilities = ((answer, value[answer]) for answer in answers)
    else:
        answer_probabilities = zip(answers, value, strict=True)
    original_probabilities = {
        answer: float(probability) for answer, probability in answer_probabilities
    }
    total = sum(original_probabilities.values())
    error = abs(total - 1.0)
    if not enabled or error <= PROBABILITY_TOLERANCE:
        return ProbabilityNormalization(original_probabilities, error)

    probabilities = rescale_probabilities(original_probabilities)
    return ProbabilityNormalization(probabilities, error, original_probabilities)
