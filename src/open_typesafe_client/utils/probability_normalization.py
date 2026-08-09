"""Probability distribution normalization."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

AnswerMode: TypeAlias = Literal["probabilities", "discrete"]

PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ProbabilityNormalization:
    """Result of processing one probability distribution."""

    probabilities: dict[str, float]
    error: float = 0.0
    original_probabilities: dict[str, float] | None = None


@dataclass
class ProbabilityNormalizationStats:
    """Aggregate normalization telemetry across questions."""

    probability_errors: dict[str, float] = field(default_factory=dict)
    original_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(
        self,
        question_id: str,
        probability_normalization: ProbabilityNormalization | None,
    ) -> None:
        """Add one question's normalization result.

        :param question_id: Question identifier.
        :param probability_normalization: Optional distribution result.
        """
        if probability_normalization is None:
            return
        if probability_normalization.error > PROBABILITY_TOLERANCE:
            self.probability_errors[question_id] = probability_normalization.error
        if probability_normalization.original_probabilities is not None:
            self.original_probabilities[question_id] = (
                probability_normalization.original_probabilities
            )

    def usage_data(self) -> dict[str, Any]:
        """Return probability telemetry for ``Usage``."""
        usage_data: dict[str, Any] = {
            "max_error": max(self.probability_errors.values(), default=0.0),
            "invalid_probs": len(self.probability_errors),
            "probability_errors": self.probability_errors,
        }
        if self.original_probabilities:
            usage_data["original_probabilities"] = self.original_probabilities
        return usage_data


def to_distribution(probabilities: dict[str, float]) -> dict[str, float]:
    """Rescale probabilities to sum to 1, falling back to uniform for a zero total.

    Used wherever a value only has meaning over a true distribution (such as a score
    expected value), independent of whether the reported probabilities are normalized.

    :param probabilities: Probabilities keyed by label.
    :return: Probabilities summing to 1.
    """
    total = sum(probabilities.values())
    if total == 0:
        uniform_probability = 1.0 / len(probabilities)
        return dict.fromkeys(probabilities, uniform_probability)
    return {
        label: probability / total for label, probability in probabilities.items()
    }


def normalize_probabilities(
    labels: list[str],
    value: Any,
    answer_mode: AnswerMode,
    enabled: bool,
) -> ProbabilityNormalization:
    """Build and optionally normalize a probability distribution.

    :param labels: Ordered distribution labels.
    :param value: LLM answer value.
    :param answer_mode: Probability or discrete LLM answer mode.
    :param enabled: Whether to normalize invalid distributions.
    :return: Normalization result.
    """
    if answer_mode == "discrete":
        selected = str(value)
        probabilities = {label: float(label == selected) for label in labels}
        return ProbabilityNormalization(probabilities)

    original_probabilities = {label: float(value[label]) for label in labels}
    total = sum(original_probabilities.values())
    error = abs(total - 1.0)
    if not enabled or error <= PROBABILITY_TOLERANCE:
        return ProbabilityNormalization(original_probabilities, error)

    probabilities = to_distribution(original_probabilities)
    return ProbabilityNormalization(probabilities, error, original_probabilities)
