"""Confidence metrics for probability distributions."""


def score_confidence(probs: list[float]) -> float:
    """Measure score concentration around its modal score."""
    if len(probs) == 1:
        return 1.0

    normalized_probs = _normalize(probs)
    mode_index = max(range(len(normalized_probs)), key=normalized_probs.__getitem__)
    distance_from_mode = sum(
        probability * abs(index - mode_index)
        for index, probability in enumerate(normalized_probs)
    )
    uniform_center = (len(normalized_probs) - 1) / 2
    uniform_mean_absolute_deviation = sum(
        abs(index - uniform_center) for index in range(len(normalized_probs))
    ) / len(normalized_probs)
    return max(0.0, 1.0 - distance_from_mode / uniform_mean_absolute_deviation)


def choice_confidence(probs: list[float]) -> float:
    """Scale peak choice probability from uniform to certainty."""
    if len(probs) == 1:
        return 1.0

    normalized_probs = _normalize(probs)
    uniform_probability = 1.0 / len(normalized_probs)
    return (max(normalized_probs) - uniform_probability) / (
        1.0 - uniform_probability
    )


def _normalize(probs: list[float]) -> list[float]:
    """Normalize confidence inputs, using uniform probabilities for zero totals."""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [probability / total for probability in probs]
