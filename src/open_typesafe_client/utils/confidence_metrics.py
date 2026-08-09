"""Confidence metrics for probability distributions."""


def score_confidence(probs: list[float]) -> float:
    """Measure score concentration around its modal score."""
    if len(probs) == 1:
        return 1.0

    mode_index = max(range(len(probs)), key=probs.__getitem__)
    distance_from_mode = sum(
        probability * abs(index - mode_index) for index, probability in enumerate(probs)
    )
    uniform_center = (len(probs) - 1) / 2
    uniform_mean_absolute_deviation = sum(
        abs(index - uniform_center) for index in range(len(probs))
    ) / len(probs)
    return max(0.0, 1.0 - distance_from_mode / uniform_mean_absolute_deviation)


def choice_confidence(probs: list[float]) -> float:
    """Scale peak choice probability from uniform to certainty."""
    if len(probs) == 1:
        return 1.0

    uniform_probability = 1.0 / len(probs)
    return (max(probs) - uniform_probability) / (1.0 - uniform_probability)
