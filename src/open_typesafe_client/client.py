"""PydanticAI-backed TypeSafe-compatible client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self, cast

from pydantic import BaseModel
from pydantic_ai.models import Model
from typesafe_client import RetryConfig, TypeSafeClient
from typesafe_client.api.models import (
    ChoiceAnswer,
    NoulAnswer,
    NoulQuestion,
    QuestionType,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResponse,
    Usage,
)
from typesafe_client.api.retry import NoRetries
from typesafe_client.values import (
    InstructionValue,
    QuestionCollectionType,
)

from open_typesafe_client.utils.confidence_metrics import (
    choice_confidence,
    score_confidence,
)
from open_typesafe_client.utils.error_handling import (
    run_with_retries,
    run_with_retries_async,
)
from open_typesafe_client.utils.probability_normalization import (
    AnswerMode,
    ProbabilityNormalization,
    ProbabilityNormalizationStats,
    normalize_probabilities,
)
from open_typesafe_client.utils.pydantic_utils import (
    Question,
    create_llm_output_model,
    create_pydantic_ai_agent,
    get_model_name,
    prepare_questions,
)

Answer = NoulAnswer | ScoreAnswer | ChoiceAnswer

_SYSTEM_PROMPT = """Evaluate every question using only the supplied document.
Return every requested answer. Probability objects are complete probability
distributions: every value is between 0 and 1 and the values sum to 1.
Do not add facts which are not supported by the document."""


def _prompt(document: InstructionValue) -> str:
    return "Document:\n" + json.dumps(document, ensure_ascii=False, sort_keys=True)


def _answer(
    question: Question,
    value: Any,
    llm_answer_mode: AnswerMode,
    should_normalize_probabilities: bool,
) -> tuple[Answer, ProbabilityNormalization | None]:
    if isinstance(question, NoulQuestion):
        probability = (
            float(bool(value)) if llm_answer_mode == "discrete" else float(value)
        )
        return NoulAnswer(type=QuestionType.Noul, noul=probability), None

    if isinstance(question, ScoreQuestion):
        labels = [str(score) for score in range(len(question.criteria))]
        probability_normalization = normalize_probabilities(
            labels,
            value,
            llm_answer_mode,
            should_normalize_probabilities,
        )
        probabilities = probability_normalization.probabilities
        score = sum(index * probabilities[str(index)] for index in range(len(labels)))
        answer = ScoreAnswer(
            type=QuestionType.Score,
            score=score,
            confidence=score_confidence(list(probabilities.values())),
            probabilities=probabilities,
        )
        return answer, probability_normalization

    labels = list(question.criteria)
    probability_normalization = normalize_probabilities(
        labels,
        value,
        llm_answer_mode,
        should_normalize_probabilities,
    )
    probabilities = probability_normalization.probabilities
    choice = max(labels, key=probabilities.__getitem__)
    answer = ChoiceAnswer(
        type=QuestionType.Choice,
        choice=choice,
        confidence=choice_confidence(list(probabilities.values())),
        probabilities=probabilities,
    )
    return answer, probability_normalization


def _response(
    model: str,
    questions: Mapping[str, Question],
    output: BaseModel,
    input_tokens: int,
    output_tokens: int,
    n_retries: int,
    n_retries_malformed_structure: int,
    latency: float,
    llm_answer_mode: AnswerMode,
    should_normalize_probabilities: bool,
) -> SystemOneResponse:
    raw_answers = output.model_dump(mode="python", by_alias=True)["answers"]
    answers: dict[str, Answer] = {}
    probability_normalization_stats = ProbabilityNormalizationStats()
    for question_id, question in questions.items():
        answer, probability_normalization = _answer(
            question,
            raw_answers[question_id],
            llm_answer_mode,
            should_normalize_probabilities,
        )
        answers[question_id] = answer
        probability_normalization_stats.add(question_id, probability_normalization)

    usage_data: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "n_retries": n_retries,
        "n_retries_malformed_structure": n_retries_malformed_structure,
        "latency": latency,
    }
    usage_data.update(probability_normalization_stats.usage_data())
    usage = Usage(**usage_data)
    return SystemOneResponse(model=model, answers=answers, usage=usage)


class OpenTypeSafeClient(TypeSafeClient):
    """Evaluate TypeSafe questions through any PydanticAI model.

    :param structured_outputs: Use the provider's native structured-output mode.
    :param llm_answer_mode: Request probabilities or discrete answers.
    :param normalize_probabilities: Normalize invalid LLM probabilities when true.
    :param n_retry_malformed_structure: Corrective retries for malformed model output.
    :param retry: Retry policy for transient provider failures.
    """

    def __init__(
        self,
        structured_outputs: bool = False,
        llm_answer_mode: AnswerMode = "probabilities",
        normalize_probabilities: bool = False,
        n_retry_malformed_structure: int = 0,
        retry: RetryConfig = NoRetries(),  # noqa: B008 - reference-compatible signature
    ) -> None:
        if llm_answer_mode not in ("probabilities", "discrete"):
            raise ValueError("llm_answer_mode must be 'probabilities' or 'discrete'")
        if n_retry_malformed_structure < 0:
            raise ValueError("n_retry_malformed_structure must be >= 0")

        self.structured_outputs = structured_outputs
        self.llm_answer_mode = llm_answer_mode
        self.normalize_probabilities = normalize_probabilities
        self.n_retry_malformed_structure = n_retry_malformed_structure
        self.retry = retry

    def system_one(
        self,
        model: str | Model,
        document: InstructionValue,
        questions: QuestionCollectionType,
    ) -> SystemOneResponse:
        """Synchronously evaluate ``questions`` against one ``document``."""
        prepared_questions = prepare_questions(questions)
        output_model = create_llm_output_model(
            prepared_questions,
            self.llm_answer_mode,
        )
        pydantic_agent = create_pydantic_ai_agent(
            model,
            output_model,
            self.structured_outputs,
            self.n_retry_malformed_structure,
            _SYSTEM_PROMPT,
        )
        started_at = time.perf_counter()

        result, n_retries = run_with_retries(
            lambda: pydantic_agent.run_sync(_prompt(document)),
            self.retry,
        )
        latency = time.perf_counter() - started_at
        return _response(
            get_model_name(model),
            prepared_questions,
            cast(BaseModel, result.output),
            result.usage.input_tokens,
            result.usage.output_tokens,
            n_retries,
            max(0, result.usage.requests - 1),
            latency,
            self.llm_answer_mode,
            self.normalize_probabilities,
        )

    async def system_one_async(
        self,
        model: str | Model,
        document: InstructionValue,
        questions: QuestionCollectionType,
    ) -> SystemOneResponse:
        """Asynchronously evaluate ``questions`` against one ``document``."""
        prepared_questions = prepare_questions(questions)
        output_model = create_llm_output_model(
            prepared_questions,
            self.llm_answer_mode,
        )
        pydantic_agent = create_pydantic_ai_agent(
            model,
            output_model,
            self.structured_outputs,
            self.n_retry_malformed_structure,
            _SYSTEM_PROMPT,
        )
        started_at = time.perf_counter()

        result, n_retries = await run_with_retries_async(
            lambda: pydantic_agent.run(_prompt(document)),
            self.retry,
        )
        latency = time.perf_counter() - started_at
        return _response(
            get_model_name(model),
            prepared_questions,
            cast(BaseModel, result.output),
            result.usage.input_tokens,
            result.usage.output_tokens,
            n_retries,
            max(0, result.usage.requests - 1),
            latency,
            self.llm_answer_mode,
            self.normalize_probabilities,
        )

    def close(self) -> None:
        """Close the client.

        PydanticAI owns provider connection pools, so no client resource is held here.
        """

    async def aclose(self) -> None:
        """Asynchronously close the client."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
