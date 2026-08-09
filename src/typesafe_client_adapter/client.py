"""PydanticAI-backed TypeSafe-compatible client."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
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

from typesafe_client_adapter.utils.confidence_metrics import (
    choice_confidence,
    score_confidence,
)
from typesafe_client_adapter.utils.error_handling import (
    run_with_retries,
    run_with_retries_async,
)
from typesafe_client_adapter.utils.probability_normalization import (
    AnswerMode,
    ProbabilityNormalization,
    normalize_probabilities,
    probability_debug_data,
    to_distribution,
)
from typesafe_client_adapter.utils.provider_debug import ProviderDebugModel
from typesafe_client_adapter.utils.pydantic_utils import (
    Question,
    create_llm_output_model,
    create_pydantic_ai_agent,
    get_model_name,
    prepare_questions,
)

Answer = NoulAnswer | ScoreAnswer | ChoiceAnswer

_SYSTEM_PROMPT = """Evaluate every question using only the supplied document.
Return every requested answer. Probability objects are complete probability
distributions: every value is between 0 and 1 and the values sum to 1."""


def _serialize_document_as_user_prompt(document: InstructionValue) -> str:
    return "Document:\n" + json.dumps(document, ensure_ascii=False, sort_keys=True)


def _convert_llm_value_to_typesafe_answer(
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
        # The score is an expected value, so it is only meaningful over a distribution
        # summing to 1. Normalize explicitly here: the reported probabilities are left
        # untouched when ``normalize_probabilities`` is disabled.
        score_distribution = to_distribution(probabilities)
        score = sum(
            index * score_distribution[str(index)] for index in range(len(labels))
        )
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


@dataclass
class _Evaluation:
    """State shared by the synchronous and asynchronous ``system_one`` paths.

    ``run_usage`` is threaded through every PydanticAI attempt so that tokens spent on
    attempts which later failed are still billed to the caller.
    """

    model_name: str
    questions: dict[str, Question]
    pydantic_agent: Agent
    provider_debug_model: ProviderDebugModel
    llm_answer_mode: AnswerMode
    should_normalize_probabilities: bool
    run_usage: RunUsage = field(default_factory=RunUsage)
    started_at: float = field(default_factory=time.perf_counter)
    _requests_before_attempt: int | None = None
    _n_retries_malformed_structure: int = 0

    def begin_attempt(self) -> None:
        """Finish accounting for the prior attempt and start the next one."""
        if self._requests_before_attempt is not None:
            self._n_retries_malformed_structure += (
                self.run_usage.requests - self._requests_before_attempt
            )
        self._requests_before_attempt = self.run_usage.requests

    def response(self, output: BaseModel, n_retries: int) -> SystemOneResponse:
        """Build the response from a successful attempt.

        :param output: Validated LLM output model.
        :param n_retries: Transient-failure retries performed.
        :return: TypeSafe-shaped response.
        """
        latency = time.perf_counter() - self.started_at
        raw_answers = output.model_dump(mode="python", by_alias=True)["answers"]
        answers: dict[str, Answer] = {}
        probability_normalizations: dict[
            str,
            ProbabilityNormalization | None,
        ] = {}
        for question_id, question in self.questions.items():
            answer, probability_normalization = _convert_llm_value_to_typesafe_answer(
                question,
                raw_answers[question_id],
                self.llm_answer_mode,
                self.should_normalize_probabilities,
            )
            answers[question_id] = answer
            probability_normalizations[question_id] = probability_normalization

        assert self._requests_before_attempt is not None
        n_retries_malformed_structure = self._n_retries_malformed_structure + max(
            0,
            self.run_usage.requests - self._requests_before_attempt - 1,
        )
        usage_data: dict[str, Any] = {
            "input_tokens": self.run_usage.input_tokens,
            "output_tokens": self.run_usage.output_tokens,
            "n_retries": n_retries,
            "n_retries_malformed_structure": n_retries_malformed_structure,
            "latency": latency,
        }
        return SystemOneResponse(
            model=self.model_name,
            answers=answers,
            usage=Usage(**usage_data),
            debug={
                **probability_debug_data(probability_normalizations),
                "llm_queries": self.provider_debug_model.llm_queries,
                "llm_responses": self.provider_debug_model.llm_responses,
                "debug_info": self.provider_debug_model.debug_info,
            },
        )


class TypeSafeClientAdapter(TypeSafeClient):
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
        # ``TypeSafeClient.__init__`` is deliberately not called: it requires a TypeSafe
        # API key and builds ``self._api_client``, neither of which this client uses.
        # Every inherited method that touches ``self._api_client`` is overridden
        # below, so a new one must be overridden here too or it raises AttributeError.
        if llm_answer_mode not in ("probabilities", "discrete"):
            raise ValueError("llm_answer_mode must be 'probabilities' or 'discrete'")
        if n_retry_malformed_structure < 0:
            raise ValueError("n_retry_malformed_structure must be >= 0")

        self.structured_outputs = structured_outputs
        self.llm_answer_mode = llm_answer_mode
        self.normalize_probabilities = normalize_probabilities
        self.n_retry_malformed_structure = n_retry_malformed_structure
        self.retry = retry

    def _evaluation(
        self,
        model: str | Model,
        questions: QuestionCollectionType,
    ) -> _Evaluation:
        """Prepare the questions, output model, and agent for one evaluation."""
        prepared_questions = prepare_questions(questions)
        output_model = create_llm_output_model(
            prepared_questions,
            self.llm_answer_mode,
        )
        pydantic_agent, provider_debug_model = create_pydantic_ai_agent(
            model,
            output_model,
            self.structured_outputs,
            self.n_retry_malformed_structure,
            _SYSTEM_PROMPT,
        )
        return _Evaluation(
            model_name=get_model_name(model),
            questions=prepared_questions,
            pydantic_agent=pydantic_agent,
            provider_debug_model=provider_debug_model,
            llm_answer_mode=self.llm_answer_mode,
            should_normalize_probabilities=self.normalize_probabilities,
        )

    def system_one(
        self,
        model: str | Model,
        document: InstructionValue,
        questions: QuestionCollectionType,
    ) -> SystemOneResponse:
        """Synchronously evaluate ``questions`` against one ``document``."""
        evaluation = self._evaluation(model, questions)

        def run_pydantic_agent_attempt() -> Any:
            evaluation.begin_attempt()
            return evaluation.pydantic_agent.run_sync(
                _serialize_document_as_user_prompt(document),
                usage=evaluation.run_usage,
            )

        try:
            result, n_retries = run_with_retries(
                run_pydantic_agent_attempt,
                self.retry,
            )
            return evaluation.response(cast(BaseModel, result.output), n_retries)
        finally:
            evaluation.provider_debug_model.remove_http_hooks()

    async def system_one_async(
        self,
        model: str | Model,
        document: InstructionValue,
        questions: QuestionCollectionType,
    ) -> SystemOneResponse:
        """Asynchronously evaluate ``questions`` against one ``document``."""
        evaluation = self._evaluation(model, questions)

        def run_pydantic_agent_attempt_async() -> Any:
            evaluation.begin_attempt()
            return evaluation.pydantic_agent.run(
                _serialize_document_as_user_prompt(document),
                usage=evaluation.run_usage,
            )

        try:
            result, n_retries = await run_with_retries_async(
                run_pydantic_agent_attempt_async,
                self.retry,
            )
            return evaluation.response(cast(BaseModel, result.output), n_retries)
        finally:
            evaluation.provider_debug_model.remove_http_hooks()

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
