"""PydanticAI-backed TypeSafe-compatible client."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Annotated, Any, Literal, Self, cast

import anthropic
import httpx
import openai
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from typesafe_client import RetryConfig, TypeSafeClient
from typesafe_client.api.api_client import (
    TypeSafeApiError,
    TypeSafeAuthError,
    TypeSafeTimeoutError,
    TypeSafeTokensExceededError,
    TypeSafeUnknownError,
)
from typesafe_client.api.models import (
    ChoiceAnswer,
    ChoiceQuestion,
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
    question_to_api_model,
)

type Mode = Literal["probabilities", "discrete"]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Question = NoulQuestion | ScoreQuestion | ChoiceQuestion

_SYSTEM_PROMPT = """Evaluate every question using only the supplied document.
Return every requested answer. Probability objects are complete probability
distributions: every value is between 0 and 1 and the values sum to 1.
Do not add facts which are not supported by the document."""

_AUTH_ERRORS = (openai.AuthenticationError, anthropic.AuthenticationError)
_CONNECTION_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    httpx.RequestError,
    TimeoutError,
)
_TOKEN_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window",
    "maximum context",
    "prompt is too long",
    "request too large",
    "too many tokens",
)


def _distribution_validator():
    @model_validator(mode="after")
    def validate_distribution(model: BaseModel) -> BaseModel:
        values = model.model_dump().values()
        if not values or abs(sum(values) - 1.0) > 0.02:
            raise ValueError("probabilities must sum to 1")
        return model

    return validate_distribution


def _probability_model(
    name: str,
    labels: list[str],
    descriptions: list[str],
) -> type[BaseModel]:
    fields = {
        f"value_{index}": (
            Probability,
            Field(alias=label, description=description),
        )
        for index, (label, description) in enumerate(
            zip(labels, descriptions, strict=True)
        )
    }
    return create_model(
        name,
        __config__=ConfigDict(extra="forbid"),
        __validators__={"validate_distribution": _distribution_validator()},
        **fields,
    )


def _literal(values: list[str]) -> Any:
    return Literal.__getitem__(tuple(values))


class OpenTypeSafeClient(TypeSafeClient):
    """Evaluate TypeSafe questions through any PydanticAI model.

    :param structured_outputs: Use the provider's native structured-output mode.
    :param noul_mode: Request Noul probabilities or discrete booleans.
    :param score_mode: Request score probabilities or discrete score levels.
    :param choice_mode: Request choice probabilities or discrete choices.
    :param n_retry_malformed_structure: Corrective retries for malformed model output.
    :param retry: Retry policy for transient provider failures.
    """

    def __init__(
        self,
        structured_outputs: bool = False,
        noul_mode: Mode = "probabilities",
        score_mode: Mode = "probabilities",
        choice_mode: Mode = "probabilities",
        n_retry_malformed_structure: int = 0,
        retry: RetryConfig = NoRetries(),  # noqa: B008 - reference-compatible signature
    ) -> None:
        modes = {
            "noul_mode": noul_mode,
            "score_mode": score_mode,
            "choice_mode": choice_mode,
        }
        for name, mode in modes.items():
            if mode not in ("probabilities", "discrete"):
                raise ValueError(f"{name} must be 'probabilities' or 'discrete'")
        if n_retry_malformed_structure < 0:
            raise ValueError("n_retry_malformed_structure must be >= 0")

        self.structured_outputs = structured_outputs
        self.noul_mode = noul_mode
        self.score_mode = score_mode
        self.choice_mode = choice_mode
        self.n_retry_malformed_structure = n_retry_malformed_structure
        self.retry = retry

    def system_one(
        self,
        model: str | Model,
        document: InstructionValue,
        questions: QuestionCollectionType,
    ) -> SystemOneResponse:
        """Synchronously evaluate ``questions`` against one ``document``."""
        normalized_questions = self._normalize_questions(questions)
        output_type = self._output_type(normalized_questions)
        agent = self._agent(model, output_type)
        started_at = time.perf_counter()

        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                result = agent.run_sync(self._prompt(document))
                latency = time.perf_counter() - started_at
                return self._response(
                    self._model_name(model),
                    normalized_questions,
                    cast(BaseModel, result.output),
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    attempt - 1,
                    max(0, result.usage.requests - 1),
                    latency,
                )
            except Exception as error:
                mapped_error = self._map_error(error)
                if attempt >= self.retry.max_attempts or not self._retryable(
                    error, mapped_error
                ):
                    raise mapped_error from error
                delay = self.retry.next_delay(attempt=attempt)
                if delay:
                    time.sleep(delay)

        raise AssertionError("retry loop did not return or raise")

    async def system_one_async(
        self,
        model: str | Model,
        document: InstructionValue,
        questions: QuestionCollectionType,
    ) -> SystemOneResponse:
        """Asynchronously evaluate ``questions`` against one ``document``."""
        normalized_questions = self._normalize_questions(questions)
        output_type = self._output_type(normalized_questions)
        agent = self._agent(model, output_type)
        started_at = time.perf_counter()

        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                result = await agent.run(self._prompt(document))
                latency = time.perf_counter() - started_at
                return self._response(
                    self._model_name(model),
                    normalized_questions,
                    cast(BaseModel, result.output),
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    attempt - 1,
                    max(0, result.usage.requests - 1),
                    latency,
                )
            except Exception as error:
                mapped_error = self._map_error(error)
                if attempt >= self.retry.max_attempts or not self._retryable(
                    error, mapped_error
                ):
                    raise mapped_error from error
                delay = self.retry.next_delay(attempt=attempt)
                if delay:
                    await asyncio.sleep(delay)

        raise AssertionError("retry loop did not return or raise")

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

    @staticmethod
    def _normalize_questions(questions: QuestionCollectionType) -> dict[str, Question]:
        if not questions:
            raise ValueError("At least one question is required.")
        normalized_questions = {
            key: question_to_api_model(question) for key, question in questions.items()
        }
        for question in normalized_questions.values():
            if (
                isinstance(question, (ScoreQuestion, ChoiceQuestion))
                and not question.criteria
            ):
                raise ValueError(
                    "Score and choice questions require at least one criterion."
                )
        return normalized_questions

    def _output_type(self, questions: Mapping[str, Question]) -> type[BaseModel]:
        fields: dict[str, tuple[Any, Field]] = {}
        for index, (question_id, question) in enumerate(questions.items()):
            answer_type = self._question_output_type(index, question)
            fields[f"answer_{index}"] = (
                answer_type,
                Field(
                    alias=question_id,
                    description=self._question_description(question),
                ),
            )

        answers_model = create_model(
            "TypeSafeAnswers",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return create_model(
            "TypeSafeEvaluation",
            __config__=ConfigDict(extra="forbid"),
            answers=(answers_model, Field(description="Answers keyed by question ID.")),
        )

    def _question_output_type(self, index: int, question: Question) -> Any:
        if isinstance(question, NoulQuestion):
            return bool if self.noul_mode == "discrete" else Probability

        if isinstance(question, ScoreQuestion):
            if self.score_mode == "discrete":
                return Annotated[int, Field(ge=0, lt=len(question.criteria))]
            labels = [str(score) for score in range(len(question.criteria))]
            descriptions = [self._instructions(value) for value in question.criteria]
            return _probability_model(
                f"ScoreProbabilities{index}", labels, descriptions
            )

        labels = list(question.criteria)
        if self.choice_mode == "discrete":
            return _literal(labels)
        descriptions = [
            self._instructions(value) for value in question.criteria.values()
        ]
        return _probability_model(f"ChoiceProbabilities{index}", labels, descriptions)

    def _agent(self, model: str | Model, output_type: type[BaseModel]) -> Agent:
        pydantic_model = self._pydantic_model(model)
        requested_output: Any = (
            NativeOutput(output_type)
            if self.structured_outputs
            else PromptedOutput(output_type)
        )
        return Agent(
            pydantic_model,
            output_type=requested_output,
            instructions=_SYSTEM_PROMPT,
            retries={"output": self.n_retry_malformed_structure},
        )

    @staticmethod
    def _pydantic_model(model: str | Model) -> str | Model:
        if not isinstance(model, str) or ":" in model:
            return model
        if model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
            return f"openai:{model}"
        if model.startswith("claude-"):
            return f"anthropic:{model}"
        return model

    @staticmethod
    def _model_name(model: str | Model) -> str:
        return model if isinstance(model, str) else model.model_name

    @staticmethod
    def _prompt(document: InstructionValue) -> str:
        return "Document:\n" + json.dumps(document, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _instructions(value: Any) -> str:
        if value is None:
            return "No additional instructions."
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _question_description(cls, question: Question) -> str:
        description = cls._instructions(question.instructions)
        if not isinstance(question, NoulQuestion) or question.criteria is None:
            return description

        true_criteria = cls._instructions(question.criteria.true)
        false_criteria = cls._instructions(question.criteria.false)
        return (
            f"{description}\nTrue criteria: {true_criteria}\n"
            f"False criteria: {false_criteria}"
        )

    def _response(
        self,
        model: str,
        questions: Mapping[str, Question],
        output: BaseModel,
        input_tokens: int,
        output_tokens: int,
        n_retries: int,
        n_retries_malformed_structure: int,
        latency: float,
    ) -> SystemOneResponse:
        raw_answers = output.model_dump(mode="python", by_alias=True)["answers"]
        answers = {
            question_id: self._answer(question, raw_answers[question_id])
            for question_id, question in questions.items()
        }
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            n_retries=n_retries,
            n_retries_malformed_structure=n_retries_malformed_structure,
            latency=latency,
        )
        return SystemOneResponse(model=model, answers=answers, usage=usage)

    def _answer(
        self, question: Question, value: Any
    ) -> NoulAnswer | ScoreAnswer | ChoiceAnswer:
        if isinstance(question, NoulQuestion):
            probability = (
                float(value)
                if self.noul_mode == "probabilities"
                else float(bool(value))
            )
            return NoulAnswer(type=QuestionType.Noul, noul=probability)

        if isinstance(question, ScoreQuestion):
            labels = [str(score) for score in range(len(question.criteria))]
            probabilities = self._probabilities(labels, value, self.score_mode)
            score = sum(
                index * probabilities[str(index)] for index in range(len(labels))
            )
            confidence = self._confidence(probabilities)
            return ScoreAnswer(
                type=QuestionType.Score,
                score=score,
                confidence=confidence,
                probabilities=probabilities,
            )

        labels = list(question.criteria)
        probabilities = self._probabilities(labels, value, self.choice_mode)
        choice = max(labels, key=probabilities.__getitem__)
        return ChoiceAnswer(
            type=QuestionType.Choice,
            choice=choice,
            confidence=self._confidence(probabilities),
            probabilities=probabilities,
        )

    @staticmethod
    def _confidence(probabilities: Mapping[str, float]) -> float:
        """Scale peak probability from the uniform baseline to certainty."""
        if len(probabilities) == 1:
            return 1.0

        uniform_probability = 1.0 / len(probabilities)
        peak_probability = max(probabilities.values())
        scaled_confidence = (peak_probability - uniform_probability) / (
            1.0 - uniform_probability
        )
        return min(1.0, max(0.0, scaled_confidence))

    @staticmethod
    def _probabilities(labels: list[str], value: Any, mode: Mode) -> dict[str, float]:
        if mode == "discrete":
            selected = str(value)
            return {label: float(label == selected) for label in labels}

        probabilities = {label: float(value[label]) for label in labels}
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("probabilities must have a positive sum")
        return {
            label: probability / total for label, probability in probabilities.items()
        }

    @classmethod
    def _map_error(cls, error: Exception) -> TypeSafeApiError:
        if isinstance(error, TypeSafeApiError):
            return error

        chain = cls._error_chain(error)
        detail = {"message": str(error)}
        if any(isinstance(item, _AUTH_ERRORS) for item in chain):
            return TypeSafeAuthError(detail)

        status_code = next(
            (
                item.status_code
                for item in chain
                if isinstance(item, ModelHTTPError) or hasattr(item, "status_code")
            ),
            None,
        )
        if status_code in (401, 403):
            return TypeSafeAuthError(detail)
        if any(
            isinstance(item, _CONNECTION_ERRORS) for item in chain
        ) or status_code in (408, 504):
            return TypeSafeTimeoutError(detail)

        error_text = " ".join(str(item).lower() for item in chain)
        if any(marker in error_text for marker in _TOKEN_ERROR_MARKERS):
            return TypeSafeTokensExceededError(detail)
        return TypeSafeUnknownError(detail, status_code)

    @classmethod
    def _retryable(cls, error: Exception, mapped_error: TypeSafeApiError) -> bool:
        chain = cls._error_chain(error)
        if any(isinstance(item, _CONNECTION_ERRORS) for item in chain):
            return True
        if any(
            isinstance(item, ModelHTTPError)
            and item.status_code in TypeSafeUnknownError.retryable_status_codes
            for item in chain
        ):
            return True
        return (
            isinstance(mapped_error, TypeSafeUnknownError)
            and mapped_error.is_retryable()
        )

    @staticmethod
    def _error_chain(error: Exception) -> list[BaseException]:
        chain: list[BaseException] = []
        current: BaseException | None = error
        while current is not None and current not in chain:
            chain.append(current)
            current = current.__cause__ or current.__context__
        return chain
