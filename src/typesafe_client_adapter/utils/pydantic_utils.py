"""Pydantic and PydanticAI model construction."""

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.models import Model
from typesafe_client.api.models import ChoiceQuestion, NoulQuestion, ScoreQuestion
from typesafe_client.values import QuestionCollectionType, question_to_api_model

from typesafe_client_adapter.utils.probability_normalization import AnswerMode

Probability: TypeAlias = Annotated[float, Field(ge=0, le=1)]
Question: TypeAlias = NoulQuestion | ScoreQuestion | ChoiceQuestion


def prepare_questions(questions: QuestionCollectionType) -> dict[str, Question]:
    """Convert and validate TypeSafe questions.

    :param questions: TypeSafe question collection.
    :return: Questions using API model types.
    """
    if not questions:
        raise ValueError("At least one question is required.")
    prepared_questions = {
        key: question_to_api_model(question) for key, question in questions.items()
    }
    for question in prepared_questions.values():
        if (
            isinstance(question, (ScoreQuestion, ChoiceQuestion))
            and not question.criteria
        ):
            raise ValueError(
                "Score and choice questions require at least one criterion."
            )
    return prepared_questions


def create_llm_output_model(
    questions: Mapping[str, Question],
    llm_answer_mode: AnswerMode,
) -> type[BaseModel]:
    """Create the complete Pydantic output model.

    :param questions: Prepared questions.
    :param llm_answer_mode: Probability or discrete answer mode.
    :return: Dynamic Pydantic output model.
    """
    fields = {}
    for index, (question_id, question) in enumerate(questions.items()):
        answer_type = _create_question_output_type(
            index,
            question,
            llm_answer_mode,
        )
        fields[f"answer_{index}"] = (
            answer_type,
            Field(
                alias=question_id,
                description=_build_question_description(question, llm_answer_mode),
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
        # "Keyed by question ID" invited models to invent an ID and nest every answer
        # under it. Name the properties as fixed instead.
        answers=(
            answers_model,
            Field(
                description=(
                    "Exactly one answer per property below. Use these property names "
                    "verbatim and do not add, rename, or nest them under any other key."
                )
            ),
        ),
    )


def create_pydantic_ai_agent(
    model: str | Model,
    output_model: type[BaseModel],
    structured_outputs: bool,
    n_retry_malformed_structure: int,
    instructions: str,
) -> Agent:
    """Create the configured PydanticAI agent.

    :param model: PydanticAI model or model name.
    :param output_model: Pydantic output model.
    :param structured_outputs: Whether to use native structured output.
    :param n_retry_malformed_structure: Output validation retry count.
    :param instructions: Agent system instructions.
    :return: Configured agent.
    """
    requested_output: Any = (
        NativeOutput(output_model)
        if structured_outputs
        else PromptedOutput(output_model)
    )
    return Agent(
        _resolve_pydantic_ai_model(model),
        output_type=requested_output,
        instructions=instructions,
        retries={"output": n_retry_malformed_structure},
    )


def get_model_name(model: str | Model) -> str:
    """Return a string model name."""
    return model if isinstance(model, str) else model.model_name


def _create_question_output_type(
    index: int,
    question: Question,
    llm_answer_mode: AnswerMode,
) -> Any:
    if isinstance(question, NoulQuestion):
        return bool if llm_answer_mode == "discrete" else Probability

    if isinstance(question, ScoreQuestion):
        if llm_answer_mode == "discrete":
            return Annotated[int, Field(ge=0, lt=len(question.criteria))]
        labels = [str(score) for score in range(len(question.criteria))]
        descriptions = [_serialize_instructions(value) for value in question.criteria]
        return _create_probability_model(
            f"ScoreProbabilities{index}", labels, descriptions
        )

    labels = list(question.criteria)
    if llm_answer_mode == "discrete":
        return Literal.__getitem__(tuple(labels))
    descriptions = [
        _serialize_instructions(value) for value in question.criteria.values()
    ]
    return _create_probability_model(
        f"ChoiceProbabilities{index}", labels, descriptions
    )


def _create_probability_model(
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
        **fields,
    )


def _build_question_description(
    question: Question,
    llm_answer_mode: AnswerMode,
) -> str:
    description = _serialize_instructions(question.instructions)
    if llm_answer_mode == "discrete":
        if isinstance(question, ScoreQuestion):
            levels = "\n".join(
                f"{score} = {_serialize_instructions(criterion)}"
                for score, criterion in enumerate(question.criteria)
            )
            return f"{description}\nScore levels, answer with the integer:\n{levels}"

        if isinstance(question, ChoiceQuestion):
            choices = "\n".join(
                f"{label} = {_serialize_instructions(criterion)}"
                for label, criterion in question.criteria.items()
            )
            return f"{description}\nChoice labels, answer with one label:\n{choices}"

    if not isinstance(question, NoulQuestion) or question.criteria is None:
        return description

    true_criteria = _serialize_instructions(question.criteria.true)
    false_criteria = _serialize_instructions(question.criteria.false)
    return (
        f"{description}\nTrue criteria: {true_criteria}\n"
        f"False criteria: {false_criteria}"
    )


def _serialize_instructions(value: Any) -> str:
    if value is None:
        return "No additional instructions."
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _resolve_pydantic_ai_model(model: str | Model) -> str | Model:
    if not isinstance(model, str) or ":" in model:
        return model
    if model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return f"openai:{model}"
    if model.startswith("claude-"):
        return f"anthropic:{model}"
    return model
