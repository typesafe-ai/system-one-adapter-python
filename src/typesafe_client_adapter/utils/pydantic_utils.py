"""Pydantic and PydanticAI model construction."""

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.models import Model
from typesafe_client.api.models import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)
from typesafe_client.values import QuestionCollectionType, question_to_api_model

from typesafe_client_adapter.utils.model_request_debug import (
    create_model_request_debug_hooks,
)
from typesafe_client_adapter.utils.probability_normalization import AnswerMode

Probability: TypeAlias = Annotated[float, Field(ge=0, le=1)]


def convert_and_validate_questions(
    questions: QuestionCollectionType,
) -> dict[str, Question]:
    """Convert and validate TypeSafe questions.

    :param questions: TypeSafe question collection.
    :return: Questions using API model types.
    """
    if not questions:
        raise ValueError("At least one question is required.")
    prepared_questions = {}
    for key, question in questions.items():
        prepared_question = question_to_api_model(question)
        if (
            isinstance(prepared_question, (ScoreQuestion, ChoiceQuestion))
            and not prepared_question.criteria
        ):
            raise ValueError(
                "Score and choice questions require at least one criterion."
            )
        prepared_questions[key] = prepared_question
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
        answer_type = _create_llm_answer_type_for_question(
            index,
            question,
            llm_answer_mode,
        )
        fields[f"answer_{index}"] = (
            answer_type,
            Field(
                alias=question_id,
                description=_build_llm_output_field_description(
                    question,
                    llm_answer_mode,
                ),
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
) -> tuple[Agent, dict[str, list[Any]]]:
    """Create the configured PydanticAI agent.

    :param model: PydanticAI model or model name.
    :param output_model: Pydantic output model.
    :param structured_outputs: Whether to use native structured output.
    :param n_retry_malformed_structure: Output validation retry count.
    :param instructions: Agent system instructions.
    :return: Configured agent and its model-request debug capture.
    """
    requested_output: Any = (
        NativeOutput(output_model)
        if structured_outputs
        else PromptedOutput(output_model)
    )
    model_request_debug_hooks, model_request_debug_data = (
        create_model_request_debug_hooks()
    )
    pydantic_agent = Agent(
        _resolve_provider_prefixed_pydantic_ai_model(model),
        output_type=requested_output,
        instructions=instructions,
        retries={"output": n_retry_malformed_structure},
        capabilities=[model_request_debug_hooks],
    )
    return pydantic_agent, model_request_debug_data


def _create_llm_answer_type_for_question(
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
        descriptions = [
            _serialize_instruction_value_for_prompt(value)
            for value in question.criteria
        ]
        return _create_probability_output_model_for_labels(
            f"ScoreProbabilities{index}", labels, descriptions
        )

    labels = list(question.criteria)
    if llm_answer_mode == "discrete":
        return Literal.__getitem__(tuple(labels))
    descriptions = [
        _serialize_instruction_value_for_prompt(value)
        for value in question.criteria.values()
    ]
    return _create_probability_output_model_for_labels(
        f"ChoiceProbabilities{index}", labels, descriptions
    )


def _create_probability_output_model_for_labels(
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


def _build_llm_output_field_description(
    question: Question,
    llm_answer_mode: AnswerMode,
) -> str:
    description = _serialize_instruction_value_for_prompt(question.instructions)
    if llm_answer_mode == "discrete":
        if isinstance(question, ScoreQuestion):
            levels = "\n".join(
                f"{score} = {_serialize_instruction_value_for_prompt(criterion)}"
                for score, criterion in enumerate(question.criteria)
            )
            return f"{description}\nScore levels, answer with the integer:\n{levels}"

        if isinstance(question, ChoiceQuestion):
            choices = "\n".join(
                f"{label} = {_serialize_instruction_value_for_prompt(criterion)}"
                for label, criterion in question.criteria.items()
            )
            return f"{description}\nChoice labels, answer with one label:\n{choices}"

    if not isinstance(question, NoulQuestion) or question.criteria is None:
        return description

    true_criteria = _serialize_instruction_value_for_prompt(question.criteria.true)
    false_criteria = _serialize_instruction_value_for_prompt(question.criteria.false)
    return (
        f"{description}\nTrue criteria: {true_criteria}\n"
        f"False criteria: {false_criteria}"
    )


def _serialize_instruction_value_for_prompt(value: Any) -> str:
    if value is None:
        return "No additional instructions."
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _resolve_provider_prefixed_pydantic_ai_model(model: str | Model) -> str | Model:
    if not isinstance(model, str) or ":" in model:
        return model
    if model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return f"openai:{model}"
    if model.startswith("claude-"):
        return f"anthropic:{model}"
    return model
