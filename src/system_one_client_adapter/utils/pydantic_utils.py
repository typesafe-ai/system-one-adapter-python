"""Pydantic and PydanticAI model construction."""

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)
from pydantic_ai.tools import GenerateToolJsonSchema
from typesafe_client.api.models import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)
from typesafe_client.values import QuestionCollectionType, question_to_api_model

from system_one_client_adapter.utils.probability_normalization import AnswerMode

Probability: TypeAlias = Annotated[float, Field(ge=0, le=1)]


def convert_question_collection_to_validated_api_question_models(
    questions: QuestionCollectionType,
) -> dict[str, Question]:
    """Convert a question collection to validated API question models.

    Reject empty collections, convert dictionary questions to API model instances,
    and require score and choice questions to define at least two criteria.

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
            and len(prepared_question.criteria) < 2
        ):
            raise ValueError(
                "Score and choice questions require at least two criteria."
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
        description = _build_llm_output_field_description(question, llm_answer_mode)
        if isinstance(answer_type, type) and issubclass(answer_type, BaseModel):
            # Choice and Score answers are nested models emitted as `$ref`s. Some
            # provider schema transformers, including Anthropic structured outputs
            # in PydanticAI, silently drop keywords beside a `$ref`. A description
            # placed only on the field would not reach the model, which would see
            # option names without the question or criteria. Put that context on the
            # model so it remains inside the referenced definition.
            answer_type.__doc__ = description
        fields[f"answer_{index}"] = (
            answer_type,
            Field(
                alias=question_id,
                description=description,
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


def create_raw_output_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    """Create the raw output schema used by PydanticAI.

    :param output_model: Dynamic model returned by :func:`create_llm_output_model`.
    :return: JSON schema before provider-specific transformations.
    """
    return TypeAdapter(output_model).json_schema(
        schema_generator=GenerateToolJsonSchema,
    )


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
        answers = [str(score) for score in range(len(question.criteria))]
    else:
        answers = list(question.criteria)
        if llm_answer_mode == "discrete":
            return Literal.__getitem__(tuple(answers))

    probability_fields = {
        f"probability_{answer_index}": (Probability, Field(alias=answer))
        for answer_index, answer in enumerate(answers)
    }
    return create_model(
        f"ProbabilityMap{index}",
        __config__=ConfigDict(extra="forbid"),
        **probability_fields,
    )


def _build_llm_output_field_description(
    question: Question,
    llm_answer_mode: AnswerMode,
) -> str:
    description = _serialize_instruction_value_for_prompt(question.instructions)
    if isinstance(question, NoulQuestion) and llm_answer_mode == "probabilities":
        description = (
            "Probability that the answer is yes or the assertion is true. "
            "0 means no or false, 0.5 means uncertain, and 1 means yes or true.\n"
            f"Question: {description}"
        )
    elif isinstance(question, ScoreQuestion) and llm_answer_mode == "probabilities":
        description = (
            "Each property maps a rubric level to the probability that the document "
            f"matches it.\nQuestion: {description}"
        )
    elif isinstance(question, ChoiceQuestion) and llm_answer_mode == "probabilities":
        description = (
            "Each property maps an option to the probability that it is the best "
            "answer.\n"
            f"Question: {description}"
        )

    if isinstance(question, ScoreQuestion):
        levels = "\n".join(
            f"{score} = {_serialize_instruction_value_for_prompt(criterion)}"
            for score, criterion in enumerate(question.criteria)
        )
        if llm_answer_mode == "discrete":
            return f"{description}\nScore levels, answer with the integer:\n{levels}"
        return f"{description}\nRequired probability keys:\n{levels}"

    if isinstance(question, ChoiceQuestion):
        choices = "\n".join(
            f"{answer} = {_serialize_instruction_value_for_prompt(criterion)}"
            for answer, criterion in question.criteria.items()
        )
        if llm_answer_mode == "discrete":
            return f"{description}\nChoice labels, answer with one label:\n{choices}"
        return f"{description}\nRequired probability keys:\n{choices}"

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
