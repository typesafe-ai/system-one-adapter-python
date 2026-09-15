"""Pydantic and PydanticAI model construction."""

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

import msgspec
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)
from pydantic_ai.tools import GenerateToolJsonSchema
from typesafe_sdk import (
    Choice,
    Noul,
    Questions,
    Score,
)
from typesafe_sdk._schemas.models import Question as WireQuestion

from open_system_one.utils.probability_normalization import AnswerMode

Question = Noul | Choice | Score

Probability: TypeAlias = Annotated[float, Field(ge=0, le=1)]


def convert_question_collection_to_validated_api_question_models(
    questions: Questions,
) -> dict[str, Question]:
    """Convert a question collection to validated API question models.

    Reject empty collections, convert dictionary questions to API model instances,
    require list criteria for scores, and at least two criteria for scores and choices.

    :param questions: TypeSafe question collection.
    :return: Questions using API model types.
    """
    if not questions:
        raise ValueError("At least one question is required.")
    prepared_questions = {}
    for key, question in questions.items():
        # The wire union validates JSON; public question types contain recursive aliases
        # that msgspec cannot decode directly in SDK 0.5.7.
        validated = msgspec.to_builtins(
            msgspec.convert(msgspec.to_builtins(question), type=WireQuestion)
        )
        question_class = {"noul": Noul, "choice": Choice, "score": Score}[
            validated.pop("type")
        ]
        prepared_question = question_class(**validated)
        if (
            isinstance(prepared_question, (Score, Choice))
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
        is_probability_map = llm_answer_mode == "probabilities" and isinstance(
            question, (Choice, Score)
        )
        # Choice and Score probability answers are nested models emitted as `$ref`s.
        # Anthropic's SDK silently discards sibling keywords in native structured
        # output, which previously removed the question and criteria. Keep the
        # reference site bare; the model definition and its concrete properties carry
        # that context instead.
        output_field = (
            Field(alias=question_id)
            if is_probability_map
            else Field(
                alias=question_id,
                description=_build_llm_output_field_description(
                    question,
                    llm_answer_mode,
                ),
            )
        )
        fields[f"answer_{index}"] = (
            answer_type,
            output_field,
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
    if isinstance(question, Noul):
        return bool if llm_answer_mode == "discrete" else Probability

    if isinstance(question, Score):
        if llm_answer_mode == "discrete":
            return Annotated[int, Field(ge=0, lt=len(question.criteria))]
        answers_and_criteria = [
            (str(score), criterion) for score, criterion in enumerate(question.criteria)
        ]
    else:
        answers = list(question.criteria)
        if llm_answer_mode == "discrete":
            return Literal.__getitem__(tuple(answers))
        answers_and_criteria = list(question.criteria.items())

    # Put criteria on their concrete properties so they survive `$ref` transforms.
    probability_fields = {
        f"probability_{answer_index}": (
            Probability,
            Field(
                alias=answer,
                description=_serialize_instruction_value_for_prompt(criterion),
            ),
        )
        for answer_index, (answer, criterion) in enumerate(answers_and_criteria)
    }
    return create_model(
        f"ProbabilityMap{index}",
        __config__=ConfigDict(extra="forbid"),
        __doc__=_build_llm_output_question_description(question, llm_answer_mode),
        **probability_fields,
    )


def _build_llm_output_field_description(
    question: Question,
    llm_answer_mode: AnswerMode,
) -> str:
    description = _build_llm_output_question_description(question, llm_answer_mode)

    if isinstance(question, Score):
        levels = "\n".join(
            f"{score} = {_serialize_instruction_value_for_prompt(criterion)}"
            for score, criterion in enumerate(question.criteria)
        )
        if llm_answer_mode == "discrete":
            return f"{description}\nScore levels, answer with the integer:\n{levels}"
        return f"{description}\nRequired probability keys:\n{levels}"

    if isinstance(question, Choice):
        choices = "\n".join(
            f"{answer} = {_serialize_instruction_value_for_prompt(criterion)}"
            for answer, criterion in question.criteria.items()
        )
        if llm_answer_mode == "discrete":
            return f"{description}\nChoice labels, answer with one label:\n{choices}"
        return f"{description}\nRequired probability keys:\n{choices}"

    if not isinstance(question, Noul) or question.criteria is None:
        return description

    true_criteria = _serialize_instruction_value_for_prompt(
        question.criteria.get("true")
    )
    false_criteria = _serialize_instruction_value_for_prompt(
        question.criteria.get("false")
    )
    return (
        f"{description}\nTrue criteria: {true_criteria}\n"
        f"False criteria: {false_criteria}"
    )


def _build_llm_output_question_description(
    question: Question,
    llm_answer_mode: AnswerMode,
) -> str:
    description = _serialize_instruction_value_for_prompt(question.instructions)
    if isinstance(question, Noul) and llm_answer_mode == "probabilities":
        description = (
            "Probability that the answer is yes or the assertion is true. "
            "0 means no or false, 0.5 means uncertain, and 1 means yes or true.\n"
            f"Question: {description}"
        )
    elif isinstance(question, Score) and llm_answer_mode == "probabilities":
        description = (
            "Each property maps a rubric level to the probability that the document "
            f"matches it.\nQuestion: {description}"
        )
    elif isinstance(question, Choice) and llm_answer_mode == "probabilities":
        description = (
            "Each property maps an option to the probability that it is the best "
            "answer.\n"
            f"Question: {description}"
        )
    return description


def _serialize_instruction_value_for_prompt(value: Any) -> str:
    if value is None:
        return "No additional instructions."
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
