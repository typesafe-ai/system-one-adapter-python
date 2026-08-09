"""Deserialize captured PydanticAI model requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    infer_model,
)
from pydantic_ai.settings import ModelSettings

_MODEL_REQUEST_PARAMETERS_ADAPTER = TypeAdapter(ModelRequestParameters)


def deserialize_llm_attempt(
    llm_attempt: Mapping[str, Any],
    model: Model | str | None = None,
) -> ModelRequestContext:
    """Deserialize one response ``debug.llm_attempts`` entry.

    The recorded parameters are provider-prepared. PydanticAI prepares direct
    ``Model.request`` calls again, so the generated structured-output instruction is
    removed first and then reconstructed exactly once.

    :param llm_attempt: Serialized model attempt from a TypeSafe response.
    :param model: Optional configured model or model identifier. Defaults to the
        recorded model identifier.
    :return: PydanticAI context containing the model and direct request arguments.
    """
    model_settings_data = llm_attempt["model_settings"]
    model_settings = (
        None
        if model_settings_data is None
        else cast(ModelSettings, dict(model_settings_data))
    )
    model_request_parameters = _MODEL_REQUEST_PARAMETERS_ADAPTER.validate_python(
        llm_attempt["model_request_parameters"]
    )
    instruction_parts = list(model_request_parameters.instruction_parts or [])
    prompted_output_instructions = (
        model_request_parameters.prompted_output_instructions
    )
    if (
        prompted_output_instructions
        and instruction_parts
        and instruction_parts[-1].content == prompted_output_instructions
    ):
        model_request_parameters = replace(
            model_request_parameters,
            instruction_parts=instruction_parts[:-1],
        )

    pydantic_model = infer_model(
        model or str(llm_attempt["debug_info"]["model_id"])
    )
    return ModelRequestContext(
        model=pydantic_model,
        messages=ModelMessagesTypeAdapter.validate_python(llm_attempt["messages"]),
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
    )
