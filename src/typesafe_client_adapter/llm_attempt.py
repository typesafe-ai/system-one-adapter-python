"""Deserialize captured PydanticAI model requests."""

from __future__ import annotations

from collections.abc import Mapping
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


def deserialize_llm_attempt(
    llm_attempt: Mapping[str, Any],
    model: Model | str | None = None,
) -> ModelRequestContext:
    """Deserialize one response ``debug.llm_attempts`` entry.

    :param llm_attempt: Serialized model attempt from a TypeSafe response.
    :param model: Optional configured model or model identifier. Defaults to the
        recorded model identifier.
    :return: PydanticAI context containing the model and direct request arguments.
    """
    return ModelRequestContext(
        model=infer_model(model or str(llm_attempt["debug_info"]["model_id"])),
        messages=ModelMessagesTypeAdapter.validate_python(llm_attempt["messages"]),
        model_settings=cast(ModelSettings | None, llm_attempt["model_settings"]),
        model_request_parameters=(
            TypeAdapter(ModelRequestParameters).validate_python(
                llm_attempt["model_request_parameters"]
            )
        ),
    )
