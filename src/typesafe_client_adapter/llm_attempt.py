"""Deserialize captured PydanticAI model requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings


def deserialize_llm_attempt(
    llm_attempt: Mapping[str, Any],
) -> tuple[list[ModelMessage], ModelSettings | None, ModelRequestParameters]:
    """Deserialize one response ``debug.llm_attempts`` entry.

    :param llm_attempt: Serialized model attempt from a TypeSafe response.
    :return: Positional arguments for PydanticAI's ``Model.request`` method.
    """
    return (
        ModelMessagesTypeAdapter.validate_python(llm_attempt["messages"]),
        cast(ModelSettings | None, llm_attempt["model_settings"]),
        TypeAdapter(ModelRequestParameters).validate_python(
            llm_attempt["model_request_parameters"]
        ),
    )
