"""Deserialize and replay captured PydanticAI model requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self, cast

from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings

_MODEL_REQUEST_PARAMETERS_ADAPTER = TypeAdapter(ModelRequestParameters)


@dataclass(frozen=True)
class PydanticAIRequest:
    """A replayable PydanticAI request restored from an ``llm_attempt``.

    :param model_id: Recorded PydanticAI model identifier.
    :param messages: Deserialized PydanticAI message history.
    :param model_settings: Recorded model settings.
    :param model_request_parameters: Recorded provider-prepared request parameters.
    """

    model_id: str
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters

    @classmethod
    def from_llm_attempt(cls, llm_attempt: Mapping[str, Any]) -> Self:
        """Deserialize one response ``debug.llm_attempts`` entry.

        :param llm_attempt: Serialized model attempt from a TypeSafe response.
        :return: Replayable PydanticAI request.
        """
        model_settings_data = llm_attempt["model_settings"]
        model_settings = (
            None
            if model_settings_data is None
            else cast(ModelSettings, dict(model_settings_data))
        )
        return cls(
            model_id=str(llm_attempt["debug_info"]["model_id"]),
            messages=ModelMessagesTypeAdapter.validate_python(
                llm_attempt["messages"]
            ),
            model_settings=model_settings,
            model_request_parameters=(
                _MODEL_REQUEST_PARAMETERS_ADAPTER.validate_python(
                    llm_attempt["model_request_parameters"]
                )
            ),
        )

    async def replay(self, model: Model | str | None = None) -> ModelResponse:
        """Repeat the captured model request.

        The recorded parameters are provider-prepared. PydanticAI prepares direct
        ``Model.request`` calls again, so the generated structured-output instruction
        is removed first and then reconstructed exactly once.

        :param model: Optional configured model or model identifier. Defaults to the
            recorded model identifier.
        :return: New PydanticAI model response.
        """
        model_request_parameters = self.model_request_parameters
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

        pydantic_model = infer_model(model or self.model_id)
        return await pydantic_model.request(
            self.messages,
            self.model_settings,
            model_request_parameters,
        )
