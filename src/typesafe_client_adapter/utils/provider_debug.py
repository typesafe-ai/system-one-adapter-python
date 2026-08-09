"""Capture reproducible PydanticAI provider calls."""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_core import to_jsonable_python


class ProviderDebugModel(WrapperModel):
    """Model wrapper recording every provider request and response.

    :param wrapped: PydanticAI model or known model name.
    """

    def __init__(self, wrapped: str | Model) -> None:
        super().__init__(wrapped)
        self.llm_queries: list[dict[str, Any]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Record one request and its response.

        :param messages: Provider-bound message history.
        :param model_settings: Model generation settings.
        :param model_request_parameters: Tools and structured-output configuration.
        :return: Wrapped model response.
        """
        prepared_settings, prepared_parameters = self.wrapped.prepare_request(
            model_settings,
            model_request_parameters,
        )
        query_entry: dict[str, Any] = {
            "messages": ModelMessagesTypeAdapter.dump_python(
                messages,
                mode="json",
            ),
            "model_settings": to_jsonable_python(
                prepared_settings,
                serialize_unknown=True,
            ),
            "model_request_parameters": to_jsonable_python(
                prepared_parameters,
                serialize_unknown=True,
            ),
            "llm_response": None,
            "debug_info": {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "provider": self.system,
                "base_url": self.base_url,
            },
        }
        self.llm_queries.append(query_entry)

        try:
            model_response = await self.wrapped.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except Exception as error:
            query_entry["debug_info"].update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise

        query_entry["llm_response"] = ModelMessagesTypeAdapter.dump_python(
            [model_response],
            mode="json",
        )[0]
        return model_response
