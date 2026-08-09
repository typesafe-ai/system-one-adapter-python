"""Capture reproducible PydanticAI model requests through a public hook."""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter, RunContext
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_core import to_jsonable_python


def create_model_request_debug_hooks() -> tuple[Hooks, dict[str, list[Any]]]:
    """Create model-request hooks and their mutable debug-data store.

    :return: PydanticAI hooks and aligned request, response, and metadata lists.
    """
    llm_queries: list[dict[str, Any]] = []

    async def capture_model_request(
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Record one request around the call to its PydanticAI model.

        :param ctx: Current PydanticAI run context.
        :param request_context: Messages, settings, model, and request parameters.
        :param handler: Function performing the model request.
        :return: Unchanged model response.
        """
        model_settings, model_request_parameters = (
            request_context.model.prepare_request(
                request_context.model_settings,
                request_context.model_request_parameters,
            )
        )
        request_debug_info = {
            "model_name": request_context.model.model_name,
            "model_id": request_context.model.model_id,
            "provider": request_context.model.system,
        }
        llm_query = {
            "messages": ModelMessagesTypeAdapter.dump_python(
                request_context.messages,
                mode="json",
            ),
            "model_settings": to_jsonable_python(
                model_settings,
                serialize_unknown=True,
            ),
            "model_request_parameters": to_jsonable_python(
                model_request_parameters,
                serialize_unknown=True,
            ),
            "llm_response": None,
            "debug_info": request_debug_info,
        }
        llm_queries.append(llm_query)

        try:
            model_response = await handler(request_context)
        except Exception as error:
            request_debug_info.update(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            )
            raise

        llm_query["llm_response"] = ModelMessagesTypeAdapter.dump_python(
            [model_response],
            mode="json",
        )[0]
        request_debug_info["finish_reason"] = model_response.finish_reason
        return model_response

    return Hooks(model_request=capture_model_request), {"llm_queries": llm_queries}
