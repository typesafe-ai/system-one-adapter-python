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
    model_request_debug_data: dict[str, list[Any]] = {
        "llm_queries": [],
        "llm_responses": [],
        "debug_info": [],
    }

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
        request_index = len(model_request_debug_data["llm_queries"])
        model_request_debug_data["llm_queries"].append(
            {
                "messages": ModelMessagesTypeAdapter.dump_python(
                    request_context.messages,
                    mode="json",
                    exclude={
                        "__all__": {
                            "conversation_id": True,
                            "parts": {"__all__": {"timestamp"}},
                            "run_id": True,
                            "timestamp": True,
                        }
                    },
                ),
                "model_settings": to_jsonable_python(
                    model_settings,
                    serialize_unknown=True,
                ),
                "model_request_parameters": to_jsonable_python(
                    model_request_parameters,
                    serialize_unknown=True,
                ),
            }
        )
        model_request_debug_data["llm_responses"].append(None)
        model_request_debug_data["debug_info"].append(
            {
                "model_name": request_context.model.model_name,
                "model_id": request_context.model.model_id,
                "provider": request_context.model.system,
            }
        )

        try:
            model_response = await handler(request_context)
        except Exception as error:
            model_request_debug_data["debug_info"][request_index].update(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            )
            raise

        model_request_debug_data["llm_responses"][request_index] = (
            ModelMessagesTypeAdapter.dump_python(
                [model_response],
                mode="json",
                exclude={
                    "__all__": {
                        "conversation_id",
                        "run_id",
                        "timestamp",
                    }
                },
            )[0]
        )
        model_request_debug_data["debug_info"][request_index].update(
            {
                "finish_reason": model_response.finish_reason,
                "response_model_name": model_response.model_name,
            }
        )
        return model_response

    return Hooks(model_request=capture_model_request), model_request_debug_data
