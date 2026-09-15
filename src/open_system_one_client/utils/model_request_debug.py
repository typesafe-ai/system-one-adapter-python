"""Capture reproducible PydanticAI model requests through a public hook."""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.models import ModelRequestContext

from open_system_one_client.utils.error_handling import RetryReasons


def create_model_request_debug_hooks(
    retry_reasons: list[RetryReasons],
) -> tuple[Hooks, dict[str, list[Any]]]:
    """Create model-request hooks and their mutable debug-data store.

    :param retry_reasons: Mutable retry-reason collector.
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
        retry_prompt_messages = [
            message_part.model_response()
            for message_part in request_context.messages[-1].parts
            if isinstance(message_part, RetryPromptPart)
        ]
        if retry_prompt_messages:
            retry_reasons.append(
                RetryReasons(
                    category="malformed_structure",
                    msg="\n\n".join(retry_prompt_messages),
                )
            )
        request_debug_info = {
            "model_name": request_context.model.model_name,
            "model_id": request_context.model.model_id,
            "provider": request_context.model.system,
        }
        llm_query = {
            "messages": request_context.messages,
            "model_settings": request_context.model_settings,
            "model_request_parameters": request_context.model_request_parameters,
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

        llm_query["llm_response"] = model_response
        request_debug_info["finish_reason"] = model_response.finish_reason
        return model_response

    return Hooks(model_request=capture_model_request), {"llm_attempts": llm_queries}
