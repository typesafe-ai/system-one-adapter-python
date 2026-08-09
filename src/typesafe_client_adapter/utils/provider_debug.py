"""Capture reproducible PydanticAI provider calls."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_core import to_jsonable_python


class _DebugCapturingModel(WrapperModel):
    """Model wrapper recording every provider request and response.

    The wrapper keeps capture aligned with every PydanticAI attempt and retry. It
    records final HTTP bodies when available, falls back to PydanticAI request data
    for non-HTTP models, and captures errors without duplicating sync and async paths.

    :param wrapped: PydanticAI model or known model name.
    """

    def __init__(self, wrapped: str | Model) -> None:
        super().__init__(wrapped)
        self.llm_queries: list[dict[str, Any]] = []
        self.llm_responses: list[dict[str, Any] | None] = []
        self.debug_info: list[dict[str, Any]] = []
        self._request_indexes: dict[int, int] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._captures_http_requests = self._install_http_hooks()

    @staticmethod
    def _json_body(content: bytes) -> dict[str, Any]:
        """Decode an HTTP body into a JSON-compatible dictionary.

        :param content: Raw HTTP body bytes.
        :return: Parsed dictionary or a dictionary wrapping non-object content.
        """
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"content": content.decode(errors="replace")}
        if isinstance(value, dict):
            return value
        return {"content": value}

    def _install_http_hooks(self) -> bool:
        """Install capture hooks on a provider SDK's HTTPX client.

        :return: Whether an HTTPX client was found.
        """
        provider = self.wrapped.provider
        provider_client = provider.client if provider is not None else None
        http_client = getattr(provider_client, "_client", None)
        if not isinstance(http_client, httpx.AsyncClient):
            return False
        self._http_client = http_client
        http_client.event_hooks["request"].append(self._capture_http_request)
        http_client.event_hooks["response"].append(self._capture_http_response)
        return True

    def remove_http_hooks(self) -> None:
        """Remove capture hooks from a caller-owned provider client."""
        if self._http_client is None:
            return
        self._http_client.event_hooks["request"].remove(self._capture_http_request)
        self._http_client.event_hooks["response"].remove(self._capture_http_response)
        self._http_client = None

    async def _capture_http_request(self, request: httpx.Request) -> None:
        """Capture one final provider HTTP request body.

        :param request: Prepared HTTPX request.
        """
        content = await request.aread()
        request_index = len(self.llm_queries)
        self._request_indexes[id(request)] = request_index
        self.llm_queries.append(self._json_body(content))
        self.llm_responses.append(None)
        self.debug_info.append(
            {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "provider": self.system,
                "method": request.method,
                "url": str(request.url.copy_with(query=None)),
            }
        )

    async def _capture_http_response(self, response: httpx.Response) -> None:
        """Capture the raw response matching one provider request.

        :param response: Provider HTTP response.
        """
        request_index = self._request_indexes.get(id(response.request))
        if request_index is None:
            return
        content = await response.aread()
        self.llm_responses[request_index] = self._json_body(content)
        self.debug_info[request_index]["status_code"] = response.status_code

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
        if not self._captures_http_requests:
            prepared_settings, prepared_parameters = self.wrapped.prepare_request(
                model_settings,
                model_request_parameters,
            )
            self.llm_queries.append(
                {
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
                }
            )
            self.llm_responses.append(None)
            self.debug_info.append(
                {
                    "transport": "pydantic_ai",
                    "model_name": self.model_name,
                    "model_id": self.model_id,
                    "provider": self.system,
                }
            )

        try:
            model_response = await self.wrapped.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except Exception as error:
            if self.debug_info:
                self.debug_info[-1].update(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            raise

        if not self._captures_http_requests:
            self.llm_responses[-1] = ModelMessagesTypeAdapter.dump_python(
                [model_response],
                mode="json",
            )[0]
        elif self.debug_info:
            self.debug_info[-1].update(
                {
                    "response_model_name": model_response.model_name,
                    "finish_reason": model_response.finish_reason,
                }
            )
        return model_response
