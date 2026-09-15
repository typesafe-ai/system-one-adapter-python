"""SDK response types extended with LLM accounting and diagnostics."""

import json
from typing import Any, Literal

import msgspec
from pydantic import TypeAdapter
from typesafe_sdk import SystemOneResponse as SDKSystemOneResponse
from typesafe_sdk import Usage as SDKUsage


class Usage(SDKUsage, kw_only=True):
    """Keep final-attempt usage alongside cumulative retry accounting."""

    input_tokens_total: int
    output_tokens_total: int
    n_retries: int
    n_retries_malformed_structure: int
    latency: float


class SystemOneResponse(SDKSystemOneResponse, frozen=True, kw_only=True):
    """SDK answers and typed views with native PydanticAI debug objects."""

    usage: Usage
    debug: dict[str, Any]

    def model_dump(
        self, *, mode: Literal["python", "json"] = "python"
    ) -> dict[str, Any]:
        """Serialize public fields, preserving PydanticAI's JSON encoding."""
        data = {
            "model": self.model,
            "answers": msgspec.to_builtins(self.answers, str_keys=mode == "json"),
            "usage": msgspec.to_builtins(self.usage),
            "debug": self.debug,
        }
        return TypeAdapter(dict[str, Any]).dump_python(data, mode=mode)

    def model_dump_json(self, *, indent: int | None = None) -> str:
        """Serialize the extended response as JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)
