"""Shared pytest configuration for cassette-backed provider tests."""

import copy
import json
import os
from typing import Any

import pytest

# Provider SDKs demand a key when the client is built, before any HTTP happens, so
# replay would fail without one. setdefault runs at import (before parametrize builds
# its clients) and never overwrites a real key, so recording is unaffected.
for _credential in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY"):
    os.environ.setdefault(_credential, "cassette-only")

# Credentials must never reach a committed cassette. Recording strips these outright
# rather than masking them, so a cassette cannot leak a key even if one is set.
FILTERED_HEADERS = [
    "authorization",
    "x-api-key",
    "api-key",
    "openai-organization",
    "openai-project",
    "cookie",
    "set-cookie",
]


# filter_headers only scrubs requests, so responses get an allowlist instead. Providers
# return organization and workspace IDs, request IDs, cookies, and per-account
# rate-limit quotas, none of which belong in a committed file. Content-length and
# content-encoding are dropped deliberately: bodies are stored decoded, so a recorded
# value would lie.
ALLOWED_RESPONSE_HEADERS = {"content-type"}


class ReadableJsonSerializer:
    """Store JSON HTTP bodies as readable objects while preserving VCR replay."""

    @staticmethod
    def _decode_json_container_if_possible(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            decoded_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        return decoded_value if isinstance(decoded_value, (dict, list)) else value

    @staticmethod
    def _encode_json_container_if_present(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    @classmethod
    def serialize(cls, cassette_dict):
        """Serialize cassette data with decoded JSON request and response bodies."""
        readable_cassette_dict = copy.deepcopy(cassette_dict)
        # Decode bodies only in the persisted representation.
        for interaction in readable_cassette_dict["interactions"]:
            request = interaction["request"]
            request["body"] = cls._decode_json_container_if_possible(
                request.get("body")
            )
            response_body = interaction["response"].get("body")
            if isinstance(response_body, dict):
                response_body["string"] = cls._decode_json_container_if_possible(
                    response_body.get("string")
                )
        return json.dumps(readable_cassette_dict, ensure_ascii=False, indent=4) + "\n"

    @classmethod
    def deserialize(cls, cassette_string):
        """Restore readable JSON bodies to the strings VCR expects."""
        cassette_dict = json.loads(cassette_string)
        # Re-encode persisted bodies for VCR's HTTP request and response objects.
        for interaction in cassette_dict["interactions"]:
            request = interaction["request"]
            request["body"] = cls._encode_json_container_if_present(request.get("body"))
            response_body = interaction["response"].get("body")
            if isinstance(response_body, dict):
                response_body["string"] = cls._encode_json_container_if_present(
                    response_body.get("string")
                )
        return cassette_dict


def pytest_recording_configure(config, vcr):
    """Replace VCR's JSON serializer with the readable-body variant."""
    # Override VCR's string-preserving JSON serializer for cassette persistence.
    vcr.register_serializer("json", ReadableJsonSerializer())


def scrub_response(response):
    """Drop every response header outside the allowlist.

    :param response: Recorded response dictionary.
    :return: The response with its headers reduced.
    """
    headers = response.get("headers") or {}
    response["headers"] = {
        name: value
        for name, value in headers.items()
        if name.lower() in ALLOWED_RESPONSE_HEADERS
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """Configure cassette recording and replay.

    Matching includes ``body`` because every provider call in a cassette posts to the
    same endpoint. On the default matchers each request would replay whichever
    interaction was recorded first, silently returning another test's response.

    :return: VCR configuration passed to ``VCR.use_cassettes``.
    """
    return {
        # The registered JSON serializer keeps HTTP bodies as nested objects, so the
        # cassette diffs like its payloads instead of escaped string literals.
        "serializer": "json",
        "filter_headers": FILTERED_HEADERS,
        "filter_query_parameters": ["api_key", "key"],
        "before_record_response": scrub_response,
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        # record_mode is deliberately unset: pytest-recording defaults it to "none"
        # (replay only) and setting it here would override the --record-mode flag.
        "decode_compressed_response": True,
    }
