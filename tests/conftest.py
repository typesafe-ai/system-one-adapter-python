"""Shared pytest configuration for cassette-backed provider tests."""

import os

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
        # JSON rather than vcrpy's default YAML, so a cassette diffs like the payloads
        # it holds instead of as a wall of block scalars.
        "serializer": "json",
        "filter_headers": FILTERED_HEADERS,
        "filter_query_parameters": ["api_key", "key"],
        "before_record_response": scrub_response,
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        # record_mode is deliberately unset: pytest-recording defaults it to "none"
        # (replay only) and setting it here would override the --record-mode flag.
        "decode_compressed_response": True,
    }
