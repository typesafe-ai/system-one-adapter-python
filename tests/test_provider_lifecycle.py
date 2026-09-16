"""Provider reuse and ownership, with real SDK pools and no model API calls."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Barrier
from typing import Any
from unittest.mock import DEFAULT, AsyncMock, Mock

import pytest
from typesafe_sdk import TypeSafeError

from system_one_adapter import AsyncSystemOneAdapterClient, Noul, SystemOneAdapterClient
from system_one_adapter.providers import Message, ProviderResult
from system_one_adapter.providers.anthropic import (
    AnthropicProvider,
    AsyncAnthropicProvider,
)
from system_one_adapter.providers.base import record_request, render_messages
from system_one_adapter.providers.openai import AsyncOpenAIProvider, OpenAIProvider
from tests.test_client_with_fake_model import FakeAsyncProvider, FakeSyncProvider

QUESTIONS = {"positive": Noul(instructions="The review is positive.")}
RESULT = ProviderResult('{"answers":{"positive":true}}', 11, 7)


@dataclass
class Lifecycle:
    asynchronous: bool
    vendor: str
    providers: list[Any] = field(default_factory=list)

    def client(self, **kwargs: Any) -> Any:
        options: dict[str, Any] = {
            "structured_outputs": True,
            "llm_answer_mode": "discrete",
            "provider": self.vendor,
            "model": "test-model",
        }
        options.update(kwargs)
        client_class = AsyncSystemOneAdapterClient if self.asynchronous else SystemOneAdapterClient
        return client_class(**options)

    async def evaluate(self, client: Any, state: str = "Great book", **kwargs: Any) -> Any:
        response = client.system_one(state, QUESTIONS, **kwargs)
        return await response if self.asynchronous else response

    async def close(self, client: Any) -> None:
        if self.asynchronous:
            await client.aclose()
        else:
            client.close()

    @asynccontextmanager
    async def context(self, client: Any) -> AsyncIterator[Any]:
        if self.asynchronous:
            async with client:
                yield client
        else:
            with client:
                yield client


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def lifecycle(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, vendor: str) -> Iterator[Lifecycle]:
    case = Lifecycle(request.param, vendor)

    def watch(provider_class: Any) -> None:
        original_init = provider_class.__init__

        def initialize(provider: Any, *args: Any, **kwargs: Any) -> None:
            original_init(provider, *args, **kwargs)
            spy = AsyncMock if case.asynchronous else Mock
            monkeypatch.setattr(provider._client, "close", spy(wraps=provider._client.close))
            case.providers.append(provider)

        def respond(provider: Any, messages: list[Message], **kwargs: Any) -> ProviderResult:
            record_request({"messages": render_messages(messages)}, api="offline-test")
            return RESULT

        async def respond_async(provider: Any, messages: list[Message], **kwargs: Any) -> ProviderResult:
            result = respond(provider, messages, **kwargs)
            await asyncio.sleep(0)
            return result

        monkeypatch.setattr(provider_class, "__init__", initialize)
        monkeypatch.setattr(provider_class, "request", respond_async if case.asynchronous else respond)

    for provider_class in (
        (AsyncOpenAIProvider, AsyncAnthropicProvider) if case.asynchronous else (OpenAIProvider, AnthropicProvider)
    ):
        watch(provider_class)
    yield case

    # Retaining providers makes missed cleanup observable; release test pools even
    # when an assertion fails or a close spy has deliberately been made to raise.
    async def cleanup() -> None:
        for provider in case.providers:
            sdk = provider._client
            if not sdk.is_closed():
                result = type(sdk).close(sdk)
                if case.asynchronous:
                    await result

    asyncio.run(cleanup())


@pytest.fixture(params=["openai", "anthropic"])
def vendor(request: pytest.FixtureRequest) -> str:
    return request.param


def test_reuses_owned_provider_and_closes_sdk_on_context_exit(lifecycle: Lifecycle) -> None:
    async def run() -> None:
        client = lifecycle.client()
        assert lifecycle.providers == []
        async with lifecycle.context(client):
            for _ in range(3):
                response = await lifecycle.evaluate(client)
                assert response.nouls["positive"].noul == 1
            assert len(lifecycle.providers) == 1
            assert not lifecycle.providers[0]._client.is_closed()
        sdk = lifecycle.providers[0]._client
        assert sdk.is_closed()
        await lifecycle.close(client)
        sdk.close.assert_called_once()
        if lifecycle.asynchronous:
            sdk.close.assert_awaited_once()
        with pytest.raises(RuntimeError, match="closed"):
            await lifecycle.evaluate(client)
        with pytest.raises(RuntimeError, match="closed"):
            async with lifecycle.context(client):
                pytest.fail("Reentered a closed client")

    asyncio.run(run())


def test_cache_uses_resolved_provider_and_model_and_is_per_client(lifecycle: Lifecycle) -> None:
    async def run() -> None:
        async with lifecycle.context(lifecycle.client()) as client:
            await lifecycle.evaluate(client)
            await lifecycle.evaluate(client, model="test-model", provider=lifecycle.vendor)
            await lifecycle.evaluate(client, model="another-model")
            other_vendor = "anthropic" if lifecycle.vendor == "openai" else "openai"
            await lifecycle.evaluate(client, provider=other_vendor)
            assert len(lifecycle.providers) == 3
            async with lifecycle.context(lifecycle.client()) as second:
                await lifecycle.evaluate(second)
                assert len(lifecycle.providers) == 4
            assert all(not provider._client.is_closed() for provider in lifecycle.providers[:3])
        assert all(provider._client.is_closed() for provider in lifecycle.providers)

    asyncio.run(run())


@pytest.mark.parametrize("constructor_default", [False, True])
def test_injected_provider_is_borrowed(lifecycle: Lifecycle, constructor_default: bool) -> None:
    async def run() -> None:
        provider_class = (
            (AsyncOpenAIProvider if lifecycle.asynchronous else OpenAIProvider)
            if lifecycle.vendor == "openai"
            else (AsyncAnthropicProvider if lifecycle.asynchronous else AnthropicProvider)
        )
        injected: Any = provider_class("test-model")
        kwargs: dict[str, Any] = {} if constructor_default else {"model": injected}
        client = lifecycle.client(model=injected) if constructor_default else lifecycle.client()
        async with lifecycle.context(client):
            await lifecycle.evaluate(client, **kwargs)
            await lifecycle.evaluate(client, model="owned-model")
            await lifecycle.evaluate(client, **kwargs)
        assert not injected._client.is_closed()
        injected._client.close.assert_not_called()
        assert lifecycle.providers[1]._client.is_closed()
        with pytest.raises(RuntimeError, match="closed"):
            await lifecycle.evaluate(client, model=injected)
        if lifecycle.asynchronous:
            await injected.aclose()
        else:
            injected.close()
        assert injected._client.is_closed()

    asyncio.run(run())


def test_custom_provider_without_close_remains_supported(lifecycle: Lifecycle) -> None:
    async def run() -> None:
        provider_class = FakeAsyncProvider if lifecycle.asynchronous else FakeSyncProvider
        provider = provider_class({"answers": {"positive": True}})
        async with lifecycle.context(lifecycle.client(model=provider)) as client:
            await lifecycle.evaluate(client)
        assert len(provider.calls) == 1
        assert lifecycle.providers == []

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["body", "request", "validation", "cancelled"])
def test_exceptional_exit_closes_owned_sdks(lifecycle: Lifecycle, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    async def run() -> None:
        error_type = {"body": ValueError, "request": TypeSafeError, "validation": ValueError, "cancelled": asyncio.CancelledError}[failure]
        with pytest.raises(error_type):
            async with lifecycle.context(lifecycle.client()) as client:
                if failure == "validation":
                    response = client.system_one("document", {})
                    if lifecycle.asynchronous:
                        await response
                else:
                    await lifecycle.evaluate(client)
                    if failure == "request":
                        mock_class = AsyncMock if lifecycle.asynchronous else Mock
                        monkeypatch.setattr(lifecycle.providers[0], "request", mock_class(side_effect=TypeSafeError("failed")))
                        await lifecycle.evaluate(client)
                    else:
                        raise error_type("interrupted")
        assert len(lifecycle.providers) == 1
        assert lifecycle.providers[0]._client.is_closed()

    asyncio.run(run())


def test_cleanup_continues_after_failure(lifecycle: Lifecycle) -> None:
    async def run() -> None:
        client = lifecycle.client()
        await lifecycle.evaluate(client)
        await lifecycle.evaluate(client, model="another-model")
        lifecycle.providers[0]._client.close.side_effect = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            await lifecycle.close(client)
        assert lifecycle.providers[1]._client.is_closed()
        await lifecycle.close(client)
        for provider in lifecycle.providers:
            provider._client.close.assert_called_once()
        with pytest.raises(RuntimeError, match="closed"):
            await lifecycle.evaluate(client)

    asyncio.run(run())


def test_close_before_first_use_does_not_construct_providers(lifecycle: Lifecycle) -> None:
    async def run() -> None:
        client = lifecycle.client()
        await lifecycle.close(client)
        await lifecycle.close(client)
        with pytest.raises(RuntimeError, match="closed"):
            await lifecycle.evaluate(client)
        assert lifecycle.providers == []

    asyncio.run(run())


def test_failed_construction_is_not_cached(lifecycle: Lifecycle, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with lifecycle.context(lifecycle.client()) as client:
            factory = Mock(wraps=client._build_provider, side_effect=[ValueError("constructor failed"), DEFAULT])
            monkeypatch.setattr(client, "_build_provider", factory)
            with pytest.raises(ValueError, match="constructor failed"):
                await lifecycle.evaluate(client)
            assert lifecycle.providers == []
            await lifecycle.evaluate(client)
            await lifecycle.evaluate(client)
            assert factory.call_count == 2
            assert len(lifecycle.providers) == 1

    asyncio.run(run())


def test_environment_is_captured_on_first_use(lifecycle: Lifecycle, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        prefix = lifecycle.vendor.upper()
        client = lifecycle.client()
        monkeypatch.setenv(f"{prefix}_API_KEY", "first-test-key")
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://first.invalid/v1")
        async with lifecycle.context(client):
            await lifecycle.evaluate(client)
            monkeypatch.setenv(f"{prefix}_API_KEY", "second-test-key")
            monkeypatch.setenv(f"{prefix}_BASE_URL", "https://second.invalid/v1")
            await lifecycle.evaluate(client)
            assert len(lifecycle.providers) == 1
            sdk = lifecycle.providers[0]._client
            assert sdk.api_key == "first-test-key"
            assert sdk.base_url.host == "first.invalid"
        async with lifecycle.context(lifecycle.client()) as fresh:
            await lifecycle.evaluate(fresh)
            sdk = lifecycle.providers[1]._client
            assert sdk.api_key == "second-test-key"
            assert sdk.base_url.host == "second.invalid"

    asyncio.run(run())


def test_async_cleanup_propagates_cancellation_after_remaining_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        client = AsyncSystemOneAdapterClient(structured_outputs=True, llm_answer_mode="discrete", provider="openai")
        first = FakeAsyncProvider(RESULT.text)
        second = FakeAsyncProvider(RESULT.text)
        cancelled_close = AsyncMock(side_effect=asyncio.CancelledError())
        second_close = AsyncMock()
        monkeypatch.setattr(first, "aclose", cancelled_close, raising=False)
        monkeypatch.setattr(second, "aclose", second_close, raising=False)
        monkeypatch.setattr(client, "_build_provider", Mock(side_effect=[first, second]))
        await client.system_one("document", QUESTIONS, model="first")
        await client.system_one("document", QUESTIONS, model="second")
        with pytest.raises(asyncio.CancelledError):
            await client.aclose()
        cancelled_close.assert_awaited_once()
        second_close.assert_awaited_once()
        await client.aclose()
        second_close.assert_awaited_once()

    asyncio.run(run())


def test_concurrent_first_use_reuses_pool_and_isolates_traces(lifecycle: Lifecycle) -> None:
    async def run() -> None:
        async with lifecycle.context(lifecycle.client()) as client:
            if lifecycle.asynchronous:
                responses = await asyncio.gather(*(lifecycle.evaluate(client, f"document-{i}") for i in range(8)))
            else:
                barrier = Barrier(8)

                def evaluate(index: int) -> Any:
                    barrier.wait(timeout=10)
                    return client.system_one(f"document-{index}", QUESTIONS)

                with ThreadPoolExecutor(max_workers=8) as executor:
                    responses = list(executor.map(evaluate, range(8)))
            assert len(lifecycle.providers) == 1
            for index, response in enumerate(responses):
                attempts = response.debug["llm_attempts"]
                assert len(attempts) == 1
                assert f"document-{index}" in attempts[0]["messages"][1]["content"]
                assert attempts[0]["request"]["messages"] == attempts[0]["messages"]
                assert response.usage.input_tokens_total == 11
                assert response.usage.n_retries == 0

    asyncio.run(run())
