from __future__ import annotations

import httpx
import pytest

from budget_crossover.gateway import CubicConcurrencyLimiter, GatewayClient


@pytest.mark.asyncio
async def test_cubic_limiter_grows_then_multiplicatively_decreases():
    limiter = CubicConcurrencyLimiter(max_concurrency=12, initial_concurrency=4)
    assert limiter.limit == 4

    for _ in range(24):
        await limiter.acquire()
        await limiter.release(successful=True)

    grown = limiter.window
    assert 4 < grown <= 12

    await limiter.acquire()
    await limiter.release(congestion=True)
    assert limiter.window == pytest.approx(max(1.0, grown * 0.7))
    assert limiter.limit <= int(grown)


@pytest.mark.asyncio
async def test_model_allowlists_match_the_two_gateway_credentials(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.test/oauth/token")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "first")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "secret-one")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_2", "second")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_2", "secret-two")
    monkeypatch.setenv("LLM_GATEWAY_CONCURRENCY_PER_KEY", "9")
    monkeypatch.delenv("LLM_GATEWAY_MODELS_1", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_MODELS_2", raising=False)

    client = GatewayClient()
    try:
        assert len(client.slots) == 2
        assert client.maximum_total_concurrency == 18
        assert client.slots[0].limiter.limit == 4
        assert client.slots[0].limiter.max_concurrency == 9
        assert client.slots[0].supports("claude-opus-4-6")
        assert client.slots[0].supports("claude-sonnet-4-6")
        assert not client.slots[1].supports("claude-sonnet-4-6")
        assert all(slot.supports("gpt-5.4") for slot in client.slots)
        assert all(slot.supports("gpt-5.4-mini") for slot in client.slots)
        assert all(slot.supports("gpt-5.4-nano") for slot in client.slots)
        assert (await client._slot("claude-opus-4-6")).index == 1
        with pytest.raises(RuntimeError, match="no configured credential"):
            await client._slot("unknown-model")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_oauth_pair_and_usage_payload_work_end_to_end(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.test/oauth/token")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "first")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "secret-one")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "claude-sonnet-4-6")
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_ID_2", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_SECRET_2", raising=False)

    token_requests = 0
    chat_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests, chat_requests
        if request.url.path == "/oauth/token":
            token_requests += 1
            assert b"client_id=first" in request.content
            assert b"client_secret=secret-one" in request.content
            return httpx.Response(
                200,
                json={"access_token": "test-token", "expires_in": 3600},
            )
        chat_requests += 1
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            headers={"x-request-id": "request-1"},
            json={
                "model": "claude-sonnet-4-6",
                "choices": [
                    {
                        "message": {"content": '{"decision":"yes"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 17,
                    "total_tokens": 118,
                },
            },
        )

    client = GatewayClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        first = await client.complete(
            model="claude-sonnet-4-6",
            system="system",
            user="case",
            max_tokens=64,
        )
        second = await client.complete(
            model="claude-sonnet-4-6",
            system="system",
            user="case two",
            max_tokens=64,
        )
    finally:
        await client.close()

    assert token_requests == 1
    assert chat_requests == 2
    assert first.credential_slot == 1
    assert first.usage.prompt_tokens == 101
    assert first.usage.completion_tokens == 17
    assert first.usage.total_tokens == 118
    assert first.concurrency_window is not None
    assert second.concurrency_window >= first.concurrency_window
