from __future__ import annotations

import httpx
import pytest

from budget_crossover.gateway import (
    CubicConcurrencyLimiter,
    GatewayClient,
    GatewayCompletionClient,
    GatewayRequestError,
    PromptTokenCount,
)
from budget_crossover.models import GatewayResponse, Usage


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
async def test_default_model_allowlists_expose_only_the_exact_canonical_model(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.test/oauth/token")
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
        assert all(slot.supports("gpt-5.4-mini") for slot in client.slots)
        assert not any(slot.supports("gpt-5.4") for slot in client.slots)
        assert not any(slot.supports("gpt-5.4-nano") for slot in client.slots)
        assert not any(slot.supports("claude-sonnet-4-6") for slot in client.slots)
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


@pytest.mark.asyncio
async def test_gateway_derives_total_without_mutating_frozen_usage(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY_1", "api-key")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "gpt-5.4-mini")
    for name in (
        "LLM_GATEWAY_API_KEY_2",
        "LLM_GATEWAY_CLIENT_ID_2",
        "LLM_GATEWAY_CLIENT_SECRET_2",
    ):
        monkeypatch.delenv(name, raising=False)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-5.4-mini",
                "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2},
            },
        )

    client = GatewayClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await client.complete(
            model="gpt-5.4-mini",
            system="system",
            user="case",
            max_tokens=64,
        )
    finally:
        await client.close()

    assert response.usage.total_tokens == 13
    assert response.usage.authoritative_total == 13


@pytest.mark.asyncio
async def test_credential_report_authenticates_both_pairs(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.test/oauth/token")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "first")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "secret-one")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_2", "second")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_2", "secret-two")
    token_ids: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            identifier = "first" if b"client_id=first" in request.content else "second"
            token_ids.add(identifier)
            return httpx.Response(
                200,
                json={"access_token": f"{identifier}-token", "expires_in": 3600},
            )
        token = request.headers["authorization"].removeprefix("Bearer ")
        model = "claude-sonnet-4-6" if token.startswith("first") else "gpt-5.4-mini"
        return httpx.Response(200, json={"data": [{"id": model}]})

    client = GatewayClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        report = await client.credential_report()
    finally:
        await client.close()

    assert token_ids == {"first", "second"}
    assert report["configured_credential_slots"] == 2
    assert [slot["status"] for slot in report["slots"]] == ["ok", "ok"]
    assert report["slots"][0]["reported_model_ids"] == ["claude-sonnet-4-6"]
    assert report["slots"][1]["reported_model_ids"] == ["gpt-5.4-mini"]


@pytest.mark.asyncio
async def test_nonretryable_gateway_error_preserves_sanitized_request_context(
    monkeypatch,
):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.test/oauth/token")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "first")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "secret-one")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "claude-sonnet-4-6")
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_ID_2", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_SECRET_2", raising=False)
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={"access_token": "test-token", "expires_in": 3600},
            )
        return httpx.Response(
            400,
            headers={"x-request-id": "bad-request-17"},
            json={"error": {"message": "temperature is unsupported; secret-one must be hidden"}},
        )

    client = GatewayClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GatewayRequestError) as raised:
            await client.complete(
                model="claude-sonnet-4-6",
                system="system",
                user="case",
                max_tokens=64,
                stage="strong_monolith",
                credential_slot=1,
            )
    finally:
        await client.close()

    error = raised.value
    assert requests == 2  # one OAuth request and one non-retried chat request
    assert error.status_code == 400
    assert error.model == "claude-sonnet-4-6"
    assert error.stage == "strong_monolith"
    assert error.credential_slot == 1
    assert error.request_id == "bad-request-17"
    assert error.retryable is False
    assert "temperature is unsupported" in error.detail
    assert "secret-one" not in error.detail
    assert "secret-one" not in str(error)


@pytest.mark.asyncio
async def test_transport_failure_remains_retryable_after_gateway_retries(
    monkeypatch,
):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY_1", "api-key")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "gpt-5.4-mini")
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_ID_2", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_SECRET_2", raising=False)
    monkeypatch.setattr("budget_crossover.gateway.asyncio.sleep", _no_sleep)
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ConnectError("temporary disconnect", request=request)

    client = GatewayClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GatewayRequestError) as raised:
            await client.complete(
                model="gpt-5.4-mini",
                system="system",
                user="case",
                max_tokens=64,
                stage="monolith",
            )
    finally:
        await client.close()

    assert requests == 5
    assert raised.value.retryable is True
    assert raised.value.status_code is None
    assert raised.value.stage == "monolith"


async def _no_sleep(_seconds: float) -> None:
    return None


class _Tokenizer:
    tokenizer_id = "tokenizer-v1"
    tokenizer_sha256 = "b" * 64

    async def count(self, *, model: str, system: str, user: str) -> PromptTokenCount:
        assert (model, system, user) == ("gpt-5.4-mini", "system", "user")
        return PromptTokenCount(
            model=model,
            prompt_tokens=17,
            tokenizer_id=self.tokenizer_id,
            tokenizer_sha256=self.tokenizer_sha256,
        )


class _Gateway:
    async def complete(self, **kwargs) -> GatewayResponse:
        return GatewayResponse(
            text='{"status":"ok"}',
            model=kwargs.pop("resolved_model", "gpt-5.4-mini"),
            usage=Usage(prompt_tokens=17, completion_tokens=4, total_tokens=21),
            latency_seconds=0.01,
            credential_slot=1,
        )


@pytest.mark.asyncio
async def test_completion_adapter_uses_exact_tokenizer_and_authoritative_gateway_usage():
    adapter = GatewayCompletionClient(
        gateway=_Gateway(),
        tokenizer=_Tokenizer(),
        model="gpt-5.4-mini",
        tokenizer_id="tokenizer-v1",
        tokenizer_sha256="b" * 64,
    )

    count = await adapter.count_prompt_tokens(
        model="gpt-5.4-mini", system="system", user="user"
    )
    response = await adapter.complete(
        model="gpt-5.4-mini",
        system="system",
        user="user",
        max_tokens=32,
        stage="preflight",
    )

    assert count == 17
    assert response.usage.authoritative_total == 21
    assert response.model == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_completion_adapter_refuses_any_requested_or_resolved_model_substitution():
    adapter = GatewayCompletionClient(
        gateway=_Gateway(),
        tokenizer=_Tokenizer(),
        model="gpt-5.4-mini",
        tokenizer_id="tokenizer-v1",
        tokenizer_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="model substitution"):
        await adapter.count_prompt_tokens(model="gpt-5.4", system="system", user="user")

    class SubstitutingGateway(_Gateway):
        async def complete(self, **kwargs) -> GatewayResponse:
            response = await super().complete(**kwargs)
            response.model = "gpt-5.4-mini-fallback"
            return response

    substituting = GatewayCompletionClient(
        gateway=SubstitutingGateway(),
        tokenizer=_Tokenizer(),
        model="gpt-5.4-mini",
        tokenizer_id="tokenizer-v1",
        tokenizer_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="resolved model"):
        await substituting.complete(
            model="gpt-5.4-mini",
            system="system",
            user="user",
            max_tokens=32,
            stage="preflight",
        )
