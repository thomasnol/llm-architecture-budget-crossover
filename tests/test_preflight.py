from pathlib import Path

import httpx
import pytest

from budget_crossover.config import ExperimentConfig
from budget_crossover.gateway import GatewayClient
from budget_crossover.preflight import run_preflight


@pytest.mark.asyncio
async def test_preflight_completes_every_model_on_every_eligible_credential(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.test/oauth/token")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "first")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "secret-one")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_2", "second")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_2", "secret-two")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "gpt-5.4-mini,claude-sonnet-4-6")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_2", "gpt-5.4-mini")
    observed: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            identifier = "first" if b"client_id=first" in request.content else "second"
            return httpx.Response(
                200,
                json={"access_token": f"{identifier}-token", "expires_in": 3600},
            )
        token = request.headers["authorization"].removeprefix("Bearer ")
        model = __import__("json").loads(request.content)["model"]
        observed.append((token, model))
        return httpx.Response(
            200,
            headers={"x-request-id": f"request-{len(observed)}"},
            json={
                "model": model,
                "choices": [
                    {
                        "message": {"content": '{"status":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            },
        )

    config = ExperimentConfig(
        experiment_name="preflight-test",
        study_kind="routing",
        hmda_source_sha256="0" * 64,
        systems=["always_primary", "always_supervisor", "selective_supervisor"],
        generator_model="gpt-5.4-mini",
        supervisor_model="claude-sonnet-4-6",
    )
    client = GatewayClient()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        report = await run_preflight(
            repo=tmp_path,
            config=config,
            client=client,
        )
    finally:
        await client.close()

    assert observed == [
        ("first-token", "claude-sonnet-4-6"),
        ("first-token", "gpt-5.4-mini"),
        ("second-token", "gpt-5.4-mini"),
    ]
    assert report["pass"] is True
    assert report["checks_expected"] == 3
    assert report["checks_passed"] == 3
    assert all(check["usage_complete"] for check in report["checks"])
    assert (tmp_path / "experiments" / "runs" / "preflight-test" / "preflight.json").exists()
