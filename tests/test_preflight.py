import json
from pathlib import Path

import pytest

from budget_crossover.config import ExperimentConfig
from budget_crossover.models import GatewayResponse, Usage
from budget_crossover.preflight import run_preflight

TOKENIZER_HASH = "a" * 64


class ExactClient:
    tokenizer_id = "exact-test-tokenizer"
    tokenizer_sha256 = TOKENIZER_HASH

    def __init__(
        self,
        *,
        text: str = '{"status":"ok"}',
        resolved_model: str = "gpt-5.4-mini",
        counted_tokens: int = 11,
        usage: Usage | None = None,
    ) -> None:
        self.text = text
        self.resolved_model = resolved_model
        self.counted_tokens = counted_tokens
        self.usage = usage or Usage(prompt_tokens=11, completion_tokens=4, total_tokens=15)

    async def count_prompt_tokens(self, *, model: str, system: str, user: str) -> int:
        assert model == "gpt-5.4-mini"
        assert system and user
        return self.counted_tokens

    async def complete(self, **kwargs) -> GatewayResponse:
        assert kwargs["model"] == "gpt-5.4-mini"
        return GatewayResponse(
            text=self.text,
            model=self.resolved_model,
            usage=self.usage,
            latency_seconds=0.01,
            credential_slot=1,
            request_id="preflight-request",
        )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="preflight-test",
        tokenizer_id="exact-test-tokenizer",
        tokenizer_sha256=TOKENIZER_HASH,
    )


@pytest.mark.asyncio
async def test_preflight_requires_exact_model_json_usage_and_tokenizer_agreement(tmp_path: Path):
    report = await run_preflight(repo=tmp_path, config=_config(), client=ExactClient())

    assert report["pass"] is True
    assert report["requested_model"] == "gpt-5.4-mini"
    assert report["resolved_model"] == "gpt-5.4-mini"
    assert report["strict_json_valid"] is True
    assert report["authoritative_usage"] is True
    assert report["tokenizer_consistent"] is True
    assert report["tokenizer_id"] == "exact-test-tokenizer"
    written = json.loads(
        (tmp_path / "experiments" / "runs" / "preflight-test" / "preflight.json").read_text()
    )
    assert written == report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client", "failed_check"),
    [
        (ExactClient(resolved_model="gpt-5.4-mini-fallback"), "exact_model_resolution"),
        (ExactClient(text='{"status":"ok","extra":true}'), "strict_json_valid"),
        (
            ExactClient(usage=Usage(prompt_tokens=None, completion_tokens=4, total_tokens=None)),
            "authoritative_usage",
        ),
        (ExactClient(counted_tokens=12), "tokenizer_consistent"),
    ],
)
async def test_preflight_fails_closed_on_protocol_deviation(
    tmp_path: Path,
    client: ExactClient,
    failed_check: str,
):
    report = await run_preflight(repo=tmp_path, config=_config(), client=client)

    assert report["pass"] is False
    assert report[failed_check] is False
    assert report["eligible_for_empirical_run"] is False


@pytest.mark.asyncio
async def test_offline_preflight_is_permanently_non_empirical_even_when_protocol_passes(
    tmp_path: Path,
):
    config = _config().model_copy(update={"execution_mode": "offline_fixture"})

    report = await run_preflight(repo=tmp_path, config=config, client=ExactClient())

    assert report["pass"] is True
    assert report["non_empirical_fixture"] is True
    assert report["eligible_for_empirical_run"] is False
