from pathlib import Path

from budget_crossover.io import read_jsonl
from budget_crossover.v3_config import V3Config
from budget_crossover.v3_models import V3Case, V3Generation
from budget_crossover.v3_runner import (
    execute_v3_generation,
    v3_error_path,
    v3_generation_path,
)


class ConfiguredFakeClient:
    configured = True
    maximum_total_concurrency = 1

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    async def close(self) -> None:
        return None


def _case() -> V3Case:
    return V3Case(
        case_id="case-observed",
        pair_id="case",
        counterfactual_variant="observed",
        source_row_id="source",
        state="DC",
        historical_action="originated",
        policy_decision="approve",
        policy_reason_codes=["meets_policy"],
        documents={
            "application": "application",
            "collateral": "collateral",
            "credit": "credit",
            "quality_control": "quality",
            "compliance_monitoring": "monitoring",
        },
        protected_attributes={
            "race": "White",
            "sex": "Female",
            "ethnicity": "Not Hispanic or Latino",
            "age_band": "35-44",
        },
        changed_protected_attribute="race",
        complexity="routine",
    )


async def test_retryable_errors_do_not_pollute_scored_generation_grid(
    monkeypatch,
    tmp_path: Path,
):
    config = V3Config(
        experiment_name="runner-test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
    )

    async def fail(*args, **kwargs):
        raise TimeoutError("transient")

    monkeypatch.setattr(
        "budget_crossover.v3_runner.GatewayClient", ConfiguredFakeClient
    )
    monkeypatch.setattr("budget_crossover.v3_runner.run_v3_system", fail)

    report = await execute_v3_generation(
        repo=tmp_path,
        config=config,
        cases=[_case()],
    )

    assert report["failed"] == 2
    assert read_jsonl(v3_generation_path(tmp_path, config), V3Generation) == []
    errors = read_jsonl(v3_error_path(tmp_path, config), V3Generation)
    assert len(errors) == 2
    assert {row.status for row in errors} == {"error"}
