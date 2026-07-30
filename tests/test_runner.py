from pathlib import Path

"""Tests for canonical resumable execution."""

from budget_crossover.config import ExperimentConfig
from budget_crossover.gateway import GatewayRequestError
from budget_crossover.io import read_jsonl
from budget_crossover.records import Case, FailureAttempt, Generation
from budget_crossover.runner import (
    error_path,
    execute_generation,
    generation_path,
)


class ConfiguredFakeClient:
    configured = True
    maximum_total_concurrency = 1

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    async def close(self) -> None:
        return None


def _case() -> Case:
    return Case(
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
    config = ExperimentConfig(
        experiment_name="runner-test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
        require_preflight=False,
    )

    async def fail(*args, **kwargs):
        raise TimeoutError("transient")

    monkeypatch.setattr("budget_crossover.runner.GatewayClient", ConfiguredFakeClient)
    monkeypatch.setattr("budget_crossover.runner.run_system", fail)

    report = await execute_generation(
        repo=tmp_path,
        config=config,
        cases=[_case()],
    )

    assert report["failed_attempts"] == 2
    assert read_jsonl(generation_path(tmp_path, config), Generation) == []
    errors = read_jsonl(error_path(tmp_path, config), FailureAttempt)
    assert len(errors) == 2
    assert {row.retryable for row in errors} == {True}


async def test_three_equivalent_permanent_errors_open_circuit_before_full_grid(
    monkeypatch,
    tmp_path: Path,
):
    config = ExperimentConfig(
        experiment_name="circuit-test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048, 4096, 8192],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
        require_preflight=False,
        permanent_error_threshold=3,
    )

    async def fail(*args, **kwargs):
        raise GatewayRequestError(
            status_code=400,
            detail='{"error":"unsupported parameter"}',
            model="claude-sonnet-4-6",
            stage="compliance_audit",
            credential_slot=1,
            request_id="request-bad",
            retryable=False,
        )

    monkeypatch.setattr("budget_crossover.runner.GatewayClient", ConfiguredFakeClient)
    monkeypatch.setattr("budget_crossover.runner.run_system", fail)

    report = await execute_generation(
        repo=tmp_path,
        config=config,
        cases=[_case()],
    )

    assert report["circuit_open"] is True
    assert report["launched"] == 3
    assert report["failed_attempts"] == 3
    assert report["remaining_cells"] == 6
    errors = read_jsonl(error_path(tmp_path, config), FailureAttempt)
    assert len(errors) == 3
    assert {row.attempted_model for row in errors} == {"claude-sonnet-4-6"}
    assert {row.stage for row in errors} == {"compliance_audit"}
    assert {row.status_code for row in errors} == {400}


async def test_transient_failures_remain_resumable_without_opening_circuit(
    monkeypatch,
    tmp_path: Path,
):
    config = ExperimentConfig(
        experiment_name="transient-test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048, 4096, 8192],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
        require_preflight=False,
    )

    async def fail(*args, **kwargs):
        raise TimeoutError("temporary timeout")

    monkeypatch.setattr("budget_crossover.runner.GatewayClient", ConfiguredFakeClient)
    monkeypatch.setattr("budget_crossover.runner.run_system", fail)

    report = await execute_generation(
        repo=tmp_path,
        config=config,
        cases=[_case()],
    )

    assert report["circuit_open"] is False
    assert report["launched"] == 6
    assert report["failed_attempts"] == 6
    assert report["remaining_cells"] == 6


async def test_repetitions_are_distinct_resumable_grid_cells(
    monkeypatch,
    tmp_path: Path,
):
    config = ExperimentConfig(
        experiment_name="repetition-test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        repetitions=2,
        bootstrap_replicates=100,
        require_preflight=False,
    )

    async def succeed(
        client,
        *,
        case,
        system,
        token_budget,
        config,
        run_id,
        repetition,
    ):
        return Generation(
            run_id=run_id,
            case_id=case.case_id,
            pair_id=case.pair_id,
            counterfactual_variant=case.counterfactual_variant,
            system=system,
            token_budget=token_budget,
            repetition=repetition,
            model=config.generator_model,
        )

    monkeypatch.setattr("budget_crossover.runner.GatewayClient", ConfiguredFakeClient)
    monkeypatch.setattr("budget_crossover.runner.run_system", succeed)

    report = await execute_generation(
        repo=tmp_path,
        config=config,
        cases=[_case()],
    )
    rows = read_jsonl(generation_path(tmp_path, config), Generation)

    assert report["completed"] == 4
    assert len(rows) == 4
    assert {(row.system, row.token_budget, row.repetition) for row in rows} == {
        ("monolith", 2048, 0),
        ("monolith", 2048, 1),
        ("adaptive", 2048, 0),
        ("adaptive", 2048, 1),
    }
