from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from budget_crossover.gateway import GatewayRequestError
from budget_crossover.io import append_jsonl, read_jsonl
from budget_crossover.models import CellResult, EvidenceItem, MechanismTrace, PublicCase
from budget_crossover.runner import (
    CellKey,
    InfrastructureAttempt,
    build_cell_grid,
    execute_cells,
)


def _case(case_id: str) -> PublicCase:
    return PublicCase(
        case_id=case_id,
        dataset="finqa",
        document_id=f"doc-{case_id}",
        question="What is the value?",
        evidence=(
            EvidenceItem(
                evidence_id=f"{case_id}-e1",
                document_id=f"doc-{case_id}",
                kind="text",
                text="The value is 10.",
                ordinal=0,
            ),
        ),
        stratum="headroom",
    )


def test_cell_grid_is_immutable_and_deterministically_interleaved_within_cases():
    grid = build_cell_grid(
        cases=[_case("a"), _case("b")],
        systems=("monolith", "verified_search"),
        tiers=("low", "high"),
        repetitions=2,
    )

    assert grid == (
        CellKey(case_id="a", system="monolith", tier="low", repetition=0),
        CellKey(case_id="a", system="verified_search", tier="low", repetition=0),
        CellKey(case_id="a", system="monolith", tier="high", repetition=0),
        CellKey(case_id="a", system="verified_search", tier="high", repetition=0),
        CellKey(case_id="a", system="monolith", tier="low", repetition=1),
        CellKey(case_id="a", system="verified_search", tier="low", repetition=1),
        CellKey(case_id="a", system="monolith", tier="high", repetition=1),
        CellKey(case_id="a", system="verified_search", tier="high", repetition=1),
        CellKey(case_id="b", system="monolith", tier="low", repetition=0),
        CellKey(case_id="b", system="verified_search", tier="low", repetition=0),
        CellKey(case_id="b", system="monolith", tier="high", repetition=0),
        CellKey(case_id="b", system="verified_search", tier="high", repetition=0),
        CellKey(case_id="b", system="monolith", tier="low", repetition=1),
        CellKey(case_id="b", system="verified_search", tier="low", repetition=1),
        CellKey(case_id="b", system="monolith", tier="high", repetition=1),
        CellKey(case_id="b", system="verified_search", tier="high", repetition=1),
    )
    with pytest.raises(ValidationError):
        grid[0].tier = "middle"


async def test_execution_is_bounded_and_appends_each_terminal_result(
    monkeypatch,
    tmp_path: Path,
):
    active = 0
    peak = 0

    async def succeed(client, *, case, system, tier, model, repetition):
        del client, model
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status="ok",
            candidate=None,
            trace=MechanismTrace(exit_reason="completed"),
        )

    monkeypatch.setattr("budget_crossover.runner.run_system", succeed)
    results_path = tmp_path / "results.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"

    summary = await execute_cells(
        cases=[_case("a"), _case("b")],
        systems=("monolith", "verified_search"),
        tiers=("low",),
        repetitions=1,
        model="gpt-test",
        client=object(),
        results_path=results_path,
        attempts_path=attempts_path,
        max_concurrency=2,
    )

    rows = read_jsonl(results_path, CellResult)
    assert peak == 2
    assert summary.scheduled == 4
    assert summary.completed == 4
    assert summary.infrastructure_attempts == 0
    assert len(rows) == 4
    assert len({(row.case_id, row.system, row.tier, row.repetition) for row in rows}) == 4


async def test_resume_skips_terminal_keys_without_duplicate_rows(monkeypatch, tmp_path: Path):
    results_path = tmp_path / "results.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    append_jsonl(
        results_path,
        CellResult(
            case_id="a",
            system="monolith",
            tier="low",
            repetition=0,
            status="architecture_failure",
            candidate=None,
            trace=MechanismTrace(exit_reason="invalid_output"),
        ),
    )
    launched: list[tuple[str, str]] = []

    async def succeed(client, *, case, system, tier, model, repetition):
        del client, model
        launched.append((case.case_id, system))
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status="ok",
            candidate=None,
            trace=MechanismTrace(exit_reason="completed"),
        )

    monkeypatch.setattr("budget_crossover.runner.run_system", succeed)

    summary = await execute_cells(
        cases=[_case("a")],
        systems=("monolith", "verified_search"),
        tiers=("low",),
        repetitions=1,
        model="gpt-test",
        client=object(),
        results_path=results_path,
        attempts_path=attempts_path,
        max_concurrency=2,
    )

    rows = read_jsonl(results_path, CellResult)
    assert launched == [("a", "verified_search")]
    assert summary.scheduled == 1
    assert summary.skipped == 1
    assert len(rows) == 2
    assert len({(row.case_id, row.system, row.tier, row.repetition) for row in rows}) == 2


async def test_infrastructure_errors_are_unscored_attempts_but_architecture_failures_are_terminal(
    monkeypatch,
    tmp_path: Path,
):
    async def mixed_outcomes(client, *, case, system, tier, model, repetition):
        del client, model
        if system == "verified_search":
            raise TimeoutError("temporary gateway timeout")
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status="architecture_failure",
            candidate=None,
            trace=MechanismTrace(exit_reason="budget_exhausted"),
        )

    monkeypatch.setattr("budget_crossover.runner.run_system", mixed_outcomes)
    results_path = tmp_path / "results.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"

    summary = await execute_cells(
        cases=[_case("a")],
        systems=("monolith", "verified_search"),
        tiers=("low",),
        repetitions=1,
        model="gpt-test",
        client=object(),
        results_path=results_path,
        attempts_path=attempts_path,
        max_concurrency=1,
    )

    results = read_jsonl(results_path, CellResult)
    attempts = read_jsonl(attempts_path, InfrastructureAttempt)
    assert [(row.system, row.status) for row in results] == [
        ("monolith", "architecture_failure")
    ]
    assert len(attempts) == 1
    assert attempts[0].system == "verified_search"
    assert attempts[0].retryable is True
    assert attempts[0].attempt_number == 1
    assert summary.completed == 1
    assert summary.infrastructure_attempts == 1
    assert summary.remaining == 1


async def test_mismatched_returned_cell_key_is_a_protocol_attempt_not_a_result(
    monkeypatch,
    tmp_path: Path,
):
    async def return_wrong_key(client, *, case, system, tier, model, repetition):
        del client, case, system, model
        return CellResult(
            case_id="different-case",
            system="verified_search",
            tier=tier.name,
            repetition=repetition,
            status="ok",
            candidate=None,
            trace=MechanismTrace(exit_reason="completed"),
        )

    monkeypatch.setattr("budget_crossover.runner.run_system", return_wrong_key)
    results_path = tmp_path / "results.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"

    summary = await execute_cells(
        cases=[_case("a")],
        systems=("monolith",),
        tiers=("low",),
        repetitions=1,
        model="gpt-test",
        client=object(),
        results_path=results_path,
        attempts_path=attempts_path,
        max_concurrency=1,
    )

    assert read_jsonl(results_path, CellResult) == []
    attempts = read_jsonl(attempts_path, InfrastructureAttempt)
    assert len(attempts) == 1
    assert attempts[0].stage == "result_protocol"
    assert attempts[0].retryable is False
    assert attempts[0].error_type == "ResultKeyMismatch"
    assert "different-case" in attempts[0].detail
    assert summary.completed == 0
    assert summary.infrastructure_attempts == 1
    assert summary.remaining == 1


async def test_three_equivalent_permanent_errors_open_and_resume_the_circuit(
    monkeypatch,
    tmp_path: Path,
):
    launches = 0

    async def permanently_fail(client, *, case, system, tier, model, repetition):
        del client, case, system, tier, model, repetition
        nonlocal launches
        launches += 1
        raise GatewayRequestError(
            status_code=400,
            detail="unsupported parameter",
            model="gpt-test",
            stage="planner",
            credential_slot=1,
            request_id=f"request-{launches}",
            retryable=False,
        )

    monkeypatch.setattr("budget_crossover.runner.run_system", permanently_fail)
    results_path = tmp_path / "results.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    kwargs = {
        "cases": [_case("a")],
        "systems": ("monolith", "verified_search"),
        "tiers": ("low", "middle", "high"),
        "repetitions": 1,
        "model": "gpt-test",
        "client": object(),
        "results_path": results_path,
        "attempts_path": attempts_path,
        "max_concurrency": 1,
    }

    first = await execute_cells(**kwargs)

    assert launches == 3
    assert first.circuit_open is True
    assert first.infrastructure_attempts == 3
    assert first.remaining == 6
    assert len(read_jsonl(attempts_path, InfrastructureAttempt)) == 3

    resumed = await execute_cells(**kwargs)

    assert launches == 3
    assert resumed.circuit_open is True
    assert resumed.infrastructure_attempts == 0
    assert len(read_jsonl(attempts_path, InfrastructureAttempt)) == 3


async def test_permanent_error_equivalence_ignores_credential_provenance(
    monkeypatch,
    tmp_path: Path,
):
    launches = 0

    async def permanently_fail(client, *, case, system, tier, model, repetition):
        del client, case, system, tier, model, repetition
        nonlocal launches
        launches += 1
        raise GatewayRequestError(
            status_code=400,
            detail="unsupported parameter",
            model="gpt-test",
            stage="planner",
            credential_slot=1 if launches % 2 else 2,
            request_id=f"request-{launches}",
            retryable=False,
        )

    monkeypatch.setattr("budget_crossover.runner.run_system", permanently_fail)
    attempts_path = tmp_path / "attempts.jsonl"

    summary = await execute_cells(
        cases=[_case("a")],
        systems=("monolith", "verified_search"),
        tiers=("low", "middle", "high"),
        repetitions=1,
        model="gpt-test",
        client=object(),
        results_path=tmp_path / "results.jsonl",
        attempts_path=attempts_path,
        max_concurrency=1,
    )

    attempts = read_jsonl(attempts_path, InfrastructureAttempt)
    assert launches == 3
    assert summary.circuit_open is True
    assert {attempt.status_code for attempt in attempts} == {400}
    assert {attempt.stage for attempt in attempts} == {"planner"}
    assert {attempt.model for attempt in attempts} == {"gpt-test"}
