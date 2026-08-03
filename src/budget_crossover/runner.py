from __future__ import annotations

"""Deterministic, resumable execution for canonical experiment cells."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field

from .budget import BUDGET_TIERS
from .gateway import GatewayRequestError
from .io import append_jsonl, read_jsonl
from .models import CellResult, FrozenModel, PublicCase, SystemName, TierName
from .systems import CompletionClient, run_system


def generation_path(repo: Path, config: Any) -> Path:
    return repo / "experiments" / "runs" / str(config.experiment_name) / "generations.jsonl"


def error_path(repo: Path, config: Any) -> Path:
    return repo / "experiments" / "runs" / str(config.experiment_name) / "errors.jsonl"


class CellKey(FrozenModel):
    case_id: str
    system: SystemName
    tier: TierName
    repetition: int = Field(ge=0)


class ExecutionSummary(FrozenModel):
    scheduled: int = Field(ge=0)
    completed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    infrastructure_attempts: int = Field(ge=0)
    remaining: int = Field(ge=0)
    circuit_open: bool


class InfrastructureAttempt(FrozenModel):
    case_id: str
    system: SystemName
    tier: TierName
    repetition: int = Field(ge=0)
    model: str
    stage: str
    retryable: bool
    error_type: str
    detail: str
    signature: str
    attempt_number: int = Field(ge=1)
    status_code: int | None = None
    request_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _cell_tuple(
    value: CellKey | CellResult | InfrastructureAttempt,
) -> tuple[str, str, str, int]:
    return (value.case_id, value.system, value.tier, value.repetition)


def _infrastructure_attempt(
    *,
    key: CellKey,
    model: str,
    error: Exception,
    attempt_number: int,
) -> InfrastructureAttempt:
    if isinstance(error, GatewayRequestError):
        return InfrastructureAttempt(
            **key.model_dump(),
            model=error.model,
            stage=error.stage,
            retryable=error.retryable,
            error_type=type(error).__name__,
            detail=error.detail,
            signature=error.signature,
            attempt_number=attempt_number,
            status_code=error.status_code,
            request_id=error.request_id,
        )
    detail = str(error)[:2000]
    retryable = isinstance(
        error,
        (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError),
    )
    return InfrastructureAttempt(
        **key.model_dump(),
        model=model,
        stage="unknown",
        retryable=retryable,
        error_type=type(error).__name__,
        detail=detail,
        signature=f"{type(error).__name__}:{detail}",
        attempt_number=attempt_number,
    )


def build_cell_grid(
    *,
    cases: Sequence[PublicCase],
    systems: Sequence[SystemName],
    tiers: Sequence[TierName],
    repetitions: int,
) -> tuple[CellKey, ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    if len(set(systems)) != len(systems):
        raise ValueError("systems must be unique")
    if len(set(tiers)) != len(tiers):
        raise ValueError("tiers must be unique")
    return tuple(
        CellKey(
            case_id=case.case_id,
            system=system,
            tier=tier,
            repetition=repetition,
        )
        for case in cases
        for repetition in range(repetitions)
        for tier in tiers
        for system in systems
    )


async def execute_cells(
    *,
    cases: Sequence[PublicCase],
    systems: Sequence[SystemName],
    tiers: Sequence[TierName],
    repetitions: int,
    model: str,
    client: CompletionClient,
    results_path: Path,
    attempts_path: Path,
    max_concurrency: int,
) -> ExecutionSummary:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    expected_grid = build_cell_grid(
        cases=cases,
        systems=systems,
        tiers=tiers,
        repetitions=repetitions,
    )
    previous = read_jsonl(results_path, CellResult)
    previous_attempts = read_jsonl(attempts_path, InfrastructureAttempt)
    completed_keys = {_cell_tuple(result) for result in previous}
    if len(completed_keys) != len(previous):
        raise ValueError("result log contains duplicate cell keys")
    grid = tuple(key for key in expected_grid if _cell_tuple(key) not in completed_keys)
    skipped = len(expected_grid) - len(grid)
    case_by_id = {case.case_id: case for case in cases}
    queue: asyncio.Queue[CellKey] = asyncio.Queue()
    for key in grid:
        queue.put_nowait(key)

    completed = 0
    infrastructure_attempts = 0
    attempt_counts = Counter(_cell_tuple(attempt) for attempt in previous_attempts)
    permanent_signatures = Counter(
        attempt.signature for attempt in previous_attempts if not attempt.retryable
    )
    circuit_open = asyncio.Event()
    if any(count >= 3 for count in permanent_signatures.values()):
        circuit_open.set()
    append_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal completed, infrastructure_attempts
        while not circuit_open.is_set():
            try:
                key = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                try:
                    result = await run_system(
                        client,
                        case=case_by_id[key.case_id],
                        system=key.system,
                        tier=BUDGET_TIERS[key.tier],
                        model=model,
                        repetition=key.repetition,
                    )
                except Exception as error:  # noqa: BLE001 - cell infrastructure boundary
                    cell = _cell_tuple(key)
                    attempt_counts[cell] += 1
                    attempt = _infrastructure_attempt(
                        key=key,
                        model=model,
                        error=error,
                        attempt_number=attempt_counts[cell],
                    )
                    async with append_lock:
                        append_jsonl(attempts_path, attempt)
                        infrastructure_attempts += 1
                        if not attempt.retryable:
                            permanent_signatures[attempt.signature] += 1
                            if permanent_signatures[attempt.signature] >= 3:
                                circuit_open.set()
                else:
                    async with append_lock:
                        append_jsonl(results_path, result)
                        completed += 1
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(max_concurrency, len(grid)) if grid else 0)
    ]
    if workers:
        await asyncio.gather(*workers)
    return ExecutionSummary(
        scheduled=len(grid),
        completed=completed,
        skipped=skipped,
        infrastructure_attempts=infrastructure_attempts,
        remaining=len(grid) - completed,
        circuit_open=circuit_open.is_set(),
    )


execute_generation = execute_cells
