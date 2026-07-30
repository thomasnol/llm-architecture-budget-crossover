from __future__ import annotations

"""Bounded, resumable experiment execution with failure circuit breaking."""

import asyncio
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

from .config import ExperimentConfig
from .gateway import GatewayClient, GatewayRequestError
from .io import append_jsonl, read_jsonl
from .manifest import ensure_manifest, record_phase, run_dir
from .records import Case, FailureAttempt, Generation
from .systems import run_system


def generation_path(repo: Path, config: ExperimentConfig) -> Path:
    return run_dir(repo, config) / "generations.jsonl"


def error_path(repo: Path, config: ExperimentConfig) -> Path:
    return run_dir(repo, config) / "errors.jsonl"


def _preflight_passed(repo: Path, config: ExperimentConfig) -> bool:
    path = run_dir(repo, config) / "preflight.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get("pass"))
    except (OSError, json.JSONDecodeError):
        return False


def _preflight_created_at(
    repo: Path,
    config: ExperimentConfig,
) -> datetime | None:
    path = run_dir(repo, config) / "preflight.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text()).get("created_at")
        return datetime.fromisoformat(value) if value else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _failure_attempt(
    *,
    error: Exception,
    config: ExperimentConfig,
    case: Case,
    system: str,
    token_budget: int,
    repetition: int,
    run_id: str,
    attempt_number: int,
    wall_time_seconds: float,
) -> FailureAttempt:
    if isinstance(error, GatewayRequestError):
        attempted_model = error.model
        stage = error.stage
        credential_slot = error.credential_slot
        status_code = error.status_code
        request_id = error.request_id
        retryable = error.retryable
        detail = error.detail
        signature = error.signature
    else:
        attempted_model = config.generator_model
        stage = "unknown"
        credential_slot = None
        status_code = None
        request_id = None
        retryable = isinstance(
            error,
            (
                TimeoutError,
                ConnectionError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ),
        )
        detail = str(error)[:2000]
        signature = f"{type(error).__name__}:{detail}"
    return FailureAttempt(
        run_id=run_id,
        case_id=case.case_id,
        pair_id=case.pair_id,
        counterfactual_variant=case.counterfactual_variant,
        system=system,
        token_budget=token_budget,
        repetition=repetition,
        attempted_model=attempted_model,
        stage=stage,
        credential_slot=credential_slot,
        status_code=status_code,
        request_id=request_id,
        retryable=retryable,
        error_type=type(error).__name__,
        detail=detail,
        signature=signature,
        attempt_number=attempt_number,
        wall_time_seconds=wall_time_seconds,
    )


async def execute_generation(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
) -> dict[str, int | float | bool]:
    if config.require_preflight and not _preflight_passed(repo, config):
        raise RuntimeError("a passing preflight.json is required before experiment execution")
    ensure_manifest(repo, config, cases)
    output = generation_path(repo, config)
    errors = error_path(repo, config)
    previous = read_jsonl(output, Generation)
    previous_attempts = read_jsonl(errors, FailureAttempt)
    completed = {
        (row.case_id, row.system, row.token_budget, row.repetition)
        for row in previous
        if row.status in {"ok", "budget_exhausted"}
    }
    attempt_counts = Counter(
        (row.case_id, row.system, row.token_budget, row.repetition) for row in previous_attempts
    )
    jobs = [
        (case, system, token_budget, repetition)
        for case in cases
        for system in config.systems
        for token_budget in config.token_budgets
        for repetition in range(config.repetitions)
        if (case.case_id, system, token_budget, repetition) not in completed
    ]
    random.Random(config.seed).shuffle(jobs)
    client = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    if not client.configured:
        await client.close()
        raise RuntimeError(
            "gateway is not configured; copy .env.example to .env and set the "
            "gateway endpoint plus credentials"
        )

    queue: asyncio.Queue[tuple[Case, str, int, int]] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    worker_count = max(
        1,
        min(client.maximum_total_concurrency, len(jobs) or 1),
    )
    started = time.monotonic()
    deadline = started + config.runtime_hours * 3600
    circuit_open = asyncio.Event()
    preflight_created_at = _preflight_created_at(repo, config)
    permanent_signatures = Counter(
        row.signature
        for row in previous_attempts
        if not row.retryable
        and (
            preflight_created_at is None
            or datetime.fromisoformat(row.created_at) >= preflight_created_at
        )
    )
    if any(count >= config.permanent_error_threshold for count in permanent_signatures.values()):
        circuit_open.set()
    counters: dict[str, int | float | bool] = {
        "completed": 0,
        "budget_exhausted": 0,
        "failed_attempts": 0,
        "skipped": len(completed),
        "scheduled": len(jobs),
        "launched": 0,
        "circuit_open": circuit_open.is_set(),
    }

    async def worker() -> None:
        while not circuit_open.is_set() and time.monotonic() <= deadline - 30:
            try:
                case, system, token_budget, repetition = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            cell = (case.case_id, system, token_budget, repetition)
            run_id = (
                f"{config.experiment_name}-{case.case_id}-{system}-b{token_budget}-r{repetition}"
            )
            counters["launched"] = int(counters["launched"]) + 1
            case_started = time.monotonic()
            try:
                result = await run_system(
                    client,
                    case=case,
                    system=system,
                    token_budget=token_budget,
                    config=config,
                    run_id=run_id,
                    repetition=repetition,
                )
                result.wall_time_seconds = time.monotonic() - case_started
                append_jsonl(output, result)
                if result.status == "budget_exhausted":
                    counters["budget_exhausted"] = int(counters["budget_exhausted"]) + 1
                else:
                    counters["completed"] = int(counters["completed"]) + 1
            except Exception as error:  # noqa: BLE001 - cell isolation boundary
                attempt_counts[cell] += 1
                attempt = _failure_attempt(
                    error=error,
                    config=config,
                    case=case,
                    system=system,
                    token_budget=token_budget,
                    repetition=repetition,
                    run_id=run_id,
                    attempt_number=attempt_counts[cell],
                    wall_time_seconds=time.monotonic() - case_started,
                )
                append_jsonl(errors, attempt)
                counters["failed_attempts"] = int(counters["failed_attempts"]) + 1
                if not attempt.retryable:
                    permanent_signatures[attempt.signature] += 1
                    if permanent_signatures[attempt.signature] >= config.permanent_error_threshold:
                        circuit_open.set()
                        counters["circuit_open"] = True
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
    try:
        remaining = max(1.0, deadline - time.monotonic())
        done, pending = await asyncio.wait(workers, timeout=remaining)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    finally:
        await client.close()

    scored_now = int(counters["completed"]) + int(counters["budget_exhausted"])
    counters["cancelled_at_deadline"] = queue.qsize() if time.monotonic() > deadline - 30 else 0
    counters["remaining_cells"] = len(jobs) - scored_now
    counters["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record_phase(repo, config, phase="generation", counters=counters)
    return counters
