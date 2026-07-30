from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from .gateway import GatewayClient
from .io import append_jsonl, read_jsonl
from .v3_config import V3Config
from .v3_manifest import ensure_v3_manifest, record_v3_phase, v3_run_dir
from .v3_models import V3Case, V3Generation
from .v3_systems import run_v3_system


def v3_generation_path(repo: Path, config: V3Config) -> Path:
    return v3_run_dir(repo, config) / "generations.jsonl"


def v3_error_path(repo: Path, config: V3Config) -> Path:
    return v3_run_dir(repo, config) / "errors.jsonl"


async def execute_v3_generation(
    *,
    repo: Path,
    config: V3Config,
    cases: list[V3Case],
) -> dict[str, int | float]:
    ensure_v3_manifest(repo, config, cases)
    output = v3_generation_path(repo, config)
    errors = v3_error_path(repo, config)
    previous = read_jsonl(output, V3Generation)
    completed = {
        (row.case_id, row.system, row.token_budget)
        for row in previous
        if row.status in {"ok", "budget_exhausted"}
    }
    jobs = [
        (case, system, token_budget)
        for case in cases
        for system in config.systems
        for token_budget in config.token_budgets
        if (case.case_id, system, token_budget) not in completed
    ]
    random.Random(config.seed).shuffle(jobs)
    client = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    if not client.configured:
        await client.close()
        raise RuntimeError(
            "gateway is not configured; copy .env.example to .env and set the "
            "gateway endpoint plus credentials"
        )
    concurrency = max(1, client.maximum_total_concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    deadline = started + config.runtime_hours * 3600
    counters: dict[str, int | float] = {
        "completed": 0,
        "budget_exhausted": 0,
        "failed": 0,
        "skipped": len(completed),
        "scheduled": len(jobs),
    }

    async def worker(case: V3Case, system: str, token_budget: int) -> None:
        if time.monotonic() > deadline - 30:
            return
        async with semaphore:
            if time.monotonic() > deadline - 30:
                return
            run_id = (
                f"{config.experiment_name}-{case.case_id}-{system}-b{token_budget}"
            )
            case_started = time.monotonic()
            try:
                result = await run_v3_system(
                    client,
                    case=case,
                    system=system,
                    token_budget=token_budget,
                    config=config,
                    run_id=run_id,
                )
                result.wall_time_seconds = time.monotonic() - case_started
                append_jsonl(output, result)
                if result.status == "budget_exhausted":
                    counters["budget_exhausted"] = (
                        int(counters["budget_exhausted"]) + 1
                    )
                else:
                    counters["completed"] = int(counters["completed"]) + 1
            except Exception as error:  # noqa: BLE001 - isolate and checkpoint failures
                append_jsonl(
                    errors,
                    V3Generation(
                        run_id=run_id,
                        case_id=case.case_id,
                        pair_id=case.pair_id,
                        counterfactual_variant=case.counterfactual_variant,
                        system=system,
                        token_budget=token_budget,
                        model=config.generator_model,
                        supervisor_model=config.supervisor_model,
                        wall_time_seconds=time.monotonic() - case_started,
                        status="error",
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
                counters["failed"] = int(counters["failed"]) + 1

    tasks = [asyncio.create_task(worker(*job)) for job in jobs]
    try:
        remaining = max(1.0, deadline - time.monotonic())
        _done, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        counters["cancelled_at_deadline"] = len(pending)
    finally:
        await client.close()
    counters["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record_v3_phase(repo, config, phase="generation", counters=counters)
    return counters
