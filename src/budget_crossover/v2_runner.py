from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from .gateway import GatewayClient
from .io import append_jsonl, read_jsonl
from .v2_config import V2Config
from .v2_models import V2Case, V2Generation
from .v2_systems import run_v2_system


def v2_generation_path(repo: Path, config: V2Config) -> Path:
    return repo / "experiments" / "runs" / config.experiment_name / "generations.jsonl"


async def execute_v2_generation(
    *,
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
) -> dict[str, int | float]:
    output = v2_generation_path(repo, config)
    previous = read_jsonl(output, V2Generation)
    completed = {
        (row.case_id, row.system, row.replicate)
        for row in previous
        if row.status == "ok"
    }
    jobs = [
        (case, system, 0)
        for case in cases
        for system in config.systems
        if (case.case_id, system, 0) not in completed
    ]
    random.Random(config.seed).shuffle(jobs)
    client = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    if not client.configured:
        await client.close()
        raise RuntimeError(
            "gateway is not configured; copy .env.example to .env and set the "
            "gateway endpoint plus credentials"
        )
    semaphore = asyncio.Semaphore(config.global_case_concurrency)
    started = time.monotonic()
    deadline = started + config.generation_runtime_hours * 3600
    counters: dict[str, int | float] = {
        "completed": 0,
        "failed": 0,
        "skipped": len(completed),
        "scheduled": len(jobs),
    }

    async def worker(case: V2Case, system: str, replicate: int) -> None:
        if time.monotonic() > deadline - 30:
            return
        async with semaphore:
            run_id = f"{config.experiment_name}-{case.case_id}-{system}-r{replicate}"
            case_started = time.monotonic()
            try:
                result = await run_v2_system(
                    client,
                    case=case,
                    system=system,
                    config=config,
                    run_id=run_id,
                )
                result.wall_time_seconds = time.monotonic() - case_started
                append_jsonl(output, result)
                counters["completed"] = int(counters["completed"]) + 1
            except Exception as error:  # noqa: BLE001 - checkpoint isolated failures
                append_jsonl(
                    output,
                    V2Generation(
                        run_id=run_id,
                        case_id=case.case_id,
                        dataset=case.dataset,
                        task=case.task,
                        system=system,
                        replicate=replicate,
                        generator_model=config.generator_model,
                        verifier_model=config.verifier_model,
                        answer_text="",
                        calls=[],
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
    return counters
