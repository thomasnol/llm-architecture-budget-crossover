from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from .architectures import run_architecture
from .config import ExperimentConfig
from .gateway import GatewayClient
from .io import append_jsonl, read_jsonl
from .models import Case, ExperimentResult


def generation_path(repo: Path, config: ExperimentConfig) -> Path:
    return repo / "experiments" / "runs" / config.experiment_name / "generations.jsonl"


async def execute_generation_run(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
) -> dict[str, int | float]:
    output_path = generation_path(repo, config)
    previous = read_jsonl(output_path, ExperimentResult)
    complete = {
        (row.case_id, row.architecture, row.nominal_budget)
        for row in previous
        if row.status == "ok"
    }
    jobs = [
        (case, architecture, budget)
        for case in cases
        for architecture in config.architectures
        for budget in config.budgets
        if (case.case_id, architecture, budget) not in complete
    ]
    random.Random(config.seed).shuffle(jobs)
    client = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    if not client.configured:
        await client.close()
        raise RuntimeError(
            "gateway is not configured; set LLM_GATEWAY_BASE_URL and credential variables"
        )
    case_semaphore = asyncio.Semaphore(
        max(config.global_case_concurrency, client.maximum_total_concurrency)
    )
    started = time.monotonic()
    deadline = started + config.generation_runtime_hours * 3600
    counters = {"completed": 0, "failed": 0, "skipped": len(complete)}

    async def worker(case: Case, architecture: str, budget: int) -> None:
        if time.monotonic() > deadline - 30:
            return
        async with case_semaphore:
            run_id = f"{config.experiment_name}-{case.case_id}-{architecture}-{budget}"
            try:
                result = await run_architecture(
                    client,
                    case=case,
                    architecture=architecture,
                    budget=budget,
                    model=config.generator_model,
                    temperature=config.temperature,
                    run_id=run_id,
                )
                append_jsonl(output_path, result)
                counters["completed"] += 1
            except Exception as error:  # noqa: BLE001 - checkpoint any per-job failure
                failure = ExperimentResult(
                    run_id=run_id,
                    case_id=case.case_id,
                    task=case.task,
                    architecture=architecture,
                    nominal_budget=budget,
                    model=config.generator_model,
                    answer_text="",
                    parsed_answer="",
                    calls=[],
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                )
                append_jsonl(output_path, failure)
                counters["failed"] += 1

    tasks = [asyncio.create_task(worker(*job)) for job in jobs]
    try:
        remaining = max(1.0, deadline - time.monotonic())
        _done, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await client.close()
    counters["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return counters
