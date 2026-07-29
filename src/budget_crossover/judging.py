from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from .config import ExperimentConfig
from .gateway import GatewayClient
from .io import append_jsonl, read_jsonl
from .models import Case, ExperimentResult, JudgeResult
from .prompts import JUDGE_SYSTEM, judge_prompt
from .runner import generation_path


def judge_path(repo: Path, config: ExperimentConfig) -> Path:
    return repo / "experiments" / "runs" / config.experiment_name / "judgments.jsonl"


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def _judge_one(
    client: GatewayClient,
    *,
    case: Case,
    generation: ExperimentResult,
    judge_model: str,
) -> JudgeResult:
    response = await client.complete(
        model=judge_model,
        system=JUDGE_SYSTEM,
        user=judge_prompt(case, generation.parsed_answer),
        max_tokens=256,
        temperature=0.0,
    )
    payload = _parse_json(response.text)
    return JudgeResult(
        run_id=generation.run_id,
        case_id=generation.case_id,
        architecture=generation.architecture,
        nominal_budget=generation.nominal_budget,
        judge_model=judge_model,
        correct=bool(payload["correct"]),
        evidence_score=max(0, min(4, int(payload["evidence_score"]))),
        unsupported_claims=bool(payload["unsupported_claims"]),
        rationale=str(payload["rationale"]),
        raw_response=response.text,
        usage=response.usage,
        latency_seconds=response.latency_seconds,
    )


async def execute_judging_run(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
) -> dict[str, int | float]:
    generations = [
        row
        for row in read_jsonl(generation_path(repo, config), ExperimentResult)
        if row.status == "ok"
    ]
    output_path = judge_path(repo, config)
    previous = read_jsonl(output_path, JudgeResult)
    complete = {(row.run_id, row.judge_model) for row in previous if row.status == "ok"}
    case_map = {case.case_id: case for case in cases}
    jobs = [
        (generation, judge_model)
        for generation in generations
        for judge_model in config.judge_models
        if (generation.run_id, judge_model) not in complete
    ]
    client = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    if not client.configured:
        await client.close()
        raise RuntimeError("gateway is not configured")
    semaphore = asyncio.Semaphore(
        max(config.global_case_concurrency, client.maximum_total_concurrency)
    )
    started = time.monotonic()
    deadline = started + config.judging_runtime_hours * 3600
    counters = {"completed": 0, "failed": 0, "skipped": len(complete), "adjudicated": 0}

    async def worker(generation: ExperimentResult, model: str) -> None:
        if time.monotonic() > deadline - 30:
            return
        async with semaphore:
            try:
                result = await _judge_one(
                    client,
                    case=case_map[generation.case_id],
                    generation=generation,
                    judge_model=model,
                )
                append_jsonl(output_path, result)
                counters["completed"] += 1
            except Exception:  # noqa: BLE001 - isolate and count individual judge failures
                counters["failed"] += 1

    await asyncio.gather(*(worker(*job) for job in jobs))

    all_judgments = read_jsonl(output_path, JudgeResult)
    grouped: dict[str, list[JudgeResult]] = {}
    for row in all_judgments:
        if row.judge_model in config.judge_models and row.status == "ok":
            grouped.setdefault(row.run_id, []).append(row)
    generation_map = {row.run_id: row for row in generations}
    adjudicated = {
        row.run_id for row in all_judgments if row.judge_model == config.adjudicator_model
    }
    disagreements = [
        run_id
        for run_id, rows in grouped.items()
        if len({row.judge_model for row in rows}) == len(config.judge_models)
        and len({row.correct for row in rows}) > 1
        and run_id not in adjudicated
    ]

    async def adjudicate(run_id: str) -> None:
        if time.monotonic() > deadline - 30:
            return
        async with semaphore:
            generation = generation_map[run_id]
            try:
                result = await _judge_one(
                    client,
                    case=case_map[generation.case_id],
                    generation=generation,
                    judge_model=config.adjudicator_model,
                )
                append_jsonl(output_path, result)
                counters["adjudicated"] += 1
            except Exception:  # noqa: BLE001 - isolate and count adjudicator failures
                counters["failed"] += 1

    await asyncio.gather(*(adjudicate(run_id) for run_id in disagreements))
    await client.close()
    counters["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return counters
