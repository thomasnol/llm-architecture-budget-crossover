from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from .gateway import GatewayClient
from .io import append_jsonl, read_jsonl
from .v2_config import V2Config
from .v2_manifest import (
    ensure_judge_sample_manifest,
    ensure_run_manifest,
    record_phase,
)
from .v2_models import V2Case, V2Generation, V2Judgment
from .v2_runner import v2_generation_path

JUDGE_SYSTEM = """You are a strict pointwise evaluator. Compare an operational
decision with the structured gold decision and supplied evidence. Ignore style.
Do not reward an answer that reaches the right label through unsupported claims.
Return only the requested JSON."""


def v2_judgment_path(repo: Path, config: V2Config) -> Path:
    return repo / "experiments" / "runs" / config.experiment_name / "judgments.jsonl"


def _parse(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise TypeError("judge response is not an object")
    return payload


def _strict_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"judge field {key!r} must be a JSON boolean")
    return value


def _prompt(case: V2Case, generation: V2Generation) -> str:
    return f"""TASK
{case.task}

QUESTION
{case.question}

EVIDENCE
{case.context}

GOLD OPERATIONAL DECISION
{json.dumps(case.gold_decision, sort_keys=True)}

CANDIDATE OPERATIONAL DECISION
{json.dumps(generation.parsed_decision, sort_keys=True)}

CANDIDATE RATIONALE
{generation.rationale}

Return ONLY:
{{"semantically_correct": true_or_false,
  "evidence_score": integer_0_to_4,
  "unsupported_claims": true_or_false,
  "rationale": "one concise sentence"}}"""


def select_judging_generations(
    generations: list[V2Generation],
    config: V2Config,
) -> list[V2Generation]:
    """Select a frozen, balanced secondary-judge audit sample.

    Exact structured scoring remains the primary outcome, so judging every
    response would add thousands of calls without adding primary information.
    Sampling is balanced by dataset and system and deterministic from the
    preregistered seed.
    """

    limit = config.judge_sample_per_system_dataset
    if limit is None:
        return sorted(generations, key=lambda row: row.run_id)
    groups: dict[tuple[str, str], list[V2Generation]] = {}
    for generation in generations:
        groups.setdefault((generation.dataset, generation.system), []).append(generation)
    selected: list[V2Generation] = []
    for (dataset, system), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row.run_id)
        digest = hashlib.sha256(f"{config.seed}:{dataset}:{system}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        if len(rows) > limit:
            rows = rng.sample(rows, limit)
        selected.extend(rows)
    return sorted(selected, key=lambda row: row.run_id)


async def execute_v2_judging(
    *,
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
) -> dict[str, int | float]:
    ensure_run_manifest(repo, config, cases)
    all_generations = [
        row
        for row in read_jsonl(v2_generation_path(repo, config), V2Generation)
        if row.status == "ok"
    ]
    expected = {
        (case.case_id, system, 0)
        for case in cases
        for system in config.systems
    }
    observed = [
        (row.case_id, row.system, row.replicate) for row in all_generations
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError(
            "generation grid must be complete and duplicate-free before freezing "
            "the secondary-judge sample"
        )
    generations = select_judging_generations(all_generations, config)
    ensure_judge_sample_manifest(
        repo=repo,
        config=config,
        selected_run_ids=[generation.run_id for generation in generations],
    )
    output = v2_judgment_path(repo, config)
    previous = read_jsonl(output, V2Judgment)
    complete = {
        (row.run_id, row.judge_model) for row in previous if row.status == "ok"
    }
    jobs = [
        (generation, model)
        for generation in generations
        for model in config.judge_models
        if (generation.run_id, model) not in complete
    ]
    case_map = {case.case_id: case for case in cases}
    client = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    if not client.configured:
        await client.close()
        raise RuntimeError("gateway is not configured")
    semaphore = asyncio.Semaphore(
        max(config.global_case_concurrency, client.maximum_total_concurrency)
    )
    started = time.monotonic()
    deadline = started + config.judging_runtime_hours * 3600
    counters: dict[str, int | float] = {
        "completed": 0,
        "failed": 0,
        "skipped": len(complete),
        "scheduled": len(jobs),
    }

    async def worker(generation: V2Generation, judge_model: str) -> None:
        if time.monotonic() > deadline - 30:
            return
        async with semaphore:
            if time.monotonic() > deadline - 30:
                return
            call_started = time.monotonic()
            try:
                response = await client.complete(
                    model=judge_model,
                    system=JUDGE_SYSTEM,
                    user=_prompt(case_map[generation.case_id], generation),
                    max_tokens=256,
                    temperature=0.0,
                )
                payload = _parse(response.text)
                append_jsonl(
                    output,
                    V2Judgment(
                        run_id=generation.run_id,
                        case_id=generation.case_id,
                        judge_model=judge_model,
                        semantically_correct=_strict_bool(
                            payload, "semantically_correct"
                        ),
                        evidence_score=int(payload["evidence_score"]),
                        unsupported_claims=_strict_bool(
                            payload, "unsupported_claims"
                        ),
                        rationale=str(payload["rationale"]),
                        raw_response=response.text,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        latency_seconds=response.latency_seconds,
                    ),
                )
                counters["completed"] = int(counters["completed"]) + 1
            except Exception:  # noqa: BLE001
                append_jsonl(
                    output,
                    V2Judgment(
                        run_id=generation.run_id,
                        case_id=generation.case_id,
                        judge_model=judge_model,
                        semantically_correct=False,
                        evidence_score=0,
                        unsupported_claims=True,
                        rationale="",
                        raw_response="",
                        latency_seconds=time.monotonic() - call_started,
                        status="error",
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
    counters["sampled_generations"] = len(generations)
    counters["available_generations"] = len(all_generations)
    record_phase(repo, config, phase="judging", counters=counters)
    return counters
