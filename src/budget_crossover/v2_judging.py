from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from .gateway import GatewayClient
from .io import append_jsonl, read_jsonl
from .v2_config import V2Config
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


async def execute_v2_judging(
    *,
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
) -> dict[str, int | float]:
    generations = [
        row
        for row in read_jsonl(v2_generation_path(repo, config), V2Generation)
        if row.status == "ok"
    ]
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
    semaphore = asyncio.Semaphore(config.global_case_concurrency)
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
                        semantically_correct=bool(payload["semantically_correct"]),
                        evidence_score=int(payload["evidence_score"]),
                        unsupported_claims=bool(payload["unsupported_claims"]),
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

    await asyncio.gather(*(worker(*job) for job in jobs))
    await client.close()
    counters["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return counters
