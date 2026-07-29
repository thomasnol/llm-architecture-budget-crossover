from __future__ import annotations

import asyncio
import json
import re

from .gateway import GatewayClient
from .models import CallRecord, Case, ExperimentResult
from .prompts import (
    ANALYST_SYSTEM,
    critique_prompt,
    debate_critic_prompt,
    debate_final_prompt,
    direct_prompt,
    draft_prompt,
    revision_prompt,
    specialist_prompt,
)


def allocate_budget(total: int, weights: list[float], minimum: int = 64) -> list[int]:
    if total < minimum * len(weights):
        raise ValueError(f"budget {total} is below {minimum * len(weights)}")
    remaining = total - minimum * len(weights)
    weight_sum = sum(weights)
    raw = [remaining * weight / weight_sum for weight in weights]
    extras = [int(value) for value in raw]
    remainder = remaining - sum(extras)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - extras[i], reverse=True)
    for index in order[:remainder]:
        extras[index] += 1
    allocation = [minimum + extra for extra in extras]
    assert sum(allocation) == total
    return allocation


def extract_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "answer" in payload:
            return str(payload["answer"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except json.JSONDecodeError:
            return match.group(1).strip()
    return cleaned


async def _call(
    client: GatewayClient,
    *,
    model: str,
    user: str,
    token_cap: int,
    temperature: float,
    stage: str,
) -> CallRecord:
    response = await client.complete(
        model=model,
        system=ANALYST_SYSTEM,
        user=user,
        max_tokens=token_cap,
        temperature=temperature,
    )
    return CallRecord(stage=stage, token_cap=token_cap, response=response)


async def run_architecture(
    client: GatewayClient,
    *,
    case: Case,
    architecture: str,
    budget: int,
    model: str,
    temperature: float,
    run_id: str,
) -> ExperimentResult:
    calls: list[CallRecord] = []
    if architecture == "direct":
        final = await _call(
            client,
            model=model,
            user=direct_prompt(case),
            token_cap=budget,
            temperature=temperature,
            stage="direct",
        )
        calls.append(final)
    elif architecture == "self_critique":
        draft_cap, critique_cap, final_cap = allocate_budget(budget, [0.44, 0.20, 0.36])
        draft = await _call(
            client,
            model=model,
            user=draft_prompt(case),
            token_cap=draft_cap,
            temperature=temperature,
            stage="draft",
        )
        calls.append(draft)
        critique = await _call(
            client,
            model=model,
            user=critique_prompt(case, draft.response.text),
            token_cap=critique_cap,
            temperature=temperature,
            stage="critique",
        )
        calls.append(critique)
        final = await _call(
            client,
            model=model,
            user=revision_prompt(case, draft.response.text, critique.response.text),
            token_cap=final_cap,
            temperature=temperature,
            stage="revision",
        )
        calls.append(final)
    elif architecture == "debate":
        first_cap, second_cap, critic_cap, final_cap = allocate_budget(
            budget, [0.25, 0.25, 0.20, 0.30]
        )
        first_task = _call(
            client,
            model=model,
            user=specialist_prompt(case, "classification"),
            token_cap=first_cap,
            temperature=temperature,
            stage="classification_specialist",
        )
        second_task = _call(
            client,
            model=model,
            user=specialist_prompt(case, "rules"),
            token_cap=second_cap,
            temperature=temperature,
            stage="rules_specialist",
        )
        first, second = await asyncio.gather(first_task, second_task)
        calls.extend([first, second])
        critic = await _call(
            client,
            model=model,
            user=debate_critic_prompt(case, first.response.text, second.response.text),
            token_cap=critic_cap,
            temperature=temperature,
            stage="reviewer",
        )
        calls.append(critic)
        final = await _call(
            client,
            model=model,
            user=debate_final_prompt(
                case, first.response.text, second.response.text, critic.response.text
            ),
            token_cap=final_cap,
            temperature=temperature,
            stage="synthesis",
        )
        calls.append(final)
    else:
        raise ValueError(f"unsupported architecture {architecture}")

    return ExperimentResult(
        run_id=run_id,
        case_id=case.case_id,
        task=case.task,
        architecture=architecture,
        nominal_budget=budget,
        model=model,
        answer_text=final.response.text,
        parsed_answer=extract_answer(final.response.text),
        calls=calls,
    )
