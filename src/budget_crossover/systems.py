from __future__ import annotations

"""Canonical budget-constrained answer systems."""

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from .budget import BudgetExceeded, BudgetLedger, BudgetTier
from .checking import check_candidate
from .models import (
    CallEvent,
    Candidate,
    CellResult,
    CheckResult,
    EvidenceItem,
    FrozenModel,
    GatewayResponse,
    MechanismTrace,
    PublicCase,
)
from .retrieval import (
    RetrievalResult,
    retrieval_input_hash,
    retrieval_query_hash,
    retrieve,
)
from .scoring import normalized_candidate_value

CORE_INSTRUCTIONS = """Follow the stage-specific instructions exactly using only supplied public
case information and evidence. Do not report confidence, inspect hidden labels, or invent facts.
Return only the structured JSON requested by the current stage, without chain-of-thought."""
PROMPT_REVISION = "conditional-crossover-2026-08-03"

CANDIDATE_SCHEMA = """{
  "value": "strict numeric string",
  "unit": "string or null",
  "scale": "ones|thousand|million|billion|percent",
  "entity": "string or null",
  "period": "string or null",
  "expression": "arithmetic expression, count(\"evidence-id\", ...), or null",
  "citations": ["evidence_id"]
}"""


class CompletionClient(Protocol):
    async def count_prompt_tokens(self, *, model: str, system: str, user: str) -> int: ...

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stage: str,
    ) -> GatewayResponse: ...


class Plan(FrozenModel):
    steps: tuple[str, ...]
    queries: tuple[str, ...]


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _empty_retrieval(case: PublicCase, tier: BudgetTier) -> RetrievalResult:
    return RetrievalResult(
        items=(),
        pre_truncation_ids=(),
        post_truncation_ids=(),
        tier_id=tier.name,
        requested_k=tier.retrieval_limit,
        query_hash=retrieval_query_hash(()),
        input_hash=retrieval_input_hash(case),
    )


def _evidence_text(items: Sequence[EvidenceItem]) -> str:
    blocks = []
    for item in items:
        metadata = {
            "evidence_id": item.evidence_id,
            "kind": item.kind,
            "headers": item.headers,
            "row_label": item.row_label,
            "unit": item.unit,
            "scale": item.scale,
            "entity": item.entity,
            "period": item.period,
        }
        blocks.append(
            f"EVIDENCE {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n{item.text}"
        )
    return "\n\n".join(blocks)


def _answer_prompt(case: PublicCase, evidence: Sequence[EvidenceItem]) -> str:
    return f"""QUESTION
{case.question}

EVIDENCE
{_evidence_text(evidence)}

CANDIDATE SCHEMA
{CANDIDATE_SCHEMA}

Return only the candidate JSON object."""


def _planner_prompt(case: PublicCase, query_limit: int) -> str:
    return f"""QUESTION
{case.question}

Decompose the question and propose at most {query_limit} deterministic evidence searches.
Return only JSON with this schema:
{{"steps": ["decomposition step"], "queries": ["search query"]}}"""


def _repair_prompt(
    case: PublicCase,
    evidence: Sequence[EvidenceItem],
    candidate: Candidate,
    check: CheckResult,
) -> str:
    findings = [
        finding.model_dump(mode="json") for finding in check.findings
    ]
    return f"""QUESTION
{case.question}

EVIDENCE
{_evidence_text(evidence)}

REJECTED CANDIDATE
{candidate.model_dump_json()}

CHECKER FINDINGS
{json.dumps(findings, ensure_ascii=False, sort_keys=True)}

CANDIDATE SCHEMA
{CANDIDATE_SCHEMA}

Repair the rejected candidate using only the checker findings as correction feedback.
Return only the repaired candidate JSON object."""


def _answer_changed(before: Candidate, after: Candidate) -> bool:
    before_normalized = normalized_candidate_value(before)
    after_normalized = normalized_candidate_value(after)
    if before_normalized is not None and after_normalized is not None:
        return before_normalized != after_normalized
    return before.value != after.value or before.unit != after.unit or before.scale != after.scale


def _plurality_key(candidate: Candidate) -> tuple[object, ...]:
    normalized = normalized_candidate_value(candidate)
    value_key: object = normalized if normalized is not None else (
        candidate.value.strip(),
        candidate.unit,
        candidate.scale,
    )
    return (
        value_key,
        " ".join((candidate.entity or "").split()).casefold(),
        " ".join((candidate.period or "").split()).casefold(),
    )


def _parse_candidate(text: str) -> Candidate | None:
    try:
        payload = json.loads(text)
        return Candidate.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def _parse_plan(text: str, query_limit: int) -> Plan | None:
    try:
        payload = json.loads(text)
        parsed = Plan.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None
    steps = tuple(step.strip() for step in parsed.steps if step.strip())
    queries = tuple(dict.fromkeys(query.strip() for query in parsed.queries if query.strip()))[
        :query_limit
    ]
    if not steps or not queries:
        return None
    return Plan(steps=steps, queries=queries)


async def _complete(
    client: CompletionClient,
    *,
    ledger: BudgetLedger,
    model: str,
    stage: str,
    user: str,
    max_tokens: int,
) -> tuple[GatewayResponse, CallEvent]:
    prompt_tokens = await client.count_prompt_tokens(
        model=model,
        system=CORE_INSTRUCTIONS,
        user=user,
    )
    reservation = ledger.authorize(
        stage=stage,
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_tokens,
    )
    response = await client.complete(
        model=model,
        system=CORE_INSTRUCTIONS,
        user=user,
        max_tokens=max_tokens,
        stage=stage,
    )
    event = ledger.commit(reservation, response.usage).model_copy(
        update={
            "model": response.model,
            "request_id": response.request_id,
        }
    )
    return response, event


def _trace(
    *,
    planned_queries: tuple[str, ...],
    actual_queries: tuple[str, ...],
    retrieval: RetrievalResult,
    ledger: BudgetLedger,
    events: tuple[CallEvent, ...],
    candidate_count: int,
    checks: tuple[CheckResult, ...] = (),
    repair_attempted: bool = False,
    accepted_candidate_index: int | None,
    answer_changed: bool | None = None,
    exit_reason: str,
) -> MechanismTrace:
    return MechanismTrace(
        planned_queries=planned_queries,
        actual_queries=actual_queries,
        query_hashes=tuple(_query_hash(query) for query in actual_queries),
        retrieval_pre_truncation_ids=retrieval.pre_truncation_ids,
        retrieval_post_truncation_ids=retrieval.post_truncation_ids,
        candidate_token_cap=256,
        candidate_count=candidate_count,
        checks=checks,
        repair_attempted=repair_attempted,
        accepted_candidate_index=accepted_candidate_index,
        answer_changed=answer_changed,
        call_events=events,
        realized_tokens=ledger.spent_tokens,
        exit_reason=exit_reason,
    )


def _budget_exhausted_result(
    *,
    case: PublicCase,
    system: str,
    tier: BudgetTier,
    repetition: int,
    planned_queries: tuple[str, ...],
    actual_queries: tuple[str, ...],
    retrieval: RetrievalResult,
    ledger: BudgetLedger,
    events: tuple[CallEvent, ...] = (),
    candidate_count: int = 0,
    checks: tuple[CheckResult, ...] = (),
) -> CellResult:
    return CellResult(
        case_id=case.case_id,
        system=system,
        tier=tier.name,
        repetition=repetition,
        status="architecture_failure",
        candidate=None,
        trace=_trace(
            planned_queries=planned_queries,
            actual_queries=actual_queries,
            retrieval=retrieval,
            ledger=ledger,
            events=events,
            candidate_count=candidate_count,
            checks=checks,
            accepted_candidate_index=None,
            exit_reason="budget_exhausted",
        ),
    )


async def run_system(
    client: CompletionClient,
    *,
    case: PublicCase,
    system: str,
    tier: BudgetTier,
    model: str,
    repetition: int = 0,
) -> CellResult:
    if system not in {"monolith", "verified_search", "unverified_search"}:
        raise ValueError(f"unsupported system: {system}")

    ledger = BudgetLedger(tier)
    events: list[CallEvent] = []
    if system == "monolith":
        queries = (case.question,)
        retrieval = retrieve(
            case,
            queries,
            limit=tier.retrieval_limit,
            max_chars_per_item=4000,
            tier_id=tier.name,
        )
        try:
            response, event = await _complete(
                client,
                ledger=ledger,
                model=model,
                stage="answer",
                user=_answer_prompt(case, retrieval.items),
                max_tokens=256,
            )
        except BudgetExceeded:
            return _budget_exhausted_result(
                case=case,
                system=system,
                tier=tier,
                repetition=repetition,
                planned_queries=queries,
                actual_queries=queries,
                retrieval=retrieval,
                ledger=ledger,
            )
        candidate = _parse_candidate(response.text)
        status = "ok" if candidate is not None else "architecture_failure"
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status=status,
            candidate=candidate,
            trace=_trace(
                planned_queries=queries,
                actual_queries=queries,
                retrieval=retrieval,
                ledger=ledger,
                events=(event,),
                candidate_count=1,
                accepted_candidate_index=0 if candidate is not None else None,
                exit_reason="completed" if candidate is not None else "invalid_output",
            ),
        )

    try:
        planner_response, planner_event = await _complete(
            client,
            ledger=ledger,
            model=model,
            stage="planner",
            user=_planner_prompt(case, tier.planned_query_limit),
            max_tokens=128,
        )
    except BudgetExceeded:
        return _budget_exhausted_result(
            case=case,
            system=system,
            tier=tier,
            repetition=repetition,
            planned_queries=(),
            actual_queries=(),
            retrieval=_empty_retrieval(case, tier),
            ledger=ledger,
        )
    events.append(planner_event)
    plan = _parse_plan(planner_response.text, tier.planned_query_limit)
    if plan is None:
        empty_retrieval = _empty_retrieval(case, tier)
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status="architecture_failure",
            candidate=None,
            trace=_trace(
                planned_queries=(),
                actual_queries=(),
                retrieval=empty_retrieval,
                ledger=ledger,
                events=tuple(events),
                candidate_count=0,
                accepted_candidate_index=None,
                exit_reason="planner_invalid",
            ),
        )

    queries = plan.queries
    retrieval = retrieve(
        case,
        queries,
        limit=tier.retrieval_limit,
        max_chars_per_item=4000,
        tier_id=tier.name,
    )
    if system == "unverified_search":
        candidates: list[Candidate] = []
        candidate_indices: list[int] = []
        candidate_attempts = 0
        for index in range(tier.candidate_limit):
            try:
                response, event = await _complete(
                    client,
                    ledger=ledger,
                    model=model,
                    stage=f"candidate_{index}",
                    user=_answer_prompt(case, retrieval.items),
                    max_tokens=256,
                )
            except BudgetExceeded:
                return CellResult(
                    case_id=case.case_id,
                    system=system,
                    tier=tier.name,
                    repetition=repetition,
                    status="architecture_failure",
                    candidate=None,
                    trace=_trace(
                        planned_queries=queries,
                        actual_queries=queries,
                        retrieval=retrieval,
                        ledger=ledger,
                        events=tuple(events),
                        candidate_count=candidate_attempts,
                        accepted_candidate_index=None,
                        exit_reason="budget_exhausted",
                    ),
                )
            events.append(event)
            candidate_attempts += 1
            candidate = _parse_candidate(response.text)
            if candidate is not None:
                candidates.append(candidate)
                candidate_indices.append(index)

        if not candidates:
            return CellResult(
                case_id=case.case_id,
                system=system,
                tier=tier.name,
                repetition=repetition,
                status="architecture_failure",
                candidate=None,
                trace=_trace(
                    planned_queries=queries,
                    actual_queries=queries,
                    retrieval=retrieval,
                    ledger=ledger,
                    events=tuple(events),
                    candidate_count=candidate_attempts,
                    accepted_candidate_index=None,
                    exit_reason="invalid_output",
                ),
            )

        counts: dict[tuple[object, ...], int] = {}
        for candidate in candidates:
            key = _plurality_key(candidate)
            counts[key] = counts.get(key, 0) + 1
        winner_position = max(
            range(len(candidates)),
            key=lambda position: counts[_plurality_key(candidates[position])],
        )
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status="ok",
            candidate=candidates[winner_position],
            trace=_trace(
                planned_queries=queries,
                actual_queries=queries,
                retrieval=retrieval,
                ledger=ledger,
                events=tuple(events),
                candidate_count=candidate_attempts,
                accepted_candidate_index=candidate_indices[winner_position],
                exit_reason="plurality_selected",
            ),
        )

    candidates: list[Candidate] = []
    checks: list[CheckResult] = []
    candidate_attempts = 0
    for index in range(tier.candidate_limit):
        try:
            response, event = await _complete(
                client,
                ledger=ledger,
                model=model,
                stage=f"candidate_{index}",
                user=_answer_prompt(case, retrieval.items),
                max_tokens=256,
            )
        except BudgetExceeded:
            return CellResult(
                case_id=case.case_id,
                system=system,
                tier=tier.name,
                repetition=repetition,
                status="architecture_failure",
                candidate=None,
                trace=_trace(
                    planned_queries=queries,
                    actual_queries=queries,
                    retrieval=retrieval,
                    ledger=ledger,
                    events=tuple(events),
                    candidate_count=candidate_attempts,
                    checks=tuple(checks),
                    accepted_candidate_index=None,
                    exit_reason="budget_exhausted",
                ),
            )
        events.append(event)
        candidate_attempts += 1
        candidate = _parse_candidate(response.text)
        if candidate is None:
            continue
        candidates.append(candidate)
        check = check_candidate(candidate, retrieval.items)
        checks.append(check)
        if check.passed:
            return CellResult(
                case_id=case.case_id,
                system=system,
                tier=tier.name,
                repetition=repetition,
                status="ok",
                candidate=candidate,
                trace=_trace(
                    planned_queries=queries,
                    actual_queries=queries,
                    retrieval=retrieval,
                    ledger=ledger,
                    events=tuple(events),
                    candidate_count=candidate_attempts,
                    checks=tuple(checks),
                    accepted_candidate_index=index,
                    exit_reason="accepted",
                ),
            )

    if tier.repair_limit and candidates:
        try:
            repair_response, repair_event = await _complete(
                client,
                ledger=ledger,
                model=model,
                stage="repair",
                user=_repair_prompt(case, retrieval.items, candidates[-1], checks[-1]),
                max_tokens=256,
            )
        except BudgetExceeded:
            return _budget_exhausted_result(
                case=case,
                system=system,
                tier=tier,
                repetition=repetition,
                planned_queries=queries,
                actual_queries=queries,
                retrieval=retrieval,
                ledger=ledger,
                events=tuple(events),
                candidate_count=candidate_attempts,
                checks=tuple(checks),
            )
        events.append(repair_event)
        repaired = _parse_candidate(repair_response.text)
        changed = _answer_changed(candidates[-1], repaired) if repaired is not None else None
        if repaired is not None:
            repair_check = check_candidate(repaired, retrieval.items)
            checks.append(repair_check)
            if repair_check.passed:
                return CellResult(
                    case_id=case.case_id,
                    system=system,
                    tier=tier.name,
                    repetition=repetition,
                    status="ok",
                    candidate=repaired,
                    trace=_trace(
                        planned_queries=queries,
                        actual_queries=queries,
                        retrieval=retrieval,
                        ledger=ledger,
                        events=tuple(events),
                        candidate_count=candidate_attempts,
                        checks=tuple(checks),
                        repair_attempted=True,
                        accepted_candidate_index=candidate_attempts,
                        answer_changed=changed,
                        exit_reason="repair_accepted",
                    ),
                )
        return CellResult(
            case_id=case.case_id,
            system=system,
            tier=tier.name,
            repetition=repetition,
            status="architecture_failure",
            candidate=None,
            trace=_trace(
                planned_queries=queries,
                actual_queries=queries,
                retrieval=retrieval,
                ledger=ledger,
                events=tuple(events),
                candidate_count=candidate_attempts,
                checks=tuple(checks),
                repair_attempted=True,
                accepted_candidate_index=None,
                answer_changed=changed,
                exit_reason="checker_exhausted",
            ),
        )

    return CellResult(
        case_id=case.case_id,
        system=system,
        tier=tier.name,
        repetition=repetition,
        status="architecture_failure",
        candidate=None,
        trace=_trace(
            planned_queries=queries,
            actual_queries=queries,
            retrieval=retrieval,
            ledger=ledger,
            events=tuple(events),
            candidate_count=candidate_attempts,
            checks=tuple(checks),
            accepted_candidate_index=None,
            exit_reason="checker_exhausted" if candidates else "invalid_output",
        ),
    )
