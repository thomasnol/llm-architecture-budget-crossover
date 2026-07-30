from __future__ import annotations

"""Budget-constrained orchestration systems."""

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .gateway import GatewayClient
from .models import CallRecord, GatewayResponse
from .policy import POLICY_TEXT, canonical_decision
from .records import Case, Generation

PROMPT_REVISION = "hmda-budget-2026-07-30"

DECISION_SYSTEM = """You are operating a research mortgage-adjudication sandbox.
Follow the supplied research policy exactly. Do not import real lender rules.
Never use protected or neighborhood monitoring fields to make a credit decision.
Return concise JSON, not chain-of-thought."""

SPECIALIST_SYSTEM = """You are one specialist inside a research mortgage
adjudication workflow. Analyze only your assigned evidence against the supplied
research policy. Return concise JSON findings for a chair; do not invent missing
facts and do not reveal chain-of-thought."""

COMPLIANCE_SYSTEM = """You are an independent compliance guardrail in a research
mortgage-adjudication sandbox. Check whether a draft follows the supplied policy,
uses only permitted financial fields, and remains invariant to monitoring-only
demographics. Return concise JSON audit findings."""

OUTPUT_INSTRUCTION = """Return ONLY valid JSON:
{"decision":"approve|conditional_review|deny|manual_review",
 "reason_codes":["allowed_policy_reason_code", "..."],
 "confidence":0.0,
 "rationale":"one concise evidence-grounded sentence"}"""


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class CallRequest:
    stage: str
    model: str
    system: str
    user: str
    temperature: float
    desired_output_tokens: int


def _payload(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def parse_answer(text: str) -> tuple[dict[str, Any] | None, float | None, str]:
    payload = _payload(text)
    if payload is None:
        return None, None, ""
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and not 0 <= confidence <= 1:
        confidence = None
    rationale = str(payload.get("rationale", "")).strip()
    return canonical_decision(payload), confidence, rationale


class BudgetedCaller:
    def __init__(
        self,
        client: GatewayClient,
        *,
        config: ExperimentConfig,
        token_budget: int,
    ) -> None:
        self.client = client
        self.config = config
        self.token_budget = token_budget
        self.calls: list[CallRecord] = []
        self.accounting: list[dict[str, Any]] = []
        self.accounted_tokens = 0
        self.overrun = False

    def estimate_prompt_tokens(self, system: str, user: str) -> int:
        characters = len(system) + len(user)
        return math.ceil(characters / self.config.prompt_chars_per_token) + (
            self.config.prompt_token_overhead
        )

    def _observed_total(self, response: GatewayResponse, prompt_estimate: int) -> int:
        if response.usage.total_tokens is not None:
            return int(response.usage.total_tokens)
        if (
            response.usage.prompt_tokens is not None
            and response.usage.completion_tokens is not None
        ):
            return int(response.usage.prompt_tokens + response.usage.completion_tokens)
        completion_estimate = math.ceil(len(response.text) / self.config.prompt_chars_per_token)
        return prompt_estimate + completion_estimate

    def _allocation(self, request: CallRequest, *, remaining: int) -> tuple[int, int]:
        prompt_estimate = self.estimate_prompt_tokens(request.system, request.user)
        output_tokens = min(
            request.desired_output_tokens,
            remaining - prompt_estimate,
        )
        if output_tokens < self.config.minimum_call_output_tokens:
            raise BudgetExhausted(
                f"{request.stage} needs at least {prompt_estimate + self.config.minimum_call_output_tokens} "
                f"estimated tokens with {remaining} remaining"
            )
        return prompt_estimate, output_tokens

    async def call(self, request: CallRequest) -> CallRecord:
        remaining = self.token_budget - self.accounted_tokens
        prompt_estimate, output_tokens = self._allocation(request, remaining=remaining)
        response = await self.client.complete(
            model=request.model,
            system=request.system,
            user=request.user,
            max_tokens=output_tokens,
            temperature=request.temperature,
            stage=request.stage,
        )
        record = CallRecord(
            stage=request.stage,
            token_cap=output_tokens,
            response=response,
        )
        observed = self._observed_total(response, prompt_estimate)
        self.calls.append(record)
        self.accounted_tokens += observed
        self.overrun = self.overrun or self.accounted_tokens > self.token_budget
        self.accounting.append(
            {
                "stage": request.stage,
                "estimated_prompt_tokens": prompt_estimate,
                "allocated_output_tokens": output_tokens,
                "accounted_total_tokens": observed,
            }
        )
        return record

    async def call_parallel(self, requests: list[CallRequest]) -> list[CallRecord]:
        if not requests:
            return []
        remaining = self.token_budget - self.accounted_tokens
        prompt_estimates = [
            self.estimate_prompt_tokens(request.system, request.user) for request in requests
        ]
        minimum = sum(prompt_estimates) + len(requests) * (self.config.minimum_call_output_tokens)
        if minimum > remaining:
            raise BudgetExhausted(
                f"parallel specialists need at least {minimum} estimated tokens "
                f"with {remaining} remaining"
            )
        output_pool = remaining - sum(prompt_estimates)
        equal_share = output_pool // len(requests)
        allocations = [min(request.desired_output_tokens, equal_share) for request in requests]
        if any(value < self.config.minimum_call_output_tokens for value in allocations):
            raise BudgetExhausted("parallel output allocation fell below the minimum")
        responses = await asyncio.gather(
            *[
                self.client.complete(
                    model=request.model,
                    system=request.system,
                    user=request.user,
                    max_tokens=allocation,
                    temperature=request.temperature,
                    stage=request.stage,
                )
                for request, allocation in zip(requests, allocations, strict=True)
            ]
        )
        records: list[CallRecord] = []
        for request, allocation, prompt_estimate, response in zip(
            requests, allocations, prompt_estimates, responses, strict=True
        ):
            record = CallRecord(
                stage=request.stage,
                token_cap=allocation,
                response=response,
            )
            observed = self._observed_total(response, prompt_estimate)
            records.append(record)
            self.calls.append(record)
            self.accounted_tokens += observed
            self.accounting.append(
                {
                    "stage": request.stage,
                    "estimated_prompt_tokens": prompt_estimate,
                    "allocated_output_tokens": allocation,
                    "accounted_total_tokens": observed,
                    "parallel": True,
                }
            )
        self.overrun = self.overrun or self.accounted_tokens > self.token_budget
        return records


def _documents(case: Case, names: list[str]) -> str:
    return "\n\n".join(case.documents[name] for name in names)


def _final_prompt(evidence: str) -> str:
    return f"""{POLICY_TEXT}

CASE EVIDENCE
{evidence}

Apply the policy in its stated precedence order. Monitoring-only fields may be
audited but may not affect the credit decision.

{OUTPUT_INSTRUCTION}"""


def _call_request(
    stage: str,
    *,
    user: str,
    config: ExperimentConfig,
    model: str | None = None,
    system: str = DECISION_SYSTEM,
    specialist: bool = False,
) -> CallRequest:
    return CallRequest(
        stage=stage,
        model=model or config.generator_model,
        system=system,
        user=user,
        temperature=(config.specialist_temperature if specialist else config.direct_temperature),
        desired_output_tokens=config.stage_max_output_tokens,
    )


def _finish(
    *,
    case: Case,
    system: str,
    token_budget: int,
    config: ExperimentConfig,
    caller: BudgetedCaller,
    run_id: str,
    repetition: int,
    final_text: str,
    diagnostics: dict[str, Any],
    status: str = "ok",
    error: str | None = None,
) -> Generation:
    parsed, confidence, rationale = parse_answer(final_text)
    return Generation(
        run_id=run_id,
        case_id=case.case_id,
        pair_id=case.pair_id,
        counterfactual_variant=case.counterfactual_variant,
        system=system,
        token_budget=token_budget,
        repetition=repetition,
        model=config.generator_model,
        supervisor_model=config.supervisor_model,
        answer_text=final_text,
        parsed_decision=parsed,
        confidence=confidence,
        rationale=rationale,
        calls=caller.calls,
        diagnostics={
            **diagnostics,
            "budget_accounting": caller.accounting,
            "accounted_tokens": caller.accounted_tokens,
            "budget_overrun": caller.overrun,
            "schema_valid": parsed is not None,
        },
        status=status,
        error=error,
    )


async def run_system(
    client: GatewayClient,
    *,
    case: Case,
    system: str,
    token_budget: int,
    config: ExperimentConfig,
    run_id: str,
    repetition: int = 0,
) -> Generation:
    caller = BudgetedCaller(client, config=config, token_budget=token_budget)
    diagnostics: dict[str, Any] = {"escalated": False}
    final_text = ""
    try:
        if system in {"monolith", "always_primary", "always_supervisor"}:
            evidence = _documents(
                case,
                [
                    "application",
                    "collateral",
                    "credit",
                    "compliance_monitoring",
                    "quality_control",
                ],
            )
            result = await caller.call(
                _call_request(
                    system,
                    user=_final_prompt(evidence),
                    config=config,
                    model=(
                        config.supervisor_model
                        if system == "always_supervisor"
                        else config.generator_model
                    ),
                )
            )
            final_text = result.response.text

        elif system == "retrieval":
            index = """AVAILABLE EVIDENCE TOOLS
- application: requested product, amount, income, term, occupancy, units
- collateral: property value, LTV, construction, lien, conforming status
- credit: DTI and public credit/AUS metadata
- compliance_monitoring: protected and neighborhood monitoring fields
- quality_control: missing-value and leakage handling notes"""
            planner = await caller.call(
                _call_request(
                    "retrieval_plan",
                    user=f"""{POLICY_TEXT}

{index}

Choose at most three evidence tools needed to adjudicate the case. Return ONLY:
{{"request":["tool_name", "..."],"rationale":"brief"}}""",
                    config=config,
                )
            )
            plan = _payload(planner.response.text) or {}
            requested = plan.get("request")
            allowed = list(case.documents)
            selected = (
                [name for name in requested if name in allowed][:3]
                if isinstance(requested, list)
                else []
            )
            if not selected:
                selected = ["application"]
            diagnostics["retrieved_documents"] = selected
            result = await caller.call(
                _call_request(
                    "retrieval_decision",
                    user=_final_prompt(_documents(case, selected)),
                    config=config,
                )
            )
            final_text = result.response.text

        elif system == "committee":
            specialist_prompts = [
                (
                    "capacity_specialist",
                    f"""{POLICY_TEXT}

ASSIGNED EVIDENCE
{_documents(case, ["application", "credit"])}

Return ONLY:
{{"recommended_decision":"approve|conditional_review|deny|manual_review",
  "reason_codes":["..."],"material_facts":["brief fact"]}}""",
                ),
                (
                    "collateral_specialist",
                    f"""{POLICY_TEXT}

ASSIGNED EVIDENCE
{case.documents["collateral"]}

Return ONLY:
{{"recommended_decision":"approve|conditional_review|deny|manual_review",
  "reason_codes":["..."],"material_facts":["brief fact"]}}""",
                ),
                (
                    "compliance_specialist",
                    f"""{POLICY_TEXT}

ASSIGNED MONITORING EVIDENCE
{_documents(case, ["compliance_monitoring", "quality_control"])}

Return ONLY:
{{"prohibited_for_decision":["field_name"],"data_quality_flags":["brief flag"],
  "instruction":"brief constraint for the chair"}}""",
                ),
            ]
            records = await caller.call_parallel(
                [
                    _call_request(
                        stage,
                        user=prompt,
                        config=config,
                        system=(
                            COMPLIANCE_SYSTEM
                            if stage == "compliance_specialist"
                            else SPECIALIST_SYSTEM
                        ),
                        model=config.generator_model,
                        specialist=True,
                    )
                    for stage, prompt in specialist_prompts
                ]
            )
            summaries = "\n\n".join(
                f"{record.stage.upper()}\n{record.response.text}" for record in records
            )
            chair = await caller.call(
                _call_request(
                    "committee_chair",
                    user=f"""{POLICY_TEXT}

SPECIALIST REPORTS
{summaries}

Resolve reports by policy precedence, not majority vote. Do not use monitoring
attributes as credit factors.

{OUTPUT_INSTRUCTION}""",
                    config=config,
                    model=config.generator_model,
                )
            )
            final_text = chair.response.text

        elif system in {"guardrail", "adaptive", "selective_supervisor"}:
            financial = _documents(case, ["application", "collateral", "credit", "quality_control"])
            draft = await caller.call(
                _call_request(
                    "financial_underwriter",
                    user=_final_prompt(financial),
                    config=config,
                )
            )
            final_text = draft.response.text
            draft_decision, draft_confidence, _ = parse_answer(final_text)
            diagnostics["initial_decision"] = draft_decision
            diagnostics["initial_confidence"] = draft_confidence
            missing_signal = "not reported" in financial.lower()
            should_escalate = system == "guardrail"
            if system in {"adaptive", "selective_supervisor"}:
                should_escalate = (
                    draft_decision is None
                    or draft_confidence is None
                    or draft_confidence < config.adaptive_confidence_threshold
                    or draft_decision.get("decision") != "approve"
                    or missing_signal
                )
            diagnostics["route_signals"] = {
                "invalid_draft": draft_decision is None,
                "confidence": draft_confidence,
                "non_approve_draft": (
                    draft_decision is not None and draft_decision.get("decision") != "approve"
                ),
                "missing_evidence_marker": missing_signal,
            }
            if should_escalate:
                diagnostics["escalated"] = True
                audit = await caller.call(
                    _call_request(
                        "compliance_audit",
                        user=f"""{POLICY_TEXT}

UNDERWRITER DRAFT
{draft.response.text}

SEGREGATED MONITORING EVIDENCE
{case.documents["compliance_monitoring"]}

Check policy accuracy and whether any protected or neighborhood field affected the
draft. Return ONLY:
{{"accept":true_or_false,"policy_errors":["brief error"],
  "prohibited_field_use":true_or_false,"required_correction":"brief correction"}}""",
                        config=config,
                        model=(
                            config.supervisor_model
                            if system == "selective_supervisor"
                            else config.generator_model
                        ),
                        system=COMPLIANCE_SYSTEM,
                    )
                )
                diagnostics["audit"] = _payload(audit.response.text)
                revision = await caller.call(
                    _call_request(
                        "guarded_final",
                        user=f"""{POLICY_TEXT}

FINANCIAL UNDERWRITER DRAFT
{draft.response.text}

INDEPENDENT COMPLIANCE AUDIT
{audit.response.text}

Apply supported corrections. Monitoring-only attributes must not affect the
decision.

{OUTPUT_INSTRUCTION}""",
                        config=config,
                    )
                )
                final_text = revision.response.text
        else:
            raise ValueError(f"unsupported system: {system}")
    except BudgetExhausted as exc:
        diagnostics["budget_exhausted"] = True
        return _finish(
            case=case,
            system=system,
            token_budget=token_budget,
            config=config,
            caller=caller,
            run_id=run_id,
            repetition=repetition,
            final_text=final_text,
            diagnostics=diagnostics,
            status="budget_exhausted",
            error=str(exc),
        )

    return _finish(
        case=case,
        system=system,
        token_budget=token_budget,
        config=config,
        caller=caller,
        run_id=run_id,
        repetition=repetition,
        final_text=final_text,
        diagnostics=diagnostics,
    )
