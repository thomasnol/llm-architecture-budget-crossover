from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .gateway import GatewayClient
from .models import CallRecord
from .v2_config import V2Config
from .v2_models import V2Case, V2Generation
from .v2_schema import canonical_decision, parse_response

PROMPT_VERSION = "v2.1.0"

SYSTEM_PROMPT = """You are a careful decision analyst. Use only the supplied case
and evidence. Do not infer missing organization-specific rules. Return the exact
JSON schema requested. The rationale is explanatory only; keep the operational
decision in the dedicated structured fields."""

VERIFIER_SYSTEM = """You are an independent verifier. Check candidate decisions
against the supplied case and evidence. Do not use a majority vote when evidence
can decide the question. Treat missing evidence as uncertainty. Follow the exact
JSON format requested and never reveal chain-of-thought."""


def _case_text(case: V2Case) -> str:
    schema = json.dumps(case.output_schema, ensure_ascii=False, indent=2)
    return f"""DATASET
{case.dataset}

TASK
{case.task}

QUESTION
{case.question}

CONTEXT
{case.context}

REQUIRED JSON SHAPE
{schema}
"""


def _direct_prompt(case: V2Case) -> str:
    return f"""{_case_text(case)}

Produce the operational decision. Return ONLY one valid JSON object with exactly
the requested decision fields plus rationale."""


def _checklist_prompt(case: V2Case) -> str:
    return f"""{_case_text(case)}

Before answering, privately perform a structured audit:
1. identify the requested decision and do not substitute a nearby task;
2. identify the controlling evidence;
3. check negations, exceptions, existing versus recommended products, and every
   relevant numerical unit;
4. consider one plausible counterexample;
5. return the best-supported operational decision.

Return ONLY one valid JSON object with exactly the requested decision fields plus
rationale."""


def _critique_prompt(case: V2Case, candidate: str, *, external: bool) -> str:
    source = "another model" if external else "your earlier draft"
    return f"""{_case_text(case)}

CANDIDATE FROM {source.upper()}
{candidate}

Audit only material decision errors: wrong classification, reversed threshold,
negation, existing-versus-recommended confusion, missing required value, or
unsupported conclusion.

Return ONLY:
{{"accept": true_or_false,
  "confidence": number_0_to_1,
  "error_type": "none|classification|threshold|negation|omission|unsupported|other",
  "feedback": "brief actionable feedback"}}"""


def _revision_prompt(case: V2Case, candidate: str, critique: str) -> str:
    return f"""{_case_text(case)}

INITIAL CANDIDATE
{candidate}

INDEPENDENT AUDIT
{critique}

Revise only when the audit identifies an evidence-supported material error.
Return ONLY one valid JSON object with exactly the requested decision fields plus
rationale."""


def _selection_prompt(case: V2Case, candidates: list[str]) -> str:
    rendered = "\n\n".join(
        f"CANDIDATE {index + 1}\n{candidate}"
        for index, candidate in enumerate(candidates)
    )
    return f"""{_case_text(case)}

{rendered}

Select or reconstruct the decision best supported by the case. Do not decide by
majority alone. Return ONLY one valid JSON object with exactly the requested
decision fields plus rationale."""


def _verification_prompt(case: V2Case, candidate: str) -> str:
    return f"""{_case_text(case)}

INITIAL CANDIDATE
{candidate}

Determine whether this operational decision is supported. Return ONLY:
{{"accept": true_or_false,
  "confidence": number_0_to_1,
  "error_type": "none|classification|threshold|negation|omission|unsupported|other",
  "feedback": "brief explanation"}}"""


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


async def _call(
    client: GatewayClient,
    *,
    model: str,
    user: str,
    max_tokens: int,
    temperature: float,
    stage: str,
    verifier: bool = False,
) -> CallRecord:
    response = await client.complete(
        model=model,
        system=VERIFIER_SYSTEM if verifier else SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return CallRecord(stage=stage, token_cap=max_tokens, response=response)


def _finish(
    *,
    case: V2Case,
    config: V2Config,
    system: str,
    run_id: str,
    calls: list[CallRecord],
    final_text: str,
    diagnostics: dict[str, Any],
) -> V2Generation:
    parsed, rationale = parse_response(final_text)
    canonical = canonical_decision(case.task, parsed)
    return V2Generation(
        run_id=run_id,
        case_id=case.case_id,
        dataset=case.dataset,
        task=case.task,
        system=system,
        generator_model=(
            config.verifier_model if system == "strong_direct" else config.generator_model
        ),
        verifier_model=(
            config.verifier_model
            if system in {"external_verify", "best_of_2", "best_of_4", "adaptive"}
            else None
        ),
        answer_text=final_text,
        parsed_decision=canonical,
        rationale=rationale,
        calls=calls,
        diagnostics={
            **diagnostics,
            "schema_valid": canonical is not None,
            "call_count": len(calls),
        },
    )


async def run_v2_system(
    client: GatewayClient,
    *,
    case: V2Case,
    system: str,
    config: V2Config,
    run_id: str,
) -> V2Generation:
    calls: list[CallRecord] = []
    diagnostics: dict[str, Any] = {"escalated": False}

    if system in {"direct", "checklist", "strong_direct"}:
        call = await _call(
            client,
            model=(
                config.verifier_model
                if system == "strong_direct"
                else config.generator_model
            ),
            user=(
                _direct_prompt(case)
                if system in {"direct", "strong_direct"}
                else _checklist_prompt(case)
            ),
            max_tokens=config.generator_max_tokens,
            temperature=config.direct_temperature,
            stage=system,
        )
        calls.append(call)
        return _finish(
            case=case,
            config=config,
            system=system,
            run_id=run_id,
            calls=calls,
            final_text=call.response.text,
            diagnostics=diagnostics,
        )

    draft = await _call(
        client,
        model=config.generator_model,
        user=_direct_prompt(case),
        max_tokens=config.generator_max_tokens,
        temperature=config.direct_temperature,
        stage="draft",
    )
    calls.append(draft)
    initial, _ = parse_response(draft.response.text)
    diagnostics["initial_decision"] = canonical_decision(case.task, initial)

    if system in {"self_critique", "external_verify"}:
        external = system == "external_verify"
        critic = await _call(
            client,
            model=config.verifier_model if external else config.generator_model,
            user=_critique_prompt(case, draft.response.text, external=external),
            max_tokens=config.critique_max_tokens,
            temperature=0.0,
            stage="external_critique" if external else "self_critique",
            verifier=external,
        )
        calls.append(critic)
        critique_payload = _payload(critic.response.text) or {}
        diagnostics.update(
            {
                "verifier_accept": critique_payload.get("accept"),
                "verifier_confidence": critique_payload.get("confidence"),
                "verifier_error_type": critique_payload.get("error_type"),
            }
        )
        revision = await _call(
            client,
            model=config.generator_model,
            user=_revision_prompt(case, draft.response.text, critic.response.text),
            max_tokens=config.generator_max_tokens,
            temperature=config.direct_temperature,
            stage="revision",
        )
        calls.append(revision)
        return _finish(
            case=case,
            config=config,
            system=system,
            run_id=run_id,
            calls=calls,
            final_text=revision.response.text,
            diagnostics=diagnostics,
        )

    if system in {"best_of_2", "best_of_4"}:
        count = int(system.rsplit("_", 1)[1])
        # The deterministic draft is candidate one; additional samples create
        # genuine proposal diversity without rerunning an identical baseline.
        extra = await asyncio.gather(
            *[
                _call(
                    client,
                    model=config.generator_model,
                    user=_checklist_prompt(case),
                    max_tokens=config.generator_max_tokens,
                    temperature=config.sampling_temperature,
                    stage=f"candidate_{index + 2}",
                )
                for index in range(count - 1)
            ]
        )
        calls.extend(extra)
        candidate_texts = [draft.response.text, *[call.response.text for call in extra]]
        candidate_decisions = [
            canonical_decision(case.task, parse_response(text)[0]) for text in candidate_texts
        ]
        diagnostics["candidate_decisions"] = candidate_decisions
        diagnostics["candidate_disagreement"] = len(
            {json.dumps(value, sort_keys=True) for value in candidate_decisions}
        ) > 1
        selection = await _call(
            client,
            model=config.verifier_model,
            user=_selection_prompt(case, candidate_texts),
            max_tokens=config.verifier_max_tokens,
            temperature=0.0,
            stage="strong_verifier_selection",
            verifier=True,
        )
        calls.append(selection)
        return _finish(
            case=case,
            config=config,
            system=system,
            run_id=run_id,
            calls=calls,
            final_text=selection.response.text,
            diagnostics=diagnostics,
        )

    if system == "adaptive":
        verification = await _call(
            client,
            model=config.verifier_model,
            user=_verification_prompt(case, draft.response.text),
            max_tokens=config.verifier_max_tokens,
            temperature=0.0,
            stage="verification_gate",
            verifier=True,
        )
        calls.append(verification)
        result = _payload(verification.response.text) or {}
        accept = result.get("accept") is True
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        diagnostics.update(
            {
                "verifier_accept": accept,
                "verifier_confidence": confidence,
                "verifier_error_type": result.get("error_type"),
            }
        )
        if (
            accept
            and confidence >= config.adaptive_accept_threshold
            and diagnostics["initial_decision"] is not None
        ):
            return _finish(
                case=case,
                config=config,
                system=system,
                run_id=run_id,
                calls=calls,
                final_text=draft.response.text,
                diagnostics=diagnostics,
            )

        diagnostics["escalated"] = True
        alternatives = await asyncio.gather(
            *[
                _call(
                    client,
                    model=config.generator_model,
                    user=_checklist_prompt(case),
                    max_tokens=config.generator_max_tokens,
                    temperature=config.sampling_temperature,
                    stage=f"escalated_candidate_{index + 1}",
                )
                for index in range(2)
            ]
        )
        calls.extend(alternatives)
        candidate_texts = [draft.response.text, *[call.response.text for call in alternatives]]
        candidate_decisions = [
            canonical_decision(case.task, parse_response(text)[0]) for text in candidate_texts
        ]
        diagnostics["candidate_decisions"] = candidate_decisions
        diagnostics["candidate_disagreement"] = len(
            {json.dumps(value, sort_keys=True) for value in candidate_decisions}
        ) > 1
        selection = await _call(
            client,
            model=config.verifier_model,
            user=_selection_prompt(case, candidate_texts),
            max_tokens=config.verifier_max_tokens,
            temperature=0.0,
            stage="escalated_selection",
            verifier=True,
        )
        calls.append(selection)
        return _finish(
            case=case,
            config=config,
            system=system,
            run_id=run_id,
            calls=calls,
            final_text=selection.response.text,
            diagnostics=diagnostics,
        )

    raise ValueError(f"unsupported system: {system}")
