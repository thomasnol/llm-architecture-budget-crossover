from __future__ import annotations

import json
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from budget_crossover.budget import BUDGET_TIERS
from budget_crossover.models import (
    Candidate,
    CellResult,
    EvidenceItem,
    GatewayResponse,
    MechanismTrace,
    PublicCase,
    Usage,
)
from budget_crossover.systems import CORE_INSTRUCTIONS, run_system


def _candidate(value: str, citation: str = "e1", *, expression: str | None = "10") -> dict:
    return Candidate(
        value=value,
        unit="USD",
        scale="ones",
        entity="Example Corp",
        period="2024",
        expression=expression,
        citations=(citation,),
    ).model_dump(mode="json")


def _case() -> PublicCase:
    evidence = tuple(
        EvidenceItem(
            evidence_id=f"e{index}",
            document_id="doc-1",
            kind="table_row",
            text=f"Metric {index} | 2024 | {index * 10}",
            headers=("Metric", "Period", "Value"),
            row_label=f"Metric {index}",
            unit="USD",
            scale="ones",
            entity="Example Corp",
            period="2024",
            ordinal=index - 1,
        )
        for index in range(1, 13)
    )
    return PublicCase(
        case_id="case-1",
        dataset="finqa",
        document_id="doc-1",
        question="What was Metric 1 in 2024?",
        evidence=evidence,
        stratum="headroom",
        metadata={"company": "Example Corp"},
    )


class ScriptedClient:
    def __init__(self, responses: Iterable[str], *, prompt_tokens: int = 100) -> None:
        self.responses = iter(responses)
        self.prompt_tokens = prompt_tokens
        self.calls: list[dict] = []

    async def count_prompt_tokens(self, *, model: str, system: str, user: str) -> int:
        del model, system, user
        return self.prompt_tokens

    async def complete(self, **kwargs) -> GatewayResponse:
        self.calls.append(kwargs)
        text = next(self.responses)
        usage = Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=20,
            total_tokens=self.prompt_tokens + 20,
        )
        return GatewayResponse(
            text=text,
            model=kwargs["model"],
            usage=usage,
            latency_seconds=0.01,
            credential_slot=1,
            request_id=f"request-{len(self.calls)}",
        )


class VariablePromptClient(ScriptedClient):
    def __init__(self, responses: Iterable[str], prompt_tokens: Iterable[int]) -> None:
        super().__init__(responses)
        self._prompt_script = iter(prompt_tokens)
        self._active_prompt_tokens = 0

    async def count_prompt_tokens(self, *, model: str, system: str, user: str) -> int:
        del model, system, user
        self._active_prompt_tokens = next(self._prompt_script)
        return self._active_prompt_tokens

    async def complete(self, **kwargs) -> GatewayResponse:
        self.prompt_tokens = self._active_prompt_tokens
        return await super().complete(**kwargs)


async def test_monolith_retrieves_tier_limit_and_makes_one_capped_answer_call():
    client = ScriptedClient([json.dumps(_candidate("10"))])

    result = await run_system(
        client,
        case=_case(),
        system="monolith",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
        repetition=2,
    )

    assert result.status == "ok"
    assert result.candidate == Candidate.model_validate(_candidate("10"))
    assert len(client.calls) == 1
    assert client.calls[0]["stage"] == "answer"
    assert client.calls[0]["max_tokens"] == 256
    assert client.calls[0]["model"] == "gpt-test"
    assert client.calls[0]["system"] == CORE_INSTRUCTIONS
    assert result.trace.actual_queries == (_case().question,)
    assert result.trace.retrieval_post_truncation_ids == ("e1", "e2")
    assert result.trace.candidate_count == 1
    assert result.trace.checks == ()
    assert result.trace.repair_attempted is False
    assert result.trace.accepted_candidate_index == 0
    assert result.trace.realized_tokens == 120
    assert result.trace.exit_reason == "completed"


async def test_verified_search_always_plans_at_low_budget_and_accepts_first_pass():
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "steps": ["Locate Metric 1", "Read the 2024 value"],
                    "queries": ["Metric 1 2024", "ignored beyond low limit"],
                }
            ),
            json.dumps(_candidate("10")),
        ]
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )

    assert [(call["stage"], call["max_tokens"]) for call in client.calls] == [
        ("planner", 128),
        ("candidate_0", 256),
    ]
    assert result.status == "ok"
    assert result.trace.planned_queries == ("Metric 1 2024",)
    assert result.trace.actual_queries == ("Metric 1 2024",)
    assert result.trace.candidate_count == 1
    assert len(result.trace.checks) == 1
    assert result.trace.checks[0].passed is True
    assert result.trace.accepted_candidate_index == 0
    assert result.trace.exit_reason == "accepted"


async def test_planner_system_instructions_do_not_conflict_with_its_plan_schema():
    client = ScriptedClient(
        [
            json.dumps({"steps": ["Find the metric"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("10")),
        ]
    )

    await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )

    planner, candidate = client.calls
    assert planner["stage"] == "planner"
    assert planner["system"] == CORE_INSTRUCTIONS
    assert "candidate schema" not in planner["system"].casefold()
    assert '"citations"' not in planner["system"]
    assert '"steps"' in planner["user"]
    assert '"queries"' in planner["user"]
    assert '"citations"' not in planner["user"]
    assert '"citations"' in candidate["user"]
    assert 'count(\"evidence-id\"' in candidate["user"]


async def test_verified_search_checks_sequentially_and_accepts_a_later_candidate():
    client = ScriptedClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("10", citation="fabricated")),
            json.dumps(_candidate("10")),
        ]
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["middle"],
        model="gpt-test",
    )

    assert [call["stage"] for call in client.calls] == [
        "planner",
        "candidate_0",
        "candidate_1",
    ]
    assert [check.passed for check in result.trace.checks] == [False, True]
    assert result.trace.candidate_count == 2
    assert result.trace.accepted_candidate_index == 1
    assert result.trace.repair_attempted is False
    assert result.trace.exit_reason == "accepted"


async def test_verified_search_repairs_once_from_checker_findings_and_rechecks():
    client = ScriptedClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("20")),
            json.dumps(_candidate("20")),
            json.dumps(_candidate("10")),
        ]
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["middle"],
        model="gpt-test",
    )

    assert [call["stage"] for call in client.calls] == [
        "planner",
        "candidate_0",
        "candidate_1",
        "repair",
    ]
    assert client.calls[-1]["max_tokens"] == 256
    assert "expression_mismatch" in client.calls[-1]["user"]
    assert "confidence" not in client.calls[-1]["user"].casefold()
    assert [check.passed for check in result.trace.checks] == [False, False, True]
    assert result.trace.candidate_count == 2
    assert result.trace.repair_attempted is True
    assert result.trace.accepted_candidate_index == 2
    assert result.trace.answer_changed is True
    assert len(result.trace.query_hashes) == len(result.trace.actual_queries)
    assert result.trace.retrieval_pre_truncation_ids
    assert result.trace.retrieval_post_truncation_ids
    assert len(result.trace.call_events) == 4
    assert all(event.usage is not None for event in result.trace.call_events)
    assert result.trace.realized_tokens == 480
    assert result.trace.exit_reason == "repair_accepted"


async def test_repair_uses_only_the_selected_rejected_candidates_findings():
    client = ScriptedClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("10", citation="fabricated")),
            json.dumps(_candidate("20")),
            json.dumps(_candidate("10")),
        ]
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["middle"],
        model="gpt-test",
    )

    repair_user = client.calls[-1]["user"]
    assert client.calls[-1]["stage"] == "repair"
    assert '"value":"20"' in repair_user
    assert "expression_mismatch" in repair_user
    assert "fabricated_citation" not in repair_user
    assert "unsupported_operand" not in repair_user
    assert result.trace.exit_reason == "repair_accepted"


async def test_verified_search_budget_exhaustion_never_returns_a_rejected_draft():
    client = VariablePromptClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("20")),
        ],
        prompt_tokens=[100, 100, 12_000],
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["middle"],
        model="gpt-test",
    )

    assert result.status == "architecture_failure"
    assert result.candidate is None
    assert [call["stage"] for call in client.calls] == ["planner", "candidate_0"]
    assert result.trace.candidate_count == 1
    assert result.trace.accepted_candidate_index is None
    assert result.trace.exit_reason == "budget_exhausted"


async def test_unaffordable_repair_is_terminal_and_never_returns_a_rejected_draft():
    client = VariablePromptClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("20")),
            json.dumps(_candidate("20")),
        ],
        prompt_tokens=[100, 100, 100, 12_000],
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["middle"],
        model="gpt-test",
    )

    assert result.status == "architecture_failure"
    assert result.candidate is None
    assert [call["stage"] for call in client.calls] == [
        "planner",
        "candidate_0",
        "candidate_1",
    ]
    assert result.trace.candidate_count == 2
    assert result.trace.repair_attempted is False
    assert result.trace.exit_reason == "budget_exhausted"


async def test_malformed_candidate_is_counted_and_scores_as_invalid_output():
    client = ScriptedClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            "not candidate json",
        ]
    )

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )

    assert result.status == "architecture_failure"
    assert result.candidate is None
    assert result.trace.candidate_count == 1
    assert result.trace.checks == ()
    assert result.trace.exit_reason == "invalid_output"


async def test_invalid_plan_does_not_claim_that_retrieval_occurred():
    client = ScriptedClient(["not planner json"])

    result = await run_system(
        client,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )

    assert result.status == "architecture_failure"
    assert result.trace.planned_queries == ()
    assert result.trace.actual_queries == ()
    assert result.trace.query_hashes == ()
    assert result.trace.retrieval_pre_truncation_ids == ()
    assert result.trace.retrieval_post_truncation_ids == ()
    assert result.trace.exit_reason == "planner_invalid"


async def test_unverified_search_generates_full_opportunity_and_uses_stable_plurality():
    client = ScriptedClient(
        [
            json.dumps({"steps": ["Find value"], "queries": ["Metric 1 2024"]}),
            json.dumps(_candidate("10")),
            json.dumps(_candidate("20")),
            json.dumps(_candidate("20.0")),
            json.dumps(_candidate("10.00")),
        ]
    )

    result = await run_system(
        client,
        case=_case(),
        system="unverified_search",
        tier=BUDGET_TIERS["high"],
        model="gpt-test",
    )

    assert [call["stage"] for call in client.calls] == [
        "planner",
        "candidate_0",
        "candidate_1",
        "candidate_2",
        "candidate_3",
    ]
    assert result.status == "ok"
    assert result.candidate.value == "10"
    assert result.trace.candidate_count == 4
    assert result.trace.checks == ()
    assert result.trace.repair_attempted is False
    assert result.trace.accepted_candidate_index == 0
    assert result.trace.exit_reason == "plurality_selected"


async def test_systems_share_the_initial_model_core_evidence_and_candidate_contract():
    planner = json.dumps({"steps": ["Answer directly"], "queries": [_case().question]})
    answer = json.dumps(_candidate("10"))
    monolith = ScriptedClient([answer])
    verified = ScriptedClient([planner, answer])
    unverified = ScriptedClient([planner, answer])

    await run_system(
        monolith,
        case=_case(),
        system="monolith",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )
    await run_system(
        verified,
        case=_case(),
        system="verified_search",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )
    await run_system(
        unverified,
        case=_case(),
        system="unverified_search",
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )

    answer_calls = [monolith.calls[0], verified.calls[1], unverified.calls[1]]
    assert {call["model"] for call in answer_calls} == {"gpt-test"}
    assert {call["system"] for call in answer_calls} == {CORE_INSTRUCTIONS}
    assert len({call["user"] for call in answer_calls}) == 1
    assert all('"citations"' in call["user"] for call in answer_calls)


def test_mechanism_trace_rejects_exit_reasons_outside_the_frozen_vocabulary():
    with pytest.raises(ValidationError, match="exit_reason"):
        MechanismTrace(exit_reason="typo_that_would_corrupt_analysis")


@pytest.mark.parametrize(
    ("field", "value"),
    [("system", "adaptive"), ("tier", "tiny"), ("status", "maybe")],
)
def test_cell_results_reject_values_outside_the_canonical_execution_vocabulary(field, value):
    payload = {
        "case_id": "case-1",
        "system": "monolith",
        "tier": "low",
        "repetition": 0,
        "status": "ok",
        "candidate": None,
        "trace": MechanismTrace(exit_reason="completed"),
    }
    payload[field] = value

    with pytest.raises(ValidationError, match=field):
        CellResult(**payload)


@pytest.mark.parametrize("system", ["monolith", "verified_search"])
async def test_unaffordable_initial_calls_are_terminal_architecture_failures(system):
    client = VariablePromptClient([], prompt_tokens=[4000])

    result = await run_system(
        client,
        case=_case(),
        system=system,
        tier=BUDGET_TIERS["low"],
        model="gpt-test",
    )

    assert result.status == "architecture_failure"
    assert result.candidate is None
    assert client.calls == []
    assert result.trace.call_events == ()
    assert result.trace.realized_tokens == 0
    assert result.trace.exit_reason == "budget_exhausted"
