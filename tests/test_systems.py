from budget_crossover.models import GatewayResponse, Usage

"""Tests for canonical orchestration systems."""
from budget_crossover.config import ExperimentConfig
from budget_crossover.records import Case
from budget_crossover.systems import run_system


class FakeClient:
    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["user"]
        if "SPECIALIST REPORTS" in user:
            text = (
                '{"decision":"approve","reason_codes":["meets_policy"],'
                '"confidence":0.95,"rationale":"Reports support approval."}'
            )
        elif '"request":["tool_name"' in user:
            text = '{"request":["application","collateral","credit"],"rationale":"needed"}'
        elif '"accept":true_or_false' in user:
            text = (
                '{"accept":true,"policy_errors":[],"prohibited_field_use":false,'
                '"required_correction":"none"}'
            )
        elif '"recommended_decision"' in user and "SPECIALIST REPORTS" not in user:
            text = (
                '{"recommended_decision":"approve","reason_codes":["meets_policy"],'
                '"material_facts":["within thresholds"]}'
            )
        elif '"prohibited_for_decision"' in user:
            text = (
                '{"prohibited_for_decision":["race","sex"],"data_quality_flags":[],'
                '"instruction":"ignore monitoring fields"}'
            )
        else:
            text = (
                '{"decision":"approve","reason_codes":["meets_policy"],'
                '"confidence":0.95,"rationale":"All thresholds pass."}'
            )
        return GatewayResponse(
            text=text,
            model=kwargs["model"],
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            latency_seconds=0.1,
            credential_slot=1,
        )


def config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048, 4096, 8192],
        systems=[
            "monolith",
            "retrieval",
            "committee",
            "guardrail",
            "adaptive",
        ],
    )


def case() -> Case:
    financial = {
        "application": "Income $120,000; amount $300,000; term 360 months.",
        "collateral": "Property $500,000; LTV 60%; conforming C.",
        "credit": "DTI 36.",
        "quality_control": "All required fields reported.",
    }
    return Case(
        case_id="pair-observed",
        pair_id="pair",
        counterfactual_variant="observed",
        source_row_id="source",
        state="DC",
        historical_action="originated",
        policy_decision="approve",
        policy_reason_codes=["meets_policy"],
        documents={
            **financial,
            "compliance_monitoring": ("Race: White; Sex: Female; monitoring only and prohibited."),
        },
        protected_attributes={
            "race": "White",
            "sex": "Female",
            "ethnicity": "Not Hispanic or Latino",
            "age_band": "35-44",
        },
        changed_protected_attribute="race",
        complexity="routine",
    )


async def test_architecture_call_counts_and_adaptive_routing():
    expected = {
        "monolith": 1,
        "retrieval": 2,
        "committee": 4,
        "guardrail": 3,
        "adaptive": 1,
    }
    for system, call_count in expected.items():
        client = FakeClient()
        result = await run_system(
            client,
            case=case(),
            system=system,
            token_budget=8192,
            config=config(),
            run_id=system,
        )
        assert result.parsed_decision == {
            "decision": "approve",
            "reason_codes": ["meets_policy"],
        }
        assert len(result.calls) == call_count
        assert result.total_tokens == call_count * 120
        assert result.diagnostics["budget_overrun"] is False


async def test_architecture_study_holds_model_constant_across_every_role():
    for system in config().systems:
        client = FakeClient()
        await run_system(
            client,
            case=case(),
            system=system,
            token_budget=8192,
            config=config(),
            run_id=f"fixed-model-{system}",
        )
        assert {call["model"] for call in client.calls} == {config().generator_model}


async def test_routing_study_assigns_primary_and_supervisor_models_explicitly():
    routing = ExperimentConfig(
        experiment_name="routing-test",
        study_kind="routing",
        hmda_source_sha256="0" * 64,
        token_budgets=[8192],
        systems=["always_primary", "always_supervisor", "selective_supervisor"],
        adaptive_confidence_threshold=0.99,
    )

    supervisor_client = FakeClient()
    await run_system(
        supervisor_client,
        case=case(),
        system="always_supervisor",
        token_budget=8192,
        config=routing,
        run_id="always-supervisor",
    )
    assert [call["model"] for call in supervisor_client.calls] == [routing.supervisor_model]

    selective_client = FakeClient()
    await run_system(
        selective_client,
        case=case(),
        system="selective_supervisor",
        token_budget=8192,
        config=routing,
        run_id="selective",
    )
    assert [call["model"] for call in selective_client.calls] == [
        routing.generator_model,
        routing.supervisor_model,
        routing.generator_model,
    ]


async def test_budget_exhaustion_is_an_observed_system_result():
    result = await run_system(
        FakeClient(),
        case=case(),
        system="monolith",
        token_budget=512,
        config=config(),
        run_id="tight",
    )
    assert result.status == "budget_exhausted"
    assert result.parsed_decision is None
    assert result.diagnostics["budget_exhausted"] is True
    assert result.diagnostics["budget_overrun"] is False
