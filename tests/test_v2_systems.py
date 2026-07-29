from budget_crossover.models import GatewayResponse, Usage
from budget_crossover.v2_config import V2Config
from budget_crossover.v2_models import V2Case
from budget_crossover.v2_systems import run_v2_system


class FakeClient:
    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["user"]
        if '"accept": true_or_false' in user:
            text = (
                '{"accept":true,"confidence":0.95,"error_type":"none",'
                '"feedback":"supported"}'
            )
        else:
            text = '{"choice":"A","rationale":"The evidence supports A."}'
        return GatewayResponse(
            text=text,
            model=kwargs["model"],
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            latency_seconds=0.1,
            credential_slot=1,
        )


def config() -> V2Config:
    return V2Config(
        experiment_name="test",
        include_mmlu=False,
        systems=["direct"],
    )


def case() -> V2Case:
    return V2Case(
        case_id="mmlu-test",
        dataset="mmlu_pro",
        task="Multiple Choice Reasoning",
        question="Which option is correct?\nA. Alpha\nB. Beta",
        context="Select one.",
        output_schema={"choice": "letter", "rationale": "string"},
        gold_decision={"choice": "A"},
    )


async def test_direct_records_gateway_usage():
    client = FakeClient()
    result = await run_v2_system(
        client,
        case=case(),
        system="direct",
        config=config(),
        run_id="direct",
    )
    assert result.parsed_decision == {"choice": "A"}
    assert len(result.calls) == 1
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.total_tokens == 120


async def test_system_call_counts_are_explicit():
    expected = {
        "strong_direct": 1,
        "self_critique": 3,
        "external_verify": 3,
        "best_of_2": 3,
        "best_of_4": 5,
        "adaptive": 2,
    }
    for system, count in expected.items():
        client = FakeClient()
        result = await run_v2_system(
            client,
            case=case(),
            system=system,
            config=config(),
            run_id=system,
        )
        assert len(result.calls) == count


async def test_strong_direct_uses_verifier_model_without_extra_calls():
    client = FakeClient()
    result = await run_v2_system(
        client,
        case=case(),
        system="strong_direct",
        config=config(),
        run_id="strong-direct",
    )
    assert len(result.calls) == 1
    assert result.generator_model == "gpt-5.4"
    assert result.verifier_model is None
    assert client.calls[0]["model"] == "gpt-5.4"


async def test_adaptive_escalates_on_rejection():
    class RejectingClient(FakeClient):
        async def complete(self, **kwargs):
            response = await super().complete(**kwargs)
            if "INITIAL CANDIDATE" in kwargs["user"] and '"accept": true_or_false' in kwargs["user"]:
                response.text = (
                    '{"accept":false,"confidence":0.9,"error_type":"other",'
                    '"feedback":"reconsider"}'
                )
            return response

    result = await run_v2_system(
        RejectingClient(),
        case=case(),
        system="adaptive",
        config=config(),
        run_id="adaptive-reject",
    )
    assert result.diagnostics["escalated"] is True
    assert len(result.calls) == 5
