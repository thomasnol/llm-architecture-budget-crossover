from pathlib import Path

import pandas as pd

from budget_crossover.io import write_jsonl
from budget_crossover.models import CallRecord, GatewayResponse, Usage
from budget_crossover.v2_analysis import analyze_v2
from budget_crossover.v2_config import V2Config
from budget_crossover.v2_manifest import ensure_run_manifest
from budget_crossover.v2_models import V2Case, V2Generation
from budget_crossover.v2_runner import v2_generation_path
from budget_crossover.v2_validation import validate_v2_run


def _call(stage: str, *, model: str = "gpt-5.4-mini") -> CallRecord:
    return CallRecord(
        stage=stage,
        token_cap=512,
        response=GatewayResponse(
            text='{"choice":"A","rationale":"supported"}',
            model=model,
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
            latency_seconds=0.1,
            credential_slot=1,
            raw_finish_reason="stop",
        ),
    )


def test_mocked_end_to_end_analysis_and_validation(tmp_path: Path):
    config = V2Config(
        experiment_name="mock_main",
        insurance_sample_size=0,
        mmlu_sample_size=0,
        include_mmlu=False,
        systems=["direct", "adaptive"],
        bootstrap_replicates=100,
        judge_sample_per_system_dataset=1,
    )
    cases = [
        V2Case(
            case_id=f"{dataset}-{index}",
            dataset=dataset,
            task="Multiple Choice Reasoning",
            question="Choose A.",
            context="A is correct.",
            output_schema={"choice": "uppercase letter", "rationale": "string"},
            gold_decision={"choice": "A"},
        )
        for dataset in ("insurance", "mmlu_pro")
        for index in range(4)
    ]
    generations = []
    for case_index, case in enumerate(cases):
        direct_choice = "A" if case_index % 2 == 0 else "B"
        generations.extend(
            [
                V2Generation(
                    run_id=f"{case.case_id}-direct",
                    case_id=case.case_id,
                    dataset=case.dataset,
                    task=case.task,
                    system="direct",
                    generator_model="gpt-5.4-mini",
                    answer_text="",
                    parsed_decision={"choice": direct_choice},
                    calls=[_call("direct")],
                    wall_time_seconds=0.1,
                ),
                V2Generation(
                    run_id=f"{case.case_id}-adaptive",
                    case_id=case.case_id,
                    dataset=case.dataset,
                    task=case.task,
                    system="adaptive",
                    generator_model="gpt-5.4-mini",
                    verifier_model="gpt-5.4",
                    answer_text="",
                    parsed_decision={"choice": "A"},
                    calls=[
                        _call("draft"),
                        _call("verification_gate", model="gpt-5.4"),
                    ],
                    diagnostics={
                        "initial_decision": {"choice": direct_choice},
                        "verifier_accept": True,
                        "verifier_confidence": 0.95,
                    },
                    wall_time_seconds=0.2,
                ),
            ]
        )
    ensure_run_manifest(tmp_path, config, cases)
    write_jsonl(v2_generation_path(tmp_path, config), generations)

    validation = validate_v2_run(
        repo=tmp_path,
        config=config,
        cases=cases,
        require_judgments=False,
        require_pilot_gate=False,
    )
    assert validation["pass"] is True

    report = analyze_v2(repo=tmp_path, config=config, cases=cases)
    assert report["successful_generations"] == 16
    assert report["generation_completion_rate"] == 1
    tables = tmp_path / "experiments" / "runs" / "mock_main" / "analysis" / "tables"
    assert (tables / "budget_policy.csv").exists()
    comparisons = pd.read_csv(tables / "paired_comparisons.csv")
    assert set(comparisons["dataset"]) == {"insurance", "mmlu_pro"}
    assert (comparisons["accuracy_difference_vs_direct"] == 0.5).all()
