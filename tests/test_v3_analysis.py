from pathlib import Path

import pandas as pd

from budget_crossover.io import write_jsonl
from budget_crossover.models import CallRecord, GatewayResponse, Usage
from budget_crossover.v3_analysis import analyze_v3
from budget_crossover.v3_config import V3Config
from budget_crossover.v3_manifest import ensure_v3_manifest
from budget_crossover.v3_models import V3Case, V3Generation
from budget_crossover.v3_runner import v3_generation_path


def _call() -> CallRecord:
    return CallRecord(
        stage="final",
        token_cap=256,
        response=GatewayResponse(
            text="",
            model="gpt-5.4-mini",
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
            latency_seconds=0.1,
            credential_slot=1,
        ),
    )


def _case(pair: int, variant: str) -> V3Case:
    protected = "White" if variant == "observed" else "Black or African American"
    return V3Case(
        case_id=f"pair-{pair}-{variant}",
        pair_id=f"pair-{pair}",
        counterfactual_variant=variant,
        source_row_id=str(pair),
        state="DC",
        historical_action="originated",
        policy_decision="approve",
        policy_reason_codes=["meets_policy"],
        documents={
            "application": "income",
            "collateral": "ltv",
            "credit": "dti",
            "quality_control": "complete",
            "compliance_monitoring": f"race {protected}",
        },
        protected_attributes={
            "race": protected,
            "sex": "Female",
            "ethnicity": "Not Hispanic or Latino",
            "age_band": "35-44",
        },
        changed_protected_attribute="race",
        complexity="routine",
    )


def test_v3_analysis_clusters_counterfactual_twins(tmp_path: Path):
    config = V3Config(
        experiment_name="mock",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
    )
    cases = [
        _case(pair, variant)
        for pair in range(4)
        for variant in ("observed", "counterfactual")
    ]
    generations = []
    for case_index, case in enumerate(cases):
        for system in config.systems:
            correct = system == "adaptive" or case_index % 2 == 0
            decision = "approve" if correct else "deny"
            reasons = ["meets_policy"] if correct else ["excessive_dti"]
            generations.append(
                V3Generation(
                    run_id=f"{case.case_id}-{system}",
                    case_id=case.case_id,
                    pair_id=case.pair_id,
                    counterfactual_variant=case.counterfactual_variant,
                    system=system,
                    token_budget=2048,
                    model=config.generator_model,
                    supervisor_model=config.supervisor_model,
                    parsed_decision={
                        "decision": decision,
                        "reason_codes": reasons,
                    },
                    calls=[_call()],
                    wall_time_seconds=0.1,
                )
            )
    ensure_v3_manifest(tmp_path, config, cases)
    write_jsonl(v3_generation_path(tmp_path, config), generations)
    report = analyze_v3(repo=tmp_path, config=config, cases=cases)
    assert report["observed_generations"] == 16
    tables = (
        tmp_path
        / "experiments"
        / "runs"
        / "mock"
        / "analysis"
        / "tables"
    )
    summary = pd.read_csv(tables / "system_summary.csv")
    adaptive = summary[summary["system"] == "adaptive"].iloc[0]
    assert adaptive["decision_accuracy"] == 1
    comparisons = pd.read_csv(tables / "paired_comparisons.csv")
    assert comparisons.iloc[0]["paired_applications"] == 4
