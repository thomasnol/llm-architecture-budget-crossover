from pathlib import Path

"""Tests for canonical experiment analysis."""

import pandas as pd
import pytest

from budget_crossover.analysis import (
    _crossover_estimates,
    _paired_comparisons,
    analyze_run,
)
from budget_crossover.config import ExperimentConfig
from budget_crossover.io import write_jsonl
from budget_crossover.manifest import ensure_manifest
from budget_crossover.models import CallRecord, GatewayResponse, Usage
from budget_crossover.records import Case, Generation
from budget_crossover.runner import generation_path


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


def _case(pair: int, variant: str) -> Case:
    protected = "White" if variant == "observed" else "Black or African American"
    return Case(
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


def test_analysis_clusters_counterfactual_twins(tmp_path: Path):
    config = ExperimentConfig(
        experiment_name="mock",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        repetitions=2,
        bootstrap_replicates=100,
    )
    cases = [
        _case(pair, variant) for pair in range(4) for variant in ("observed", "counterfactual")
    ]
    generations = []
    for case_index, case in enumerate(cases):
        for system in config.systems:
            for repetition in range(config.repetitions):
                correct = system == "adaptive" or case_index % 2 == 0
                decision = "approve" if correct else "deny"
                reasons = ["meets_policy"] if correct else ["excessive_dti"]
                generations.append(
                    Generation(
                        run_id=f"{case.case_id}-{system}-{repetition}",
                        case_id=case.case_id,
                        pair_id=case.pair_id,
                        counterfactual_variant=case.counterfactual_variant,
                        system=system,
                        token_budget=2048,
                        repetition=repetition,
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
    ensure_manifest(tmp_path, config, cases)
    write_jsonl(generation_path(tmp_path, config), generations)
    report = analyze_run(repo=tmp_path, config=config, cases=cases)
    assert report["observed_generations"] == 32
    assert report["unique_generation_cells"] == 32
    tables = tmp_path / "experiments" / "runs" / "mock" / "analysis" / "tables"
    summary = pd.read_csv(tables / "system_summary.csv")
    adaptive = summary[summary["system"] == "adaptive"].iloc[0]
    assert adaptive["decision_accuracy"] == 1
    comparisons = pd.read_csv(tables / "paired_comparisons.csv")
    assert comparisons.iloc[0]["paired_applications"] == 4


def test_analysis_rejects_incomplete_grid_unless_diagnostic_is_explicit(
    tmp_path: Path,
):
    config = ExperimentConfig(
        experiment_name="incomplete",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
    )
    cases = [_case(0, "observed"), _case(0, "counterfactual")]
    one_generation = Generation(
        run_id="only-one",
        case_id=cases[0].case_id,
        pair_id=cases[0].pair_id,
        counterfactual_variant=cases[0].counterfactual_variant,
        system="monolith",
        token_budget=2048,
        model=config.generator_model,
        parsed_decision={
            "decision": "approve",
            "reason_codes": ["meets_policy"],
        },
        calls=[_call()],
    )
    ensure_manifest(tmp_path, config, cases)
    write_jsonl(generation_path(tmp_path, config), [one_generation])

    with pytest.raises(RuntimeError, match="incomplete generation grid"):
        analyze_run(repo=tmp_path, config=config, cases=cases)

    report = analyze_run(
        repo=tmp_path,
        config=config,
        cases=cases,
        diagnostic=True,
    )
    assert report["diagnostic"] is True
    assert report["incomplete"] is True
    assert report["generation_completion_rate"] == 0.25


def test_coverage_itt_accuracy_and_counterfactual_flip_rate_are_explicit(
    tmp_path: Path,
):
    config = ExperimentConfig(
        experiment_name="estimands",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
    )
    cases = [_case(0, "observed"), _case(0, "counterfactual")]
    generations = []
    for system, decisions in {
        "monolith": ["approve", "deny"],
        "adaptive": ["approve", "approve"],
    }.items():
        for case, decision in zip(cases, decisions, strict=True):
            generations.append(
                Generation(
                    run_id=f"{case.case_id}-{system}",
                    case_id=case.case_id,
                    pair_id=case.pair_id,
                    counterfactual_variant=case.counterfactual_variant,
                    system=system,
                    token_budget=2048,
                    model=config.generator_model,
                    parsed_decision={
                        "decision": decision,
                        "reason_codes": (
                            ["meets_policy"] if decision == "approve" else ["excessive_dti"]
                        ),
                    },
                    calls=[_call()],
                )
            )
    ensure_manifest(tmp_path, config, cases)
    write_jsonl(generation_path(tmp_path, config), generations)

    analyze_run(repo=tmp_path, config=config, cases=cases)
    summary = pd.read_csv(
        tmp_path
        / "experiments"
        / "runs"
        / "estimands"
        / "analysis"
        / "tables"
        / "system_summary.csv"
    ).set_index("system")

    assert summary.loc["monolith", "coverage_rate"] == 1
    assert summary.loc["monolith", "intention_to_treat_accuracy"] == 0.5
    assert summary.loc["monolith", "conditional_decision_accuracy"] == 0.5
    assert summary.loc["monolith", "counterfactual_flip_rate"] == 1
    assert summary.loc["adaptive", "intention_to_treat_accuracy"] == 1
    assert summary.loc["adaptive", "counterfactual_flip_rate"] == 0


def test_crossover_estimate_bootstraps_at_source_application_level():
    config = ExperimentConfig(
        experiment_name="crossover",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048, 4096],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
    )
    rows = []
    for pair in range(4):
        for budget in config.token_budgets:
            rows.extend(
                [
                    {
                        "pair_id": f"pair-{pair}",
                        "system": "monolith",
                        "token_budget": budget,
                        "both_decisions_correct": 1,
                    },
                    {
                        "pair_id": f"pair-{pair}",
                        "system": "adaptive",
                        "token_budget": budget,
                        "both_decisions_correct": int(budget == 4096),
                    },
                ]
            )

    crossover = _crossover_estimates(pd.DataFrame(rows), config=config)

    assert crossover.iloc[0]["system"] == "adaptive"
    assert crossover.iloc[0]["crossover_detected"]
    assert crossover.iloc[0]["crossover_budget"] == 4096
    assert crossover.iloc[0]["crossover_ci_low"] == 4096
    assert crossover.iloc[0]["crossover_ci_high"] == 4096


def test_identical_curves_do_not_create_a_spurious_crossover():
    config = ExperimentConfig(
        experiment_name="no-crossover",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048, 4096],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
    )
    rows = [
        {
            "pair_id": f"pair-{pair}",
            "system": system,
            "token_budget": budget,
            "both_decisions_correct": 1,
        }
        for pair in range(4)
        for budget in config.token_budgets
        for system in config.systems
    ]

    crossover = _crossover_estimates(pd.DataFrame(rows), config=config)

    assert not crossover.iloc[0]["crossover_detected"]
    assert pd.isna(crossover.iloc[0]["crossover_budget"])
    assert crossover.iloc[0]["bootstrap_crossover_support_rate"] == 0


def test_mcnemar_requires_success_in_every_repetition():
    config = ExperimentConfig(
        experiment_name="repeated-pairs",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        repetitions=2,
        bootstrap_replicates=100,
    )
    pair_frame = pd.DataFrame(
        [
            {
                "pair_id": "pair-0",
                "system": system,
                "token_budget": 2048,
                "both_decisions_correct": value,
            }
            for system, values in {
                "monolith": [0, 0],
                "adaptive": [1, 0],
            }.items()
            for value in values
        ]
    )

    comparison = _paired_comparisons(pair_frame, config=config)

    assert comparison.iloc[0]["accuracy_difference_vs_baseline"] == 0.5
    assert comparison.iloc[0]["improved_applications"] == 0
    assert comparison.iloc[0]["mcnemar_success_rule"] == "all_repetitions_both_twins_correct"
