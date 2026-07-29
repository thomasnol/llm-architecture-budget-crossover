from pathlib import Path

from budget_crossover.v2_config import load_v2_config
from budget_crossover.v2_dataset import build_insurance_v2_cases, build_v2_case_set

REPO = Path(__file__).resolve().parents[1]


def test_v2_builds_all_unique_cases_without_outcome_selection():
    cases = build_insurance_v2_cases(
        REPO / "data" / "raw" / "train.parquet",
        evidence_condition="pooled",
        fixed_source_model="o3",
        max_context_chars=64000,
    )
    assert len(cases) == 80
    assert len({case.case_id for case in cases}) == 80
    assert all(case.gold_decision for case in cases)
    assert all(case.metadata["fixed_source_model"] == "o3" for case in cases)
    assert all(case.evidence_chars < 64000 for case in cases)


def test_fixed_and_pooled_conditions_keep_identical_gold():
    fixed = build_insurance_v2_cases(
        REPO / "data" / "raw" / "train.parquet",
        evidence_condition="fixed",
        fixed_source_model="o3",
        max_context_chars=64000,
    )
    pooled = build_insurance_v2_cases(
        REPO / "data" / "raw" / "train.parquet",
        evidence_condition="pooled",
        fixed_source_model="o3",
        max_context_chars=64000,
    )
    assert {case.case_id: case.gold_decision for case in fixed} == {
        case.case_id: case.gold_decision for case in pooled
    }
    assert any(
        pooled_case.evidence_chars > fixed_case.evidence_chars
        for pooled_case, fixed_case in zip(pooled, fixed, strict=True)
    )


def test_pilot_and_main_case_sets_are_disjoint():
    pilot_config = load_v2_config(REPO / "configs" / "v2_pilot.yaml")
    main_config = load_v2_config(REPO / "configs" / "v2_main.yaml")
    pilot = build_v2_case_set(REPO, pilot_config)
    main = build_v2_case_set(REPO, main_config)
    assert len(pilot) == 30
    assert len(main) == 268
    assert {case.case_id for case in pilot}.isdisjoint(
        {case.case_id for case in main}
    )
    assert sum(case.dataset == "insurance" for case in main) == 68
    assert sum(case.dataset == "mmlu_pro" for case in main) == 200
