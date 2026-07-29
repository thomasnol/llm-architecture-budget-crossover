from pathlib import Path

from budget_crossover.dataset import build_cases, sample_cases

REPO = Path(__file__).resolve().parents[1]


def test_builds_unique_leakage_controlled_cases():
    cases = build_cases(REPO / "data" / "raw" / "train.parquet")
    assert len(cases) == 80
    assert len({case.case_id for case in cases}) == 80
    assert max(case.evidence_chars for case in cases) < 72000
    assert all(case.accepted_reference_answers for case in cases)


def test_main_sample_is_reproducible_and_stratified():
    cases = build_cases(REPO / "data" / "raw" / "train.parquet")
    quotas = {
        "Appetite Check": 14,
        "Business Classification": 2,
        "Deductibles": 7,
        "Policy Limits": 13,
        "Product Recommendations": 14,
        "Small Business Elibility Check": 10,
    }
    first = sample_cases(cases, sample_size=60, seed=20260728, task_quotas=quotas)
    second = sample_cases(cases, sample_size=60, seed=20260728, task_quotas=quotas)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert {task: sum(case.task == task for case in first) for task in quotas} == quotas
