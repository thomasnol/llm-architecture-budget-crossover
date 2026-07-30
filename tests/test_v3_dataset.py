from pathlib import Path

from budget_crossover.v3_config import load_v3_config
from budget_crossover.v3_dataset import build_v3_case_set, case_set_profile
from budget_crossover.v3_validation import validate_v3_cases

REPO = Path(__file__).resolve().parents[1]


def test_v3_case_sets_are_balanced_paired_and_disjoint():
    pilot_config = load_v3_config(REPO / "configs" / "v3_pilot.yaml")
    main_config = load_v3_config(REPO / "configs" / "v3_main.yaml")
    pilot = build_v3_case_set(REPO, pilot_config)
    main = build_v3_case_set(REPO, main_config)

    pilot_profile = case_set_profile(pilot)
    main_profile = case_set_profile(main)
    assert pilot_profile["cases"] == 48
    assert main_profile["cases"] == 192
    assert set(main_profile["policy_decisions"].values()) == {24}
    assert set(main_profile["states"].values()) == {24}
    assert {case.source_row_id for case in pilot}.isdisjoint(
        {case.source_row_id for case in main}
    )


def test_v3_counterfactuals_change_only_one_monitoring_attribute():
    config = load_v3_config(REPO / "configs" / "v3_pilot.yaml")
    cases = build_v3_case_set(REPO, config)
    report = validate_v3_cases(repo=REPO, config=config, cases=cases)
    assert report["pass"] is True
    assert report["post_decision_fields_supplied_to_models"] is False
    assert report["historical_action_used_as_gold"] is False
