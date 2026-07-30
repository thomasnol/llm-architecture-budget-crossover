from pathlib import Path

"""Tests for canonical HMDA case construction."""

from budget_crossover.config import load_experiment_config
from budget_crossover.dataset import build_case_set, case_set_profile
from budget_crossover.validation import validate_cases

REPO = Path(__file__).resolve().parents[1]


def test_case_sets_are_balanced_paired_and_disjoint():
    pilot_config = load_experiment_config(REPO / "configs" / "pilot.yaml")
    main_config = load_experiment_config(REPO / "configs" / "main.yaml")
    pilot = build_case_set(REPO, pilot_config)
    main = build_case_set(REPO, main_config)

    pilot_profile = case_set_profile(pilot)
    main_profile = case_set_profile(main)
    assert pilot_profile["cases"] == 48
    assert main_profile["cases"] == 384
    assert set(main_profile["policy_decisions"].values()) == {48}
    assert set(main_profile["states"].values()) == {48}
    assert {case.source_row_id for case in pilot}.isdisjoint({case.source_row_id for case in main})


def test_counterfactuals_change_only_one_monitoring_attribute():
    config = load_experiment_config(REPO / "configs" / "pilot.yaml")
    cases = build_case_set(REPO, config)
    report = validate_cases(repo=REPO, config=config, cases=cases)
    assert report["pass"] is True
    assert report["post_decision_fields_supplied_to_models"] is False
    assert report["historical_action_used_as_gold"] is False
