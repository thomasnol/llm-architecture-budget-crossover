import json
from pathlib import Path

import pytest

from budget_crossover.validation import (
    OperationalGateInputs,
    assert_pilot_gate,
    evaluate_operational_gates,
    validate_cases,
    validate_run,
)


def _passing_inputs() -> OperationalGateInputs:
    return OperationalGateInputs(
        expected_cells=100,
        observed_cells=100,
        authoritative_usage_cells=100,
        expected_paired_cells=100,
        observed_paired_cells=100,
        unique_paired_cells=100,
        label_leakage_count=0,
        budget_overrun_count=0,
        schema_valid_cells=99,
        matched_blocks_total=100,
        unresolved_external_matched_blocks=1,
        expected_mechanism_counts={"planner_calls": 100, "checker_calls": 100},
        observed_mechanism_counts={"planner_calls": 100, "checker_calls": 100},
        low_tier_cases=100,
        low_tier_feasible_cases=100,
        verified_search_median_tokens={"low": 100.0, "middle": 120.0, "high": 144.0},
        easy_monolith_correct=90,
        easy_monolith_total=100,
        hard_monolith_correct=30,
        hard_monolith_total=100,
        checker_true_negatives=95,
        checker_actual_negatives=100,
        checker_true_positives=60,
        checker_actual_positives=100,
        correct_first_drafts_repaired=100,
        correct_to_wrong_repairs=5,
        checker_detected_wrong_first_drafts=100,
        wrong_first_drafts_corrected=20,
    )


def test_operational_gate_emits_every_component_and_is_non_overridable(tmp_path: Path):
    output = tmp_path / "operational_gate.json"

    artifact = evaluate_operational_gates(_passing_inputs(), output_path=output)

    assert artifact.passed is True
    assert artifact.override_allowed is False
    assert {component.name for component in artifact.components} == {
        "complete_grid",
        "unique_paired_cells",
        "authoritative_usage",
        "label_leakage",
        "budget_overruns",
        "schema_validity",
        "unresolved_external_matched_blocks",
        "exact_mechanism_counts",
        "low_tier_feasibility",
        "verified_search_low_to_middle_token_growth",
        "verified_search_middle_to_high_token_growth",
        "easy_monolith_accuracy",
        "hard_monolith_accuracy_lower",
        "hard_monolith_accuracy_upper",
        "checker_specificity",
        "checker_sensitivity",
        "correct_to_wrong_repair",
        "wrong_first_draft_correction",
    }
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["override_allowed"] is False
    assert payload["inputs"] == _passing_inputs().model_dump(mode="json")
    assert all("value" in component for component in payload["components"])
    matched = next(
        component
        for component in artifact.components
        if component.name == "unresolved_external_matched_blocks"
    )
    assert matched.zero_denominator_rule == "pass_only_when_total_and_unresolved_are_zero"


@pytest.mark.parametrize(
    ("updates", "failed_component"),
    [
        (
            {
                "observed_cells": 99,
                "authoritative_usage_cells": 99,
                "schema_valid_cells": 99,
            },
            "complete_grid",
        ),
        ({"unique_paired_cells": 99}, "unique_paired_cells"),
        ({"authoritative_usage_cells": 99}, "authoritative_usage"),
        ({"label_leakage_count": 1}, "label_leakage"),
        ({"budget_overrun_count": 1}, "budget_overruns"),
        ({"schema_valid_cells": 98}, "schema_validity"),
        (
            {"unresolved_external_matched_blocks": 2},
            "unresolved_external_matched_blocks",
        ),
        ({"observed_mechanism_counts": {"planner_calls": 99}}, "exact_mechanism_counts"),
        ({"low_tier_feasible_cases": 99}, "low_tier_feasibility"),
        (
            {"verified_search_median_tokens": {"low": 100, "middle": 119, "high": 144}},
            "verified_search_low_to_middle_token_growth",
        ),
        (
            {"verified_search_median_tokens": {"low": 100, "middle": 120, "high": 143}},
            "verified_search_middle_to_high_token_growth",
        ),
        ({"easy_monolith_correct": 89}, "easy_monolith_accuracy"),
        ({"hard_monolith_correct": 29}, "hard_monolith_accuracy_lower"),
        ({"hard_monolith_correct": 86}, "hard_monolith_accuracy_upper"),
        ({"checker_true_negatives": 94}, "checker_specificity"),
        ({"checker_true_positives": 59}, "checker_sensitivity"),
        ({"correct_to_wrong_repairs": 6}, "correct_to_wrong_repair"),
        ({"wrong_first_drafts_corrected": 19}, "wrong_first_draft_correction"),
    ],
)
def test_each_operational_gate_fails_at_the_first_disallowed_boundary(
    updates: dict[str, object], failed_component: str
):
    artifact = evaluate_operational_gates(_passing_inputs().model_copy(update=updates))

    assert artifact.passed is False
    components = {component.name: component for component in artifact.components}
    assert components[failed_component].passed is False
    assert artifact.failed_components == (failed_component,)


def test_operational_gate_threshold_boundaries_are_inclusive():
    lower = evaluate_operational_gates(_passing_inputs())
    upper_hard = evaluate_operational_gates(
        _passing_inputs().model_copy(update={"hard_monolith_correct": 85})
    )

    assert lower.passed is True
    assert upper_hard.passed is True


def test_zero_checker_and_repair_opportunities_emit_failed_components():
    artifact = evaluate_operational_gates(
        _passing_inputs().model_copy(
            update={
                "checker_true_negatives": 0,
                "checker_actual_negatives": 0,
                "checker_true_positives": 0,
                "checker_actual_positives": 0,
                "correct_first_drafts_repaired": 0,
                "correct_to_wrong_repairs": 0,
                "checker_detected_wrong_first_drafts": 0,
                "wrong_first_drafts_corrected": 0,
            }
        )
    )

    assert set(artifact.failed_components) >= {
        "checker_specificity",
        "checker_sensitivity",
        "correct_to_wrong_repair",
        "wrong_first_draft_correction",
    }
    assert len(artifact.components) == 18


@pytest.mark.parametrize(
    ("bridge", "message"),
    [
        (validate_cases, "legacy case validator was removed"),
        (validate_run, "legacy run validator was removed"),
        (assert_pilot_gate, "legacy pilot gate was removed"),
    ],
)
def test_legacy_validation_bridges_fail_closed_with_migration_errors(bridge, message):
    with pytest.raises(RuntimeError, match=message):
        bridge()
