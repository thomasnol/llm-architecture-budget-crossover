import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from budget_crossover.runner import CellKey
from budget_crossover.validation import (
    GateComponent,
    OperationalGateArtifact,
    OperationalGateInputs,
    assert_pilot_gate,
    evaluate_operational_gates,
    validate_cases,
    validate_run,
)


def _cell_grid() -> tuple[CellKey, ...]:
    return tuple(
        CellKey(
            case_id=f"case-{index}",
            system="monolith",
            tier="low",
            repetition=0,
        )
        for index in range(100)
    )


def _passing_inputs() -> OperationalGateInputs:
    grid = _cell_grid()
    return OperationalGateInputs(
        expected_cell_keys=grid,
        observed_cell_keys=grid,
        authoritative_usage_cells=100,
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
                "observed_cell_keys": _cell_grid()[:-1],
                "authoritative_usage_cells": 99,
                "schema_valid_cells": 99,
            },
            "complete_grid",
        ),
        (
            {
                "observed_cell_keys": _cell_grid() + (_cell_grid()[0],),
                "authoritative_usage_cells": 101,
                "schema_valid_cells": 100,
            },
            "unique_paired_cells",
        ),
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


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=("nan", "positive-infinity", "negative-infinity"),
)
def test_gate_inputs_reject_non_finite_verified_search_medians(non_finite: float):
    with pytest.raises(ValidationError, match="finite"):
        _passing_inputs().model_copy(
            update={
                "verified_search_median_tokens": {
                    "low": 100.0,
                    "middle": non_finite,
                    "high": 144.0,
                }
            }
        )


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=("nan", "positive-infinity", "negative-infinity"),
)
@pytest.mark.parametrize("field", ["value", "threshold"])
def test_gate_components_reject_non_finite_numeric_telemetry(
    field: str,
    non_finite: float,
):
    payload = {
        "name": "telemetry",
        "passed": False,
        "value": {"nested_rate": 0.5},
        "comparison": ">=",
        "threshold": {"nested_rate": 0.9},
    }
    payload[field] = {"nested_rate": non_finite}

    with pytest.raises(ValidationError, match="finite"):
        GateComponent(**payload)
    with pytest.raises(ValidationError, match="finite"):
        GateComponent.model_construct(**payload)


def test_finite_failed_telemetry_still_emits_an_operational_gate_artifact():
    artifact = evaluate_operational_gates(
        _passing_inputs().model_copy(
            update={
                "verified_search_median_tokens": {
                    "low": 100.0,
                    "middle": 119.0,
                    "high": 144.0,
                }
            }
        )
    )

    assert artifact.passed is False
    assert artifact.failed_components == (
        "verified_search_low_to_middle_token_growth",
    )
    growth = next(
        component
        for component in artifact.components
        if component.name == "verified_search_low_to_middle_token_growth"
    )
    assert growth.value == pytest.approx(0.19)


def test_complete_grid_compares_exact_cell_keys_not_only_counts():
    expected = _cell_grid()
    replacement = expected[0].model_copy(update={"repetition": 1})
    observed = (replacement, *expected[1:])

    artifact = evaluate_operational_gates(
        _passing_inputs().model_copy(update={"observed_cell_keys": observed})
    )

    components = {component.name: component for component in artifact.components}
    assert artifact.passed is False
    assert artifact.failed_components == ("complete_grid",)
    assert components["complete_grid"].value["missing_cell_keys"] == (
        "case-0:monolith:low:0",
    )
    assert components["complete_grid"].value["unexpected_cell_keys"] == (
        "case-0:monolith:low:1",
    )
    assert components["unique_paired_cells"].passed is True


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


def test_gate_artifact_constructor_and_copy_cannot_forge_a_failed_verdict():
    failed = evaluate_operational_gates(
        _passing_inputs().model_copy(update={"label_leakage_count": 1})
    )
    forged_payload = failed.model_dump(mode="python", round_trip=True)
    forged_payload.update(
        passed=True,
        override_allowed=True,
        failed_components=(),
    )

    with pytest.raises(ValidationError):
        OperationalGateArtifact.model_validate(forged_payload)
    with pytest.raises(ValidationError):
        failed.model_copy(
            update={
                "passed": True,
                "override_allowed": True,
                "failed_components": (),
            }
        )


def test_gate_artifact_model_construct_cannot_bypass_summary_validation():
    failed = evaluate_operational_gates(
        _passing_inputs().model_copy(update={"label_leakage_count": 1})
    )
    forged_payload = failed.model_dump(mode="python", round_trip=True)
    forged_payload.update(passed=True, failed_components=())

    with pytest.raises(ValidationError):
        OperationalGateArtifact.model_construct(**forged_payload)


def test_gate_artifact_recomputes_nested_components_from_typed_inputs():
    passing = evaluate_operational_gates(_passing_inputs())
    forged_components = tuple(
        GateComponent.model_construct(
            **(
                component.model_dump(mode="python", round_trip=True)
                | ({"passed": False, "value": 1} if component.name == "label_leakage" else {})
            )
        )
        for component in passing.components
    )
    forged_payload = passing.model_dump(mode="python", round_trip=True)
    forged_payload.update(
        passed=False,
        failed_components=("label_leakage",),
        components=forged_components,
    )

    with pytest.raises(ValidationError):
        OperationalGateArtifact.model_validate(forged_payload)


def test_gate_input_nested_maps_are_deeply_immutable():
    inputs = _passing_inputs()

    with pytest.raises(TypeError, match="frozen mapping"):
        inputs.expected_mechanism_counts["planner_calls"] = 99
    with pytest.raises(TypeError, match="frozen mapping"):
        inputs.verified_search_median_tokens["middle"] = 1


def test_gate_component_nested_maps_are_deeply_immutable():
    artifact = evaluate_operational_gates(_passing_inputs())
    mechanism = next(
        component
        for component in artifact.components
        if component.name == "exact_mechanism_counts"
    )

    with pytest.raises(TypeError, match="frozen mapping"):
        mechanism.value["planner_calls"] = 99
    with pytest.raises(TypeError, match="frozen mapping"):
        mechanism.threshold["planner_calls"] = 99


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
