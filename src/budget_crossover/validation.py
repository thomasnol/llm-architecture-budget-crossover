from __future__ import annotations

"""Non-overridable operational gates for the canonical experiment."""

import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .models import FrozenModel


class OperationalGateInputs(FrozenModel):
    expected_cells: int = Field(gt=0)
    observed_cells: int = Field(ge=0)
    authoritative_usage_cells: int = Field(ge=0)
    expected_paired_cells: int = Field(gt=0)
    observed_paired_cells: int = Field(ge=0)
    unique_paired_cells: int = Field(ge=0)
    label_leakage_count: int = Field(ge=0)
    budget_overrun_count: int = Field(ge=0)
    schema_valid_cells: int = Field(ge=0)
    matched_blocks_total: int = Field(ge=0)
    unresolved_external_matched_blocks: int = Field(ge=0)
    expected_mechanism_counts: dict[str, int]
    observed_mechanism_counts: dict[str, int]
    low_tier_cases: int = Field(ge=0)
    low_tier_feasible_cases: int = Field(ge=0)
    verified_search_median_tokens: dict[str, float]
    easy_monolith_correct: int = Field(ge=0)
    easy_monolith_total: int = Field(ge=0)
    hard_monolith_correct: int = Field(ge=0)
    hard_monolith_total: int = Field(ge=0)
    checker_true_negatives: int = Field(ge=0)
    checker_actual_negatives: int = Field(ge=0)
    checker_true_positives: int = Field(ge=0)
    checker_actual_positives: int = Field(ge=0)
    correct_first_drafts_repaired: int = Field(ge=0)
    correct_to_wrong_repairs: int = Field(ge=0)
    checker_detected_wrong_first_drafts: int = Field(ge=0)
    wrong_first_drafts_corrected: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> OperationalGateInputs:
        bounded = (
            (self.authoritative_usage_cells, self.observed_cells),
            (self.unique_paired_cells, self.observed_paired_cells),
            (self.schema_valid_cells, self.observed_cells),
            (self.unresolved_external_matched_blocks, self.matched_blocks_total),
            (self.low_tier_feasible_cases, self.low_tier_cases),
            (self.easy_monolith_correct, self.easy_monolith_total),
            (self.hard_monolith_correct, self.hard_monolith_total),
            (self.checker_true_negatives, self.checker_actual_negatives),
            (self.checker_true_positives, self.checker_actual_positives),
            (self.correct_to_wrong_repairs, self.correct_first_drafts_repaired),
            (
                self.wrong_first_drafts_corrected,
                self.checker_detected_wrong_first_drafts,
            ),
        )
        if any(numerator > denominator for numerator, denominator in bounded):
            raise ValueError("gate numerators cannot exceed their denominators")
        if set(self.verified_search_median_tokens) != {"low", "middle", "high"}:
            raise ValueError("verified-search medians require low, middle, and high tiers")
        if any(value <= 0 for value in self.verified_search_median_tokens.values()):
            raise ValueError("verified-search median tokens must be positive")
        if any(value < 0 for value in self.expected_mechanism_counts.values()):
            raise ValueError("expected mechanism counts must be nonnegative")
        if any(value < 0 for value in self.observed_mechanism_counts.values()):
            raise ValueError("observed mechanism counts must be nonnegative")
        return self


class GateComponent(FrozenModel):
    name: str
    passed: bool
    value: Any
    comparison: str
    threshold: Any
    numerator: int | None = None
    denominator: int | None = None
    zero_denominator_rule: str | None = None


class OperationalGateArtifact(FrozenModel):
    schema_version: int = 1
    passed: bool
    override_allowed: bool = False
    failed_components: tuple[str, ...]
    inputs: OperationalGateInputs
    components: tuple[GateComponent, ...]


def evaluate_operational_gates(
    inputs: OperationalGateInputs,
    *,
    output_path: Path | None = None,
) -> OperationalGateArtifact:
    """Evaluate every preregistered component without an override path."""
    medians = inputs.verified_search_median_tokens
    observed_rate_denominator = inputs.observed_cells or 1

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    components = (
        GateComponent(
            name="complete_grid",
            passed=inputs.observed_cells == inputs.expected_cells,
            value=inputs.observed_cells,
            comparison="==",
            threshold=inputs.expected_cells,
            numerator=inputs.observed_cells,
            denominator=inputs.expected_cells,
        ),
        GateComponent(
            name="unique_paired_cells",
            passed=(
                inputs.observed_paired_cells == inputs.expected_paired_cells
                and inputs.unique_paired_cells == inputs.observed_paired_cells
            ),
            value=inputs.unique_paired_cells,
            comparison="==",
            threshold=inputs.expected_paired_cells,
            numerator=inputs.unique_paired_cells,
            denominator=inputs.expected_paired_cells,
        ),
        GateComponent(
            name="authoritative_usage",
            passed=(
                inputs.observed_cells > 0
                and inputs.authoritative_usage_cells == inputs.observed_cells
            ),
            value=inputs.authoritative_usage_cells / observed_rate_denominator,
            comparison="==",
            threshold=1.0,
            numerator=inputs.authoritative_usage_cells,
            denominator=inputs.observed_cells,
        ),
        GateComponent(
            name="label_leakage",
            passed=inputs.label_leakage_count == 0,
            value=inputs.label_leakage_count,
            comparison="==",
            threshold=0,
        ),
        GateComponent(
            name="budget_overruns",
            passed=inputs.budget_overrun_count == 0,
            value=inputs.budget_overrun_count,
            comparison="==",
            threshold=0,
        ),
        GateComponent(
            name="schema_validity",
            passed=(
                inputs.observed_cells > 0
                and inputs.schema_valid_cells * 100 >= inputs.observed_cells * 99
            ),
            value=inputs.schema_valid_cells / observed_rate_denominator,
            comparison=">=",
            threshold=0.99,
            numerator=inputs.schema_valid_cells,
            denominator=inputs.observed_cells,
        ),
        GateComponent(
            name="unresolved_external_matched_blocks",
            passed=(
                inputs.matched_blocks_total == 0
                or inputs.unresolved_external_matched_blocks * 100
                <= inputs.matched_blocks_total
            ),
            value=(
                inputs.unresolved_external_matched_blocks / inputs.matched_blocks_total
                if inputs.matched_blocks_total
                else 0.0
            ),
            comparison="<=",
            threshold=0.01,
            numerator=inputs.unresolved_external_matched_blocks,
            denominator=inputs.matched_blocks_total,
            zero_denominator_rule="pass_only_when_total_and_unresolved_are_zero",
        ),
        GateComponent(
            name="exact_mechanism_counts",
            passed=(
                dict(inputs.observed_mechanism_counts)
                == dict(inputs.expected_mechanism_counts)
            ),
            value=dict(inputs.observed_mechanism_counts),
            comparison="==",
            threshold=dict(inputs.expected_mechanism_counts),
        ),
        GateComponent(
            name="low_tier_feasibility",
            passed=(
                inputs.low_tier_cases > 0
                and inputs.low_tier_feasible_cases == inputs.low_tier_cases
            ),
            value=rate(inputs.low_tier_feasible_cases, inputs.low_tier_cases),
            comparison="==",
            threshold=1.0,
            numerator=inputs.low_tier_feasible_cases,
            denominator=inputs.low_tier_cases,
        ),
        GateComponent(
            name="verified_search_low_to_middle_token_growth",
            passed=medians["middle"] >= medians["low"] * 1.20,
            value=medians["middle"] / medians["low"] - 1.0,
            comparison=">=",
            threshold=0.20,
        ),
        GateComponent(
            name="verified_search_middle_to_high_token_growth",
            passed=medians["high"] >= medians["middle"] * 1.20,
            value=medians["high"] / medians["middle"] - 1.0,
            comparison=">=",
            threshold=0.20,
        ),
        GateComponent(
            name="easy_monolith_accuracy",
            passed=(
                inputs.easy_monolith_total > 0
                and inputs.easy_monolith_correct * 100
                >= inputs.easy_monolith_total * 90
            ),
            value=rate(inputs.easy_monolith_correct, inputs.easy_monolith_total),
            comparison=">=",
            threshold=0.90,
            numerator=inputs.easy_monolith_correct,
            denominator=inputs.easy_monolith_total,
        ),
        GateComponent(
            name="hard_monolith_accuracy_lower",
            passed=(
                inputs.hard_monolith_total > 0
                and inputs.hard_monolith_correct * 100
                >= inputs.hard_monolith_total * 30
            ),
            value=rate(inputs.hard_monolith_correct, inputs.hard_monolith_total),
            comparison=">=",
            threshold=0.30,
            numerator=inputs.hard_monolith_correct,
            denominator=inputs.hard_monolith_total,
        ),
        GateComponent(
            name="hard_monolith_accuracy_upper",
            passed=(
                inputs.hard_monolith_total > 0
                and inputs.hard_monolith_correct * 100
                <= inputs.hard_monolith_total * 85
            ),
            value=rate(inputs.hard_monolith_correct, inputs.hard_monolith_total),
            comparison="<=",
            threshold=0.85,
            numerator=inputs.hard_monolith_correct,
            denominator=inputs.hard_monolith_total,
        ),
        GateComponent(
            name="checker_specificity",
            passed=(
                inputs.checker_actual_negatives > 0
                and inputs.checker_true_negatives * 100
                >= inputs.checker_actual_negatives * 95
            ),
            value=rate(inputs.checker_true_negatives, inputs.checker_actual_negatives),
            comparison=">=",
            threshold=0.95,
            numerator=inputs.checker_true_negatives,
            denominator=inputs.checker_actual_negatives,
        ),
        GateComponent(
            name="checker_sensitivity",
            passed=(
                inputs.checker_actual_positives > 0
                and inputs.checker_true_positives * 100
                >= inputs.checker_actual_positives * 60
            ),
            value=rate(inputs.checker_true_positives, inputs.checker_actual_positives),
            comparison=">=",
            threshold=0.60,
            numerator=inputs.checker_true_positives,
            denominator=inputs.checker_actual_positives,
        ),
        GateComponent(
            name="correct_to_wrong_repair",
            passed=(
                inputs.correct_first_drafts_repaired > 0
                and inputs.correct_to_wrong_repairs * 100
                <= inputs.correct_first_drafts_repaired * 5
            ),
            value=rate(
                inputs.correct_to_wrong_repairs,
                inputs.correct_first_drafts_repaired,
            ),
            comparison="<=",
            threshold=0.05,
            numerator=inputs.correct_to_wrong_repairs,
            denominator=inputs.correct_first_drafts_repaired,
        ),
        GateComponent(
            name="wrong_first_draft_correction",
            passed=(
                inputs.checker_detected_wrong_first_drafts > 0
                and inputs.wrong_first_drafts_corrected * 100
                >= inputs.checker_detected_wrong_first_drafts * 20
            ),
            value=rate(
                inputs.wrong_first_drafts_corrected,
                inputs.checker_detected_wrong_first_drafts,
            ),
            comparison=">=",
            threshold=0.20,
            numerator=inputs.wrong_first_drafts_corrected,
            denominator=inputs.checker_detected_wrong_first_drafts,
        ),
    )
    failed_components = tuple(
        component.name for component in components if not component.passed
    )
    artifact = OperationalGateArtifact(
        passed=not failed_components,
        failed_components=failed_components,
        inputs=inputs,
        components=components,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact


# Import-only bridges for the Task 5 CLI rebuild. They fail closed rather than
# applying removed domain-specific validation semantics.
def validate_cases(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError("the legacy case validator was removed; migrate to canonical artifacts")


def validate_run(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError("the legacy run validator was removed; migrate to operational gates")


def assert_pilot_gate(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("the legacy pilot gate was removed; migrate to the gate artifact")
