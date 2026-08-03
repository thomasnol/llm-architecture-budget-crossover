from __future__ import annotations

"""Non-overridable operational gates for the canonical experiment."""

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import FrozenDict, FrozenModel
from .runner import CellKey


def _freeze_gate_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("gate telemetry numbers must be finite")
    if isinstance(value, dict):
        return FrozenDict(
            {key: _freeze_gate_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_gate_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_gate_value(item) for item in value)
    return value


class OperationalGateInputs(FrozenModel):
    expected_cell_keys: tuple[CellKey, ...]
    observed_cell_keys: tuple[CellKey, ...]
    authoritative_usage_cells: int = Field(ge=0)
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

    @field_validator(
        "expected_mechanism_counts",
        "observed_mechanism_counts",
        "verified_search_median_tokens",
        mode="after",
    )
    @classmethod
    def freeze_mappings(cls, value: dict[str, Any]) -> FrozenDict:
        return _freeze_gate_value(value)

    @model_validator(mode="after")
    def validate_counts(self) -> OperationalGateInputs:
        if not self.expected_cell_keys:
            raise ValueError("expected cell keys must be nonempty")
        if len(set(self.expected_cell_keys)) != len(self.expected_cell_keys):
            raise ValueError("expected cell keys must be unique")
        observed_cells = len(self.observed_cell_keys)
        bounded = (
            (self.authoritative_usage_cells, observed_cells),
            (self.schema_valid_cells, observed_cells),
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


def _cell_key_text(key: CellKey) -> str:
    return f"{key.case_id}:{key.system}:{key.tier}:{key.repetition}"


def _sorted_cell_key_text(keys: set[CellKey]) -> tuple[str, ...]:
    return tuple(
        _cell_key_text(key)
        for key in sorted(
            keys,
            key=lambda key: (key.case_id, key.system, key.tier, key.repetition),
        )
    )


class GateComponent(FrozenModel):
    name: str
    passed: bool
    value: Any
    comparison: str
    threshold: Any
    numerator: int | None = None
    denominator: int | None = None
    zero_denominator_rule: str | None = None

    @classmethod
    def model_construct(
        cls,
        _fields_set: set[str] | None = None,
        **values: Any,
    ) -> GateComponent:
        """Retain Pydantic's API without exposing its validation bypass."""
        del _fields_set
        return cls.model_validate(values)

    @field_validator("value", "threshold", mode="after")
    @classmethod
    def freeze_nested_values(cls, value: Any) -> Any:
        return _freeze_gate_value(value)


class OperationalGateArtifact(FrozenModel):
    schema_version: Literal[1] = 1
    passed: bool
    override_allowed: Literal[False] = False
    failed_components: tuple[str, ...]
    inputs: OperationalGateInputs
    components: tuple[GateComponent, ...]

    @classmethod
    def model_construct(
        cls,
        _fields_set: set[str] | None = None,
        **values: Any,
    ) -> OperationalGateArtifact:
        """Retain Pydantic's API without exposing its validation bypass."""
        del _fields_set
        return cls.model_validate(values)

    @model_validator(mode="after")
    def validate_derived_verdict(self) -> OperationalGateArtifact:
        inputs = OperationalGateInputs.model_validate(
            self.inputs.model_dump(mode="python", round_trip=True)
        )
        canonical_components = _evaluate_gate_components(inputs)
        if self.components != canonical_components:
            raise ValueError("gate components must be derived from the typed input snapshot")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "components", canonical_components)
        names = tuple(component.name for component in canonical_components)
        if len(names) != len(set(names)):
            raise ValueError("operational gate component names must be unique")
        derived_failures = tuple(
            component.name for component in canonical_components if not component.passed
        )
        if self.failed_components != derived_failures:
            raise ValueError("failed components must be derived from component verdicts")
        if self.passed is not (not derived_failures):
            raise ValueError("gate pass must be derived from component verdicts")
        return self


def _evaluate_gate_components(
    inputs: OperationalGateInputs,
) -> tuple[GateComponent, ...]:
    medians = inputs.verified_search_median_tokens
    expected_keys = set(inputs.expected_cell_keys)
    observed_keys = set(inputs.observed_cell_keys)
    missing_keys = expected_keys - observed_keys
    unexpected_keys = observed_keys - expected_keys
    expected_cells = len(inputs.expected_cell_keys)
    observed_cells = len(inputs.observed_cell_keys)
    unique_cells = len(observed_keys)
    observed_rate_denominator = observed_cells or 1

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    components = (
        GateComponent(
            name="complete_grid",
            passed=not missing_keys and not unexpected_keys,
            value={
                "missing_cell_keys": _sorted_cell_key_text(missing_keys),
                "unexpected_cell_keys": _sorted_cell_key_text(unexpected_keys),
            },
            comparison="==",
            threshold={"missing_cell_keys": (), "unexpected_cell_keys": ()},
            numerator=len(expected_keys & observed_keys),
            denominator=expected_cells,
        ),
        GateComponent(
            name="unique_paired_cells",
            passed=unique_cells == observed_cells,
            value=unique_cells,
            comparison="==",
            threshold=observed_cells,
            numerator=unique_cells,
            denominator=observed_cells,
        ),
        GateComponent(
            name="authoritative_usage",
            passed=(
                observed_cells > 0
                and inputs.authoritative_usage_cells == observed_cells
            ),
            value=inputs.authoritative_usage_cells / observed_rate_denominator,
            comparison="==",
            threshold=1.0,
            numerator=inputs.authoritative_usage_cells,
            denominator=observed_cells,
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
                observed_cells > 0
                and inputs.schema_valid_cells * 100 >= observed_cells * 99
            ),
            value=inputs.schema_valid_cells / observed_rate_denominator,
            comparison=">=",
            threshold=0.99,
            numerator=inputs.schema_valid_cells,
            denominator=observed_cells,
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
    return components


def evaluate_operational_gates(
    inputs: OperationalGateInputs,
    *,
    output_path: Path | None = None,
) -> OperationalGateArtifact:
    """Evaluate every preregistered component without an override path."""
    components = _evaluate_gate_components(inputs)
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
