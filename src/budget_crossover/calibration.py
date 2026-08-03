from __future__ import annotations

"""Outcome-free development calibration for frozen tier ceilings."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .models import FrozenModel, TierName

CALIBRATION_CEILINGS = (8192, 16384, 24576, 32768, 49152, 65536)


class DevelopmentFitObservation(FrozenModel):
    """The only development signal calibration may observe."""

    case_id: str
    mandatory_tokens: int = Field(ge=0)


class CalibrationStep(FrozenModel):
    ceiling: int = Field(gt=0)
    development_cases: int = Field(gt=0)
    cannot_fit_cases: int = Field(ge=0)
    cannot_fit_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_rate(self) -> CalibrationStep:
        if self.cannot_fit_cases > self.development_cases:
            raise ValueError("cannot-fit cases cannot exceed development cases")
        if self.cannot_fit_rate != self.cannot_fit_cases / self.development_cases:
            raise ValueError("cannot-fit rate must equal the recorded case counts")
        return self


class CalibrationSelection(FrozenModel):
    tier: TierName
    selected_ceiling: int
    feasibility_pass: bool
    progression: tuple[CalibrationStep, ...]
    observations: tuple[DevelopmentFitObservation, ...]

    @model_validator(mode="after")
    def validate_selection(self) -> CalibrationSelection:
        if not self.observations:
            raise ValueError("calibration selection requires development observations")
        if len({row.case_id for row in self.observations}) != len(self.observations):
            raise ValueError("development calibration case IDs must be unique")
        expected = _calibration_progression(self.observations)
        if self.progression != expected:
            raise ValueError("calibration progression is not the exact examined prefix")
        if self.selected_ceiling != expected[-1].ceiling:
            raise ValueError("selected ceiling must equal the terminal examined ceiling")
        expected_feasibility = expected[-1].cannot_fit_rate <= 0.01
        if self.feasibility_pass is not expected_feasibility:
            raise ValueError("feasibility must be derived from the terminal failure rate")
        return self


class FrozenCalibration(FrozenModel):
    schema_version: int = 1
    frozen_before_pilot: Literal[True] = True
    outcome_fields_available: Literal[False] = False
    selections: tuple[CalibrationSelection, ...]


def _calibration_progression(
    observations: Sequence[DevelopmentFitObservation],
) -> tuple[CalibrationStep, ...]:
    progression: list[CalibrationStep] = []
    for ceiling in CALIBRATION_CEILINGS:
        cannot_fit = sum(row.mandatory_tokens > ceiling for row in observations)
        cannot_fit_rate = cannot_fit / len(observations)
        progression.append(
            CalibrationStep(
                ceiling=ceiling,
                development_cases=len(observations),
                cannot_fit_cases=cannot_fit,
                cannot_fit_rate=cannot_fit_rate,
            )
        )
        if cannot_fit_rate <= 0.01:
            break
    return tuple(progression)


def select_calibration_ceiling(
    tier: TierName,
    observations: Sequence[DevelopmentFitObservation],
) -> CalibrationSelection:
    """Advance only when more than 1% of development cases cannot fit."""
    if not observations:
        raise ValueError("development calibration requires at least one case")
    if len({row.case_id for row in observations}) != len(observations):
        raise ValueError("development calibration case IDs must be unique")

    frozen_observations = tuple(observations)
    progression = _calibration_progression(frozen_observations)

    return CalibrationSelection(
        tier=tier,
        selected_ceiling=progression[-1].ceiling,
        feasibility_pass=progression[-1].cannot_fit_rate <= 0.01,
        progression=progression,
        observations=frozen_observations,
    )


def freeze_calibration(
    selections: Sequence[CalibrationSelection],
    *,
    output_path: Path,
    pilot_started: bool,
) -> FrozenCalibration:
    """Write the immutable, outcome-free calibration artifact exactly once."""
    if pilot_started:
        raise RuntimeError("calibration must be frozen before pilot")
    validated = tuple(
        CalibrationSelection.model_validate(
            selection.model_dump(mode="python", round_trip=True)
        )
        for selection in selections
    )
    if not validated:
        raise ValueError("at least one tier calibration selection is required")
    if len({selection.tier for selection in validated}) != len(validated):
        raise ValueError("calibration selections must contain unique tiers")
    if {selection.tier for selection in validated} != {"low", "middle", "high"}:
        raise ValueError("calibration must freeze low, middle, and high tiers together")
    if any(not selection.feasibility_pass for selection in validated):
        raise RuntimeError("cannot freeze a tier that failed mandatory-action feasibility")
    artifact = FrozenCalibration(
        selections=validated,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(artifact.model_dump(mode="json"), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact


# Import-only bridge for the Task 5 CLI rebuild.
def calibrate_run(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError(
        "the legacy calibration workflow was removed; migrate to outcome-free calibration"
    )
