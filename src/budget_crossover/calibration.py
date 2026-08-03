from __future__ import annotations

"""Outcome-free development calibration for frozen tier ceilings."""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from .models import FrozenModel, TierName

CALIBRATION_CEILINGS = (8192, 16384, 24576, 32768, 49152, 65536)


class DevelopmentFitObservation(FrozenModel):
    """The only development signal calibration may observe."""

    case_id: str
    mandatory_tokens: int = Field(ge=0)


class CalibrationStep(FrozenModel):
    ceiling: int
    development_cases: int
    cannot_fit_cases: int
    cannot_fit_rate: float


class CalibrationSelection(FrozenModel):
    tier: TierName
    selected_ceiling: int
    feasibility_pass: bool
    progression: tuple[CalibrationStep, ...]


class FrozenCalibration(FrozenModel):
    schema_version: int = 1
    frozen_before_pilot: bool
    outcome_fields_available: bool
    selections: tuple[CalibrationSelection, ...]


def select_calibration_ceiling(
    tier: TierName,
    observations: Sequence[DevelopmentFitObservation],
) -> CalibrationSelection:
    """Advance only when more than 1% of development cases cannot fit."""
    if not observations:
        raise ValueError("development calibration requires at least one case")
    if len({row.case_id for row in observations}) != len(observations):
        raise ValueError("development calibration case IDs must be unique")

    progression: list[CalibrationStep] = []
    selected = CALIBRATION_CEILINGS[-1]
    feasibility_pass = False
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
        selected = ceiling
        if cannot_fit_rate <= 0.01:
            feasibility_pass = True
            break

    return CalibrationSelection(
        tier=tier,
        selected_ceiling=selected,
        feasibility_pass=feasibility_pass,
        progression=tuple(progression),
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
    if not selections:
        raise ValueError("at least one tier calibration selection is required")
    if len({selection.tier for selection in selections}) != len(selections):
        raise ValueError("calibration selections must contain unique tiers")
    if {selection.tier for selection in selections} != {"low", "middle", "high"}:
        raise ValueError("calibration must freeze low, middle, and high tiers together")
    if any(not selection.feasibility_pass for selection in selections):
        raise RuntimeError("cannot freeze a tier that failed mandatory-action feasibility")
    artifact = FrozenCalibration(
        frozen_before_pilot=True,
        outcome_fields_available=False,
        selections=tuple(selections),
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
