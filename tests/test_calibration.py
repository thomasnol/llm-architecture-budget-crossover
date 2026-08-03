import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from budget_crossover.calibration import (
    DevelopmentFitObservation,
    calibrate_run,
    freeze_calibration,
    select_calibration_ceiling,
)


def test_ceiling_advances_only_while_more_than_one_percent_cannot_fit():
    observations = tuple(
        DevelopmentFitObservation(
            case_id=f"case-{index}",
            mandatory_tokens=(20_000 if index == 0 else 8_000),
        )
        for index in range(100)
    )

    selection = select_calibration_ceiling("high", observations)

    assert selection.selected_ceiling == 8192
    assert selection.feasibility_pass is True
    assert [step.ceiling for step in selection.progression] == [8192]
    assert selection.progression[0].cannot_fit_cases == 1
    assert selection.progression[0].cannot_fit_rate == pytest.approx(0.01)


def test_calibration_observations_reject_outcome_and_system_difference_fields():
    for forbidden in ("correct", "accuracy", "system_difference"):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            DevelopmentFitObservation(
                case_id="case-1",
                mandatory_tokens=8_000,
                **{forbidden: 0},
            )


def test_ceiling_progresses_only_along_the_frozen_sequence_and_fails_closed_at_cap():
    selection = select_calibration_ceiling(
        "middle",
        tuple(
            DevelopmentFitObservation(case_id=f"case-{index}", mandatory_tokens=25_000)
            for index in range(100)
        ),
    )

    assert [step.ceiling for step in selection.progression] == [8192, 16384, 24576, 32768]
    assert selection.selected_ceiling == 32768
    assert selection.feasibility_pass is True

    capped = select_calibration_ceiling(
        "high",
        (
            DevelopmentFitObservation(case_id="case-a", mandatory_tokens=70_000),
            DevelopmentFitObservation(case_id="case-b", mandatory_tokens=70_000),
        ),
    )
    assert [step.ceiling for step in capped.progression] == [
        8192,
        16384,
        24576,
        32768,
        49152,
        65536,
    ]
    assert capped.selected_ceiling == 65536
    assert capped.feasibility_pass is False


def test_calibration_freezes_before_pilot_and_cannot_be_overwritten(tmp_path: Path):
    selections = tuple(
        select_calibration_ceiling(
            tier,
            (DevelopmentFitObservation(case_id="case-1", mandatory_tokens=8_000),),
        )
        for tier in ("low", "middle", "high")
    )
    output = tmp_path / "calibration.json"

    frozen = freeze_calibration(selections, output_path=output, pilot_started=False)

    assert frozen.frozen_before_pilot is True
    assert frozen.outcome_fields_available is False
    assert json.loads(output.read_text())["selections"][0]["selected_ceiling"] == 8192
    with pytest.raises(FileExistsError):
        freeze_calibration(selections, output_path=output, pilot_started=False)
    with pytest.raises(RuntimeError, match="before pilot"):
        freeze_calibration(
            selections,
            output_path=tmp_path / "late.json",
            pilot_started=True,
        )


def test_calibration_freeze_requires_low_middle_and_high_tiers(tmp_path: Path):
    low = select_calibration_ceiling(
        "low",
        (DevelopmentFitObservation(case_id="case-1", mandatory_tokens=8_000),),
    )

    with pytest.raises(ValueError, match="low, middle, and high"):
        freeze_calibration(
            (low,),
            output_path=tmp_path / "partial.json",
            pilot_started=False,
        )


def test_legacy_calibration_bridge_fails_closed_with_a_migration_error():
    with pytest.raises(RuntimeError, match="legacy calibration workflow was removed"):
        calibrate_run()
