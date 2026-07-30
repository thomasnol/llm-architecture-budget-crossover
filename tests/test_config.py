from pathlib import Path

import pytest
from pydantic import ValidationError

from budget_crossover.config import ExperimentConfig, load_experiment_config

REPO = Path(__file__).resolve().parents[1]


def test_architecture_and_routing_systems_cannot_be_mixed():
    with pytest.raises(ValidationError, match="unsupported systems"):
        ExperimentConfig(
            experiment_name="mixed",
            study_kind="architecture",
            hmda_source_sha256="0" * 64,
            systems=["monolith", "selective_supervisor"],
        )


def test_routing_study_requires_all_three_model_routing_baselines():
    with pytest.raises(ValidationError, match="routing study requires"):
        ExperimentConfig(
            experiment_name="routing",
            study_kind="routing",
            hmda_source_sha256="0" * 64,
            systems=["always_primary", "selective_supervisor"],
        )


def test_main_config_uses_one_repetition_and_7680_cells():
    config = load_experiment_config(REPO / "configs" / "main.yaml")

    case_count = config.base_application_count * 2
    expected_cells = (
        case_count * len(config.systems) * len(config.token_budgets) * config.repetitions
    )

    assert config.repetitions == 1
    assert expected_cells == 7_680
