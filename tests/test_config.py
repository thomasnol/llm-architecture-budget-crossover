import pytest
from pydantic import ValidationError

from budget_crossover.config import ExperimentConfig


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
