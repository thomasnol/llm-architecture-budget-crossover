from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from budget_crossover.config import ExperimentConfig, load_experiment_config

REPO = Path(__file__).resolve().parents[1]


def test_canonical_config_freezes_three_systems_tiers_and_failure_semantics():
    config = load_experiment_config(REPO / "configs" / "main.yaml")

    assert config.systems == ("monolith", "verified_search", "unverified_search")
    assert config.confirmatory_systems == ("monolith", "verified_search")
    assert config.tiers == ("low", "middle", "high")
    assert config.tier_token_limits == {"low": 4096, "middle": 12288, "high": 32768}
    assert config.model == "gpt-5.4-mini"
    assert config.default_sample_cap == 1000
    assert config.scoring_absolute_tolerance == Decimal("0.000001")
    assert config.scoring_relative_tolerance == Decimal("0.000001")
    assert config.failure_semantics.architecture_failure == "score_incorrect"
    assert config.failure_semantics.infrastructure_failure == "matched_block"
    assert config.failure_semantics.equality_is_crossover is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": "gpt-5.4"}, "exactly gpt-5.4-mini"),
        ({"systems": ("monolith", "verified_search")}, "exactly"),
        ({"tiers": ("low", "high")}, "exactly"),
        ({"default_sample_cap": 1001}, "at most 1000"),
        ({"scoring_absolute_tolerance": Decimal("0.01")}, "strict scoring"),
    ],
)
def test_config_refuses_protocol_substitution(changes: dict, message: str):
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig(**changes)


def test_legacy_hmda_configuration_is_a_migration_error():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentConfig(hmda_year=2024, hmda_source_sha256="0" * 64)
