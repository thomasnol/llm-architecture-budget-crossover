from __future__ import annotations

import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget import BUDGET_TIERS
from .models import SystemName, TierName

ENV_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CANONICAL_SYSTEMS: tuple[SystemName, ...] = (
    "monolith",
    "verified_search",
    "unverified_search",
)
CONFIRMATORY_SYSTEMS: tuple[SystemName, ...] = ("monolith", "verified_search")
CANONICAL_TIERS: tuple[TierName, ...] = ("low", "middle", "high")
EXACT_MODEL = "gpt-5.4-mini"
STRICT_TOLERANCE = Decimal("0.000001")


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        configured = os.getenv(match.group(1))
        return configured if configured else match.group(2)

    return os.path.expandvars(ENV_DEFAULT_PATTERN.sub(replace, value))


def _expand(value: object) -> object:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSnapshotConfig(FrozenConfigModel):
    path: Path
    sha256: str

    @model_validator(mode="after")
    def validate_sha256(self) -> SourceSnapshotConfig:
        if SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("snapshot sha256 must be a lowercase SHA-256 digest")
        return self


class FailureSemantics(FrozenConfigModel):
    invalid_output: Literal["score_incorrect"] = "score_incorrect"
    refusal: Literal["score_incorrect"] = "score_incorrect"
    architecture_tool_failure: Literal["score_incorrect"] = "score_incorrect"
    budget_exhaustion: Literal["score_incorrect"] = "score_incorrect"
    abstention: Literal["score_incorrect"] = "score_incorrect"
    architecture_failure: Literal["score_incorrect"] = "score_incorrect"
    infrastructure_failure: Literal["matched_block"] = "matched_block"
    equality_is_crossover: Literal[False] = False


class RetryPolicy(FrozenConfigModel):
    max_attempts: int = Field(default=5, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=0.75, gt=0)
    maximum_backoff_seconds: float = Field(default=20.0, gt=0)
    permanent_error_circuit_breaker: int = Field(default=3, ge=1)


class ExperimentConfig(FrozenConfigModel):
    experiment_name: str = "canonical-main"
    execution_mode: Literal["gateway", "offline_fixture"] = "gateway"
    seed: int = 20260803
    finqa_snapshot: SourceSnapshotConfig = Field(
        default_factory=lambda: SourceSnapshotConfig(
            path=Path("data/raw/finqa.json"), sha256="0" * 64
        )
    )
    tatqa_snapshot: SourceSnapshotConfig = Field(
        default_factory=lambda: SourceSnapshotConfig(
            path=Path("data/raw/tatqa.json"), sha256="0" * 64
        )
    )
    finance_complex_snapshot: SourceSnapshotConfig | None = Field(
        default_factory=lambda: SourceSnapshotConfig(
            path=Path("data/raw/financecomplexqa.json"), sha256="0" * 64
        )
    )
    prepared_data_dir: Path = Path("data/processed/canonical")
    run_root: Path = Path("experiments/runs")
    development_fit_path: Path = Path("experiments/inputs/development_fit.jsonl")
    pilot_gate_metrics_path: Path = Path("experiments/inputs/pilot_gate_metrics.json")
    systems: tuple[SystemName, ...] = CANONICAL_SYSTEMS
    confirmatory_systems: tuple[SystemName, ...] = CONFIRMATORY_SYSTEMS
    tiers: tuple[TierName, ...] = CANONICAL_TIERS
    model: str = EXACT_MODEL
    deployment: str = EXACT_MODEL
    tokenizer_id: str = "gateway:gpt-5.4-mini:chat"
    tokenizer_sha256: str = "0" * 64
    default_sample_cap: int = Field(default=1000, ge=1)
    development_cases: int = Field(default=100, ge=2)
    operational_pilot_cases: int = Field(default=60, ge=2)
    main_cases: int = Field(default=1000, ge=2)
    easy_reserve_cases: int = Field(default=100, ge=2)
    repetitions: int = Field(default=1, ge=1)
    scoring_absolute_tolerance: Decimal = Field(default=STRICT_TOLERANCE, ge=0)
    scoring_relative_tolerance: Decimal = Field(default=STRICT_TOLERANCE, ge=0)
    failure_semantics: FailureSemantics = Field(default_factory=FailureSemantics)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_concurrency: int = Field(default=4, ge=1)
    expected_finance_complex_cases: int = Field(default=113, ge=1)
    bootstrap_replicates: int = Field(default=5000, ge=1)

    @model_validator(mode="after")
    def validate_protocol(self) -> ExperimentConfig:
        if self.systems != CANONICAL_SYSTEMS:
            raise ValueError(f"systems must be exactly {CANONICAL_SYSTEMS!r}")
        if self.confirmatory_systems != CONFIRMATORY_SYSTEMS:
            raise ValueError(
                f"confirmatory systems must be exactly {CONFIRMATORY_SYSTEMS!r}"
            )
        if self.tiers != CANONICAL_TIERS:
            raise ValueError(f"tiers must be exactly {CANONICAL_TIERS!r}")
        if self.model != EXACT_MODEL or self.deployment != EXACT_MODEL:
            raise ValueError("model and deployment must resolve exactly gpt-5.4-mini")
        if SHA256_PATTERN.fullmatch(self.tokenizer_sha256) is None:
            raise ValueError("tokenizer_sha256 must be a lowercase SHA-256 digest")
        if not self.tokenizer_id.strip():
            raise ValueError("tokenizer_id must be non-empty")
        if self.default_sample_cap > 1000:
            raise ValueError("default sample cap must be at most 1000")
        if self.main_cases > self.default_sample_cap:
            raise ValueError("main cases cannot exceed the default sample cap")
        if any(
            value % 2
            for value in (
                self.development_cases,
                self.operational_pilot_cases,
                self.main_cases,
                self.easy_reserve_cases,
            )
        ):
            raise ValueError("balanced split sizes must be even")
        if (
            self.scoring_absolute_tolerance > STRICT_TOLERANCE
            or self.scoring_relative_tolerance > STRICT_TOLERANCE
        ):
            raise ValueError("strict scoring tolerances cannot exceed 0.000001")
        if self.retry_policy.maximum_backoff_seconds < self.retry_policy.initial_backoff_seconds:
            raise ValueError("maximum retry backoff cannot be below the initial backoff")
        return self

    @property
    def tier_token_limits(self) -> dict[str, int]:
        return {name: BUDGET_TIERS[name].token_limit for name in self.tiers}


def load_experiment_config(path: Path) -> ExperimentConfig:
    load_dotenv(path.resolve().parent.parent / ".env", override=False)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("experiment configuration must be a YAML mapping")
    return ExperimentConfig.model_validate(_expand(payload))
