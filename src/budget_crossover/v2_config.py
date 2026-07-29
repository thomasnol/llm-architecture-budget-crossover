from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

_DEFAULT_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        configured = os.getenv(match.group(1))
        return configured if configured else match.group(2)

    return os.path.expandvars(_DEFAULT_ENV_PATTERN.sub(replace, value))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


class V2Config(BaseModel):
    experiment_name: str
    seed: int = 20260729
    insurance_sample_size: int = 80
    mmlu_sample_size: int = 200
    include_mmlu: bool = True
    exclude_pilot_cases: bool = False
    pilot_insurance_sample_size: int = 12
    pilot_mmlu_sample_size: int = 18
    evidence_condition: str = "pooled"
    fixed_source_model: str = "o3"
    max_context_chars: int = 64000
    systems: list[str] = Field(
        default_factory=lambda: [
            "direct",
            "checklist",
            "strong_direct",
            "self_critique",
            "external_verify",
            "best_of_2",
            "best_of_4",
            "adaptive",
        ]
    )
    generator_model: str = "gpt-5.4-mini"
    verifier_model: str = "gpt-5.4"
    judge_models: list[str] = Field(
        default_factory=lambda: ["gpt-5.4", "claude-sonnet-4-6"]
    )
    generator_max_tokens: int = 512
    critique_max_tokens: int = 384
    verifier_max_tokens: int = 512
    direct_temperature: float = 0.0
    sampling_temperature: float = 0.7
    adaptive_accept_threshold: float = 0.85
    global_case_concurrency: int = 8
    request_timeout_seconds: int = 180
    generation_runtime_hours: float = 4.5
    judging_runtime_hours: float = 1.5
    judge_sample_per_system_dataset: int | None = 30
    bootstrap_replicates: int = 5000
    pilot_min_schema_validity: float = 0.98
    pilot_min_direct_accuracy: float = 0.25
    pilot_max_direct_accuracy: float = 0.90
    pilot_min_system_disagreement: float = 0.10
    pilot_experiment_name: str = "v2_pilot"
    minimum_schema_validity: float = 0.98
    maximum_truncation_rate: float = 0.02
    sesoi_accuracy_difference: float = 0.05
    model_prices_per_million: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_systems(self) -> V2Config:
        supported = {
            "direct",
            "checklist",
            "strong_direct",
            "self_critique",
            "external_verify",
            "best_of_2",
            "best_of_4",
            "adaptive",
        }
        unknown = set(self.systems) - supported
        if unknown:
            raise ValueError(f"unsupported systems: {sorted(unknown)}")
        if self.evidence_condition not in {"pooled", "fixed"}:
            raise ValueError("evidence_condition must be pooled or fixed")
        if len(self.systems) != len(set(self.systems)):
            raise ValueError("systems must not contain duplicates")
        if "direct" not in self.systems:
            raise ValueError("systems must include direct")
        if not 1 <= self.global_case_concurrency <= 8:
            raise ValueError("global_case_concurrency must be between 1 and 8")
        if self.generation_runtime_hours + self.judging_runtime_hours > 6.25:
            raise ValueError(
                "main generation plus judging runtime must leave room for the "
                "1.25-hour pilot inside the eight-hour study limit"
            )
        if (
            self.judge_sample_per_system_dataset is not None
            and self.judge_sample_per_system_dataset < 1
        ):
            raise ValueError("judge_sample_per_system_dataset must be positive or null")
        for value, name in [
            (self.minimum_schema_validity, "minimum_schema_validity"),
            (self.maximum_truncation_rate, "maximum_truncation_rate"),
            (self.sesoi_accuracy_difference, "sesoi_accuracy_difference"),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


def load_v2_config(path: Path) -> V2Config:
    load_dotenv(path.resolve().parent.parent / ".env", override=False)
    payload = yaml.safe_load(path.read_text())
    return V2Config.model_validate(_expand(payload))
