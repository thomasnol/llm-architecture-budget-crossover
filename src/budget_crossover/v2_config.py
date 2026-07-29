from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
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
    evidence_condition: str = "pooled"
    fixed_source_model: str = "o3"
    max_context_chars: int = 64000
    systems: list[str] = Field(
        default_factory=lambda: [
            "direct",
            "checklist",
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
    adjudicator_model: str = "claude-opus-4-6"
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
    bootstrap_replicates: int = 5000
    pilot_min_schema_validity: float = 0.98
    pilot_min_direct_accuracy: float = 0.25
    pilot_max_direct_accuracy: float = 0.90
    pilot_min_system_disagreement: float = 0.10
    model_prices_per_million: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_systems(self) -> V2Config:
        supported = {
            "direct",
            "checklist",
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
        return self


def load_v2_config(path: Path) -> V2Config:
    payload = yaml.safe_load(path.read_text())
    return V2Config.model_validate(_expand(payload))
