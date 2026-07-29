from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]+))?\}")


class ExperimentConfig(BaseModel):
    experiment_name: str
    seed: int = 20260728
    sample_size: int = Field(ge=1, le=80)
    task_quotas: dict[str, int] | None = None
    budgets: list[int]
    architectures: list[str]
    generator_model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_context_chars: int = Field(default=72000, ge=8000)
    request_timeout_seconds: float = Field(default=180, gt=0)
    global_case_concurrency: int = Field(default=8, ge=1, le=16)
    generation_runtime_hours: float = Field(default=4.75, gt=0, le=8.0)
    judging_runtime_hours: float = Field(default=3.0, gt=0, le=8.0)
    judge_models: list[str]
    adjudicator_model: str
    bootstrap_replicates: int = Field(default=5000, ge=100)

    @field_validator("judging_runtime_hours")
    @classmethod
    def validate_total_runtime(cls, value: float, info) -> float:
        generation = info.data.get("generation_runtime_hours", 0.0)
        if generation + value > 8.0:
            raise ValueError("generation_runtime_hours + judging_runtime_hours must be <= 8")
        return value

    @field_validator("task_quotas")
    @classmethod
    def validate_task_quotas(cls, value: dict[str, int] | None) -> dict[str, int] | None:
        if value is not None and any(count < 1 for count in value.values()):
            raise ValueError("task quota values must be positive")
        return value

    @field_validator("budgets")
    @classmethod
    def validate_budgets(cls, value: list[int]) -> list[int]:
        if sorted(set(value)) != value:
            raise ValueError("budgets must be sorted and unique")
        if value[0] < 256 or value[-1] > 12000:
            raise ValueError("budgets must remain in [256, 12000]")
        return value

    @field_validator("architectures")
    @classmethod
    def validate_architectures(cls, value: list[str]) -> list[str]:
        supported = {"direct", "self_critique", "debate"}
        unknown = set(value) - supported
        if unknown:
            raise ValueError(f"unsupported architectures: {sorted(unknown)}")
        return value


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            expanded = os.getenv(name) or match.group(2)
            if not expanded:
                raise ValueError(f"required environment variable {name} is not set")
            return expanded

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> ExperimentConfig:
    load_dotenv()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(_expand_env(raw))
