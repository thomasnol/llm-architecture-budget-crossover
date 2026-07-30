from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

_DEFAULT_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")

SUPPORTED_SYSTEMS = {
    "monolith",
    "strong_monolith",
    "retrieval",
    "committee",
    "guardrail",
    "adaptive",
}


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


class V3Config(BaseModel):
    experiment_name: str
    execution_mode: Literal["gateway", "offline_smoke"] = "gateway"
    seed: int = 20260729
    hmda_year: int = 2024
    hmda_states: list[str] = Field(default_factory=lambda: ["DC", "ND", "VT", "WY"])
    hmda_raw_filename: str = "hmda_2024_dc_nd_vt_wy.csv"
    hmda_source_sha256: str
    base_application_count: int = 96
    exclude_pilot_applications: bool = True
    pilot_base_application_count: int = 24
    pilot_experiment_name: str = "v3_pilot"
    token_budgets: list[int] = Field(default_factory=lambda: [2048, 4096, 8192])
    systems: list[str] = Field(
        default_factory=lambda: [
            "monolith",
            "strong_monolith",
            "retrieval",
            "committee",
            "guardrail",
            "adaptive",
        ]
    )
    generator_model: str = "gpt-5.4-mini"
    supervisor_model: str = "claude-sonnet-4-6"
    stage_max_output_tokens: int = 256
    minimum_call_output_tokens: int = 64
    prompt_chars_per_token: float = 2.5
    prompt_token_overhead: int = 64
    direct_temperature: float = 0.0
    specialist_temperature: float = 0.2
    adaptive_confidence_threshold: float = 0.90
    request_timeout_seconds: int = 180
    runtime_hours: float = 6.0
    bootstrap_replicates: int = 5000
    minimum_high_budget_schema_validity: float = 0.95
    maximum_budget_overrun_rate: float = 0.01
    sesoi_accuracy_difference: float = 0.05
    model_prices_per_million: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_design(self) -> V3Config:
        unknown = set(self.systems) - SUPPORTED_SYSTEMS
        if unknown:
            raise ValueError(f"unsupported systems: {sorted(unknown)}")
        if "monolith" not in self.systems or "adaptive" not in self.systems:
            raise ValueError("systems must include monolith and adaptive")
        if len(self.systems) != len(set(self.systems)):
            raise ValueError("systems must not contain duplicates")
        if not self.hmda_states or any(
            not re.fullmatch(r"[A-Z]{2}", value) for value in self.hmda_states
        ):
            raise ValueError("hmda_states must contain two-letter uppercase codes")
        if len(self.hmda_states) != len(set(self.hmda_states)):
            raise ValueError("hmda_states must not contain duplicates")
        if not re.fullmatch(r"[0-9a-f]{64}", self.hmda_source_sha256):
            raise ValueError("hmda_source_sha256 must be a lowercase SHA-256 digest")
        if not self.token_budgets or self.token_budgets != sorted(set(self.token_budgets)):
            raise ValueError("token_budgets must be unique and strictly increasing")
        if self.token_budgets[0] < 512:
            raise ValueError("the smallest token budget must be at least 512")
        if self.base_application_count < 1 or self.pilot_base_application_count < 1:
            raise ValueError("application counts must be positive")
        if self.exclude_pilot_applications and self.experiment_name == self.pilot_experiment_name:
            raise ValueError("the pilot cannot exclude itself")
        if not self.generator_model.strip() or not self.supervisor_model.strip():
            raise ValueError("generator and supervisor model IDs must be non-empty")
        if not 0 < self.minimum_call_output_tokens <= self.stage_max_output_tokens:
            raise ValueError("call output limits are inconsistent")
        if self.prompt_chars_per_token <= 0 or self.prompt_token_overhead < 0:
            raise ValueError("prompt estimation parameters must be positive")
        if self.runtime_hours <= 0 or self.runtime_hours > 8:
            raise ValueError("runtime_hours must be in (0, 8]")
        for value, name in [
            (self.adaptive_confidence_threshold, "adaptive_confidence_threshold"),
            (
                self.minimum_high_budget_schema_validity,
                "minimum_high_budget_schema_validity",
            ),
            (self.maximum_budget_overrun_rate, "maximum_budget_overrun_rate"),
            (self.sesoi_accuracy_difference, "sesoi_accuracy_difference"),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


def load_v3_config(path: Path) -> V3Config:
    load_dotenv(path.resolve().parent.parent / ".env", override=False)
    payload = yaml.safe_load(path.read_text())
    return V3Config.model_validate(_expand(payload))
