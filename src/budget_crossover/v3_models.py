from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .models import CallRecord


class V3Case(BaseModel):
    case_id: str
    pair_id: str
    counterfactual_variant: str
    dataset: str = "hmda_policy_sandbox"
    source_row_id: str
    state: str
    historical_action: str
    policy_decision: str
    policy_reason_codes: list[str]
    documents: dict[str, str]
    protected_attributes: dict[str, str]
    changed_protected_attribute: str
    complexity: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def gold_decision(self) -> dict[str, Any]:
        return {
            "decision": self.policy_decision,
            "reason_codes": sorted(self.policy_reason_codes),
        }

    @property
    def evidence_chars(self) -> int:
        return sum(len(value) for value in self.documents.values())


class V3Generation(BaseModel):
    run_id: str
    case_id: str
    pair_id: str
    counterfactual_variant: str
    system: str
    token_budget: int
    model: str
    supervisor_model: str | None = None
    answer_text: str = ""
    parsed_decision: dict[str, Any] | None = None
    confidence: float | None = None
    rationale: str = ""
    calls: list[CallRecord] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    wall_time_seconds: float | None = None
    status: str = "ok"
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def prompt_tokens(self) -> int | None:
        values = [call.response.usage.prompt_tokens for call in self.calls]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    @property
    def completion_tokens(self) -> int | None:
        values = [call.response.usage.completion_tokens for call in self.calls]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    @property
    def total_tokens(self) -> int | None:
        values = [call.response.usage.total_tokens for call in self.calls]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    @property
    def summed_api_latency_seconds(self) -> float:
        return sum(call.response.latency_seconds for call in self.calls)
