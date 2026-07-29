from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .models import CallRecord


class V2Case(BaseModel):
    case_id: str
    dataset: str
    task: str
    question: str
    context: str
    output_schema: dict[str, Any]
    gold_decision: dict[str, Any]
    evidence_condition: str = "canonical"
    evidence_chars: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class V2Generation(BaseModel):
    run_id: str
    case_id: str
    dataset: str
    task: str
    system: str
    replicate: int = 0
    generator_model: str
    verifier_model: str | None = None
    answer_text: str
    parsed_decision: dict[str, Any] | None = None
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
    def latency_seconds(self) -> float:
        # Parallel candidate calls contribute their observed API time. This summed
        # quantity is deliberately named API latency in the analysis; end-to-end
        # wall time is recorded separately by the runner.
        return sum(call.response.latency_seconds for call in self.calls)


class V2Judgment(BaseModel):
    run_id: str
    case_id: str
    judge_model: str
    semantically_correct: bool
    evidence_score: int = Field(ge=0, le=4)
    unsupported_claims: bool
    rationale: str
    raw_response: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float
    status: str = "ok"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
