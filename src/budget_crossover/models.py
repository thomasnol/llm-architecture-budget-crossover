from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    case_id: int
    task: str
    company_name: str
    company_metadata: dict[str, Any]
    underwriter_messages: list[str]
    evidence: list[str]
    accepted_reference_answers: list[str]
    canonical_reference: str
    evidence_chars: int
    tool_evidence_count: int
    source_primary_id: int
    source_model: str


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class GatewayResponse(BaseModel):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float
    credential_slot: int
    concurrency_window: float | None = None
    request_id: str | None = None
    raw_finish_reason: str | None = None


class CallRecord(BaseModel):
    stage: str
    token_cap: int
    response: GatewayResponse


class ExperimentResult(BaseModel):
    run_id: str
    case_id: int
    task: str
    architecture: str
    nominal_budget: int
    model: str
    answer_text: str
    parsed_answer: str
    calls: list[CallRecord]
    status: str = "ok"
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def prompt_tokens(self) -> int | None:
        values = [c.response.usage.prompt_tokens for c in self.calls]
        return (
            sum(v for v in values if v is not None) if any(v is not None for v in values) else None
        )

    @property
    def completion_tokens(self) -> int | None:
        values = [c.response.usage.completion_tokens for c in self.calls]
        return (
            sum(v for v in values if v is not None) if any(v is not None for v in values) else None
        )

    @property
    def total_tokens(self) -> int | None:
        values = [c.response.usage.total_tokens for c in self.calls]
        return (
            sum(v for v in values if v is not None) if any(v is not None for v in values) else None
        )

    @property
    def latency_seconds(self) -> float:
        return sum(c.response.latency_seconds for c in self.calls)


class JudgeResult(BaseModel):
    run_id: str
    case_id: int
    architecture: str
    nominal_budget: int
    judge_model: str
    correct: bool
    evidence_score: int = Field(ge=0, le=4)
    unsupported_claims: bool
    rationale: str
    raw_response: str
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float
    status: str = "ok"
