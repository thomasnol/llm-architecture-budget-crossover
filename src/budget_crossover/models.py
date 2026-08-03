from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Scale = Literal["ones", "thousand", "million", "billion", "percent"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep  # Validation reconstructs the complete immutable object graph.
        values = self.model_dump(mode="python", round_trip=True)
        if update:
            values.update(update)
        return type(self).model_validate(values)


class FrozenDict(dict[str, Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("frozen mapping does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


class EvidenceItem(FrozenModel):
    evidence_id: str
    document_id: str
    kind: str
    text: str
    table_id: str | None = None
    headers: tuple[str, ...] = ()
    row_label: str | None = None
    unit: str | None = None
    scale: Scale | None = None
    entity: str | None = None
    period: str | None = None
    ordinal: int = Field(ge=0)


class DescriptiveMetadata(FrozenModel):
    company: str | None = None
    title: str | None = None
    section: str | None = None
    language: str | None = None
    filing_type: str | None = None
    industry: str | None = None
    tags: tuple[str, ...] = ()


class PublicCase(FrozenModel):
    case_id: str
    dataset: str
    document_id: str
    question: str
    evidence: tuple[EvidenceItem, ...]
    stratum: str
    metadata: DescriptiveMetadata = Field(default_factory=DescriptiveMetadata)

    @model_validator(mode="after")
    def validate_public_boundary(self) -> PublicCase:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within a public case")
        if any(item.document_id != self.document_id for item in self.evidence):
            raise ValueError("evidence document_id must match the public case document_id")
        return self


class AnswerSpec(FrozenModel):
    value: Decimal
    unit: str | None
    scale: Scale = "ones"
    entity: str | None
    period: str | None
    absolute_tolerance: Decimal = Field(default=Decimal(0), ge=0)
    relative_tolerance: Decimal = Field(default=Decimal(0), ge=0)


class HiddenLabel(FrozenModel):
    case_id: str
    answer: AnswerSpec
    gold_derivation: str
    gold_support_ids: tuple[str, ...]
    source_lineage: tuple[str, ...]


class Candidate(FrozenModel):
    value: str
    unit: str | None
    scale: Scale = "ones"
    entity: str | None
    period: str | None
    expression: str | None
    citations: tuple[str, ...]


class CheckFinding(FrozenModel):
    code: str
    message: str
    evidence_ids: tuple[str, ...] = ()


class CheckResult(FrozenModel):
    passed: bool
    findings: tuple[CheckFinding, ...] = ()
    evaluated_expression: Decimal | None = None


class Reservation(FrozenModel):
    reservation_id: str
    prompt_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=0)

    @property
    def reserved_tokens(self) -> int:
        return self.prompt_tokens + self.max_output_tokens


class CallEvent(FrozenModel):
    stage: str
    reservation: Reservation
    usage: Usage | None = None
    model: str | None = None
    request_id: str | None = None
    protocol_violation: bool = False


class MechanismTrace(FrozenModel):
    planned_queries: tuple[str, ...] = ()
    actual_queries: tuple[str, ...] = ()
    query_hashes: tuple[str, ...] = ()
    retrieval_pre_truncation_ids: tuple[str, ...] = ()
    retrieval_post_truncation_ids: tuple[str, ...] = ()
    candidate_token_cap: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    checks: tuple[CheckResult, ...] = ()
    repair_attempted: bool = False
    accepted_candidate_index: int | None = Field(default=None, ge=0)
    answer_changed: bool | None = None
    call_events: tuple[CallEvent, ...] = ()
    realized_tokens: int = Field(default=0, ge=0)
    exit_reason: str


class CellResult(FrozenModel):
    case_id: str
    system: str
    tier: str
    repetition: int = Field(ge=0)
    status: str
    candidate: Candidate | None
    trace: MechanismTrace


class RunManifest(FrozenModel):
    run_id: str
    resolved_config: dict[str, Any]
    artifact_hashes: dict[str, str]
    expected_cell_keys: tuple[str, ...]

    @field_validator("resolved_config", "artifact_hashes", mode="after")
    @classmethod
    def freeze_mappings(cls, value: dict[str, Any]) -> FrozenDict:
        return _freeze(value)


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


class Usage(FrozenModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Usage:
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")
        return self

    @property
    def authoritative_total(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


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
