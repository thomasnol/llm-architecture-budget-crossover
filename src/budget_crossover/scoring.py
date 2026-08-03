from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .models import AnswerSpec, Candidate, FrozenModel, HiddenLabel, PublicCase

_NUMBER = re.compile(
    r"^(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)$"
)
_SCALE_FACTORS = {
    "ones": Decimal(1),
    "thousand": Decimal(1000),
    "million": Decimal(1000000),
    "billion": Decimal(1000000000),
    "percent": Decimal("0.01"),
}
_CURRENCY_UNITS = {"$": "usd", "€": "eur", "£": "gbp"}
_UNIT_ALIASES = {
    "$": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "usd": "usd",
    "€": "eur",
    "euro": "eur",
    "euros": "eur",
    "eur": "eur",
    "£": "gbp",
    "pound": "gbp",
    "pounds": "gbp",
    "gbp": "gbp",
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
}


class ScoringResult(FrozenModel):
    correct: bool
    candidate_value: Decimal | None
    gold_value: Decimal
    reason: str


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def normalize_unit(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    return _UNIT_ALIASES.get(normalized, normalized)


def _parse_numeric(value: str) -> tuple[Decimal, str | None, bool] | None:
    raw = value.strip()
    if not raw or any(character.isspace() for character in raw):
        return None

    negative_parentheses = raw.startswith("(") and raw.endswith(")")
    if negative_parentheses:
        raw = raw[1:-1]
    elif "(" in raw or ")" in raw:
        return None

    sign = Decimal(-1) if negative_parentheses else Decimal(1)
    if raw[:1] in {"+", "-"}:
        if raw[0] == "-":
            sign *= -1
        raw = raw[1:]

    currency = None
    if raw[:1] in _CURRENCY_UNITS:
        currency = _CURRENCY_UNITS[raw[0]]
        raw = raw[1:]
        if raw[:1] in {"+", "-"}:
            if raw[0] == "-":
                sign *= -1
            raw = raw[1:]

    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1]

    match = _NUMBER.fullmatch(raw)
    if match is None:
        return None
    try:
        parsed = Decimal(match.group("number").replace(",", "")) * sign
    except InvalidOperation:
        return None
    return parsed, currency, is_percent


def normalized_candidate_value(candidate: Candidate) -> tuple[Decimal, str | None] | None:
    parsed = _parse_numeric(candidate.value)
    if parsed is None:
        return None
    value, currency, is_percent = parsed
    unit = normalize_unit(candidate.unit)
    if currency is not None:
        if unit is not None and unit != currency:
            return None
        unit = currency
    if is_percent:
        if unit is not None and unit != "percent":
            return None
        if candidate.scale not in {"ones", "percent"}:
            return None
        unit = "percent"
        factor = _SCALE_FACTORS["percent"]
    else:
        factor = _SCALE_FACTORS[candidate.scale]
    return value * factor, unit


def normalized_answer_value(answer: AnswerSpec) -> Decimal:
    return answer.value * _SCALE_FACTORS[answer.scale]


def score_candidate(candidate: Candidate, answer: AnswerSpec) -> ScoringResult:
    gold_value = normalized_answer_value(answer)
    parsed = normalized_candidate_value(candidate)
    if parsed is None:
        return ScoringResult(
            correct=False,
            candidate_value=None,
            gold_value=gold_value,
            reason="invalid_candidate_value",
        )
    candidate_value, candidate_unit = parsed

    expected_unit = normalize_unit(answer.unit)
    if expected_unit is not None and candidate_unit != expected_unit:
        return ScoringResult(
            correct=False,
            candidate_value=candidate_value,
            gold_value=gold_value,
            reason="unit_mismatch",
        )
    if answer.entity is not None and (
        candidate.entity is None
        or _normalize_text(candidate.entity) != _normalize_text(answer.entity)
    ):
        return ScoringResult(
            correct=False,
            candidate_value=candidate_value,
            gold_value=gold_value,
            reason="entity_mismatch",
        )
    if answer.period is not None and (
        candidate.period is None
        or _normalize_text(candidate.period) != _normalize_text(answer.period)
    ):
        return ScoringResult(
            correct=False,
            candidate_value=candidate_value,
            gold_value=gold_value,
            reason="period_mismatch",
        )

    difference = abs(candidate_value - gold_value)
    absolute_tolerance = answer.absolute_tolerance * _SCALE_FACTORS[answer.scale]
    relative_tolerance = answer.relative_tolerance * abs(gold_value)
    tolerance = max(absolute_tolerance, relative_tolerance)
    return ScoringResult(
        correct=difference <= tolerance,
        candidate_value=candidate_value,
        gold_value=gold_value,
        reason="within_tolerance" if difference <= tolerance else "outside_tolerance",
    )


def serialize_gold_oracle(public_case: PublicCase, hidden_label: HiddenLabel) -> Candidate:
    if public_case.case_id != hidden_label.case_id:
        raise ValueError("public and hidden case_id values must match before oracle serialization")
    answer = hidden_label.answer
    return Candidate(
        value=format(answer.value, "f"),
        unit=answer.unit,
        scale=answer.scale,
        entity=answer.entity,
        period=answer.period,
        expression=hidden_label.gold_derivation,
        citations=hidden_label.gold_support_ids,
    )
