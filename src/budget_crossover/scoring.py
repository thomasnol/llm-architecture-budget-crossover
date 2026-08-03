from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext

from .models import AnswerSpec, Candidate, FrozenModel, HiddenLabel, PublicCase

_UNSIGNED_NUMBER = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
_STRICT_NUMERIC = re.compile(
    rf"^(?:"
    rf"\((?P<parentheses_currency>[$€£])?(?P<parentheses_number>{_UNSIGNED_NUMBER})"
    rf"(?P<parentheses_percent>%)?\)"
    rf"|(?P<sign>[+-])?(?P<currency>[$€£])?(?P<number>{_UNSIGNED_NUMBER})(?P<percent>%)?"
    rf")$"
)
_EMBEDDED_NUMERIC = re.compile(
    rf"(?<![\w.,$€£%+-])"
    rf"(?:\((?:[$€£])?{_UNSIGNED_NUMBER}%?\)|[+-]?(?:[$€£])?{_UNSIGNED_NUMBER}%?)"
    rf"(?![\w.,%])"
)
_SCALE_EXPONENTS = {
    "ones": 0,
    "thousand": 3,
    "million": 6,
    "billion": 9,
    "percent": -2,
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
    match = _STRICT_NUMERIC.fullmatch(raw)
    if match is None:
        return None
    parenthesized = match.group("parentheses_number") is not None
    number = match.group("parentheses_number") or match.group("number")
    currency_symbol = match.group("parentheses_currency") or match.group("currency")
    currency = _CURRENCY_UNITS.get(currency_symbol) if currency_symbol else None
    is_percent = bool(match.group("parentheses_percent") or match.group("percent"))
    if currency is not None and is_percent:
        return None
    try:
        parsed = Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None
    if parenthesized or match.group("sign") == "-":
        parsed = parsed.copy_negate()
    return parsed, currency, is_percent


def extract_strict_numeric_values(text: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for match in _EMBEDDED_NUMERIC.finditer(text):
        parsed = _parse_numeric(match.group(0))
        if parsed is not None:
            values.append(parsed[0])
    return tuple(values)


def _scale_decimal(value: Decimal, scale: str) -> Decimal:
    parts = value.as_tuple()
    return Decimal((parts.sign, parts.digits, parts.exponent + _SCALE_EXPONENTS[scale]))


def _comparison_precision(*values: Decimal) -> int:
    digit_count = sum(len(value.as_tuple().digits) for value in values)
    exponent_span = sum(abs(value.as_tuple().exponent) for value in values)
    return max(50, digit_count + exponent_span + 10)


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
        scale = "percent"
    else:
        scale = candidate.scale
    return _scale_decimal(value, scale), unit


def normalized_answer_value(answer: AnswerSpec) -> Decimal:
    return _scale_decimal(answer.value, answer.scale)


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

    absolute_tolerance = _scale_decimal(answer.absolute_tolerance, answer.scale)
    with localcontext() as context:
        context.prec = _comparison_precision(
            candidate_value,
            gold_value,
            absolute_tolerance,
            answer.relative_tolerance,
        )
        difference = abs(candidate_value - gold_value)
        relative_tolerance = answer.relative_tolerance * abs(gold_value)
        tolerance = max(absolute_tolerance, relative_tolerance)
        correct = difference <= tolerance
    return ScoringResult(
        correct=correct,
        candidate_value=candidate_value,
        gold_value=gold_value,
        reason="within_tolerance" if correct else "outside_tolerance",
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
