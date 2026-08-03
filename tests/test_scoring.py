from decimal import Decimal

import pytest

from budget_crossover.models import AnswerSpec, Candidate, HiddenLabel, PublicCase
from budget_crossover.scoring import score_candidate, serialize_gold_oracle


def _answer(
    value: str,
    *,
    unit: str | None = None,
    scale: str = "ones",
    entity: str | None = None,
    period: str | None = None,
    absolute_tolerance: str = "0",
    relative_tolerance: str = "0",
) -> AnswerSpec:
    return AnswerSpec(
        value=Decimal(value),
        unit=unit,
        scale=scale,
        entity=entity,
        period=period,
        absolute_tolerance=Decimal(absolute_tolerance),
        relative_tolerance=Decimal(relative_tolerance),
    )


def _candidate(
    value: str,
    *,
    unit: str | None = None,
    scale: str = "ones",
    entity: str | None = None,
    period: str | None = None,
) -> Candidate:
    return Candidate(
        value=value,
        unit=unit,
        scale=scale,
        entity=entity,
        period=period,
        expression=None,
        citations=(),
    )


@pytest.mark.parametrize(
    ("candidate", "answer", "normalized"),
    [
        (_candidate("1,200"), _answer("1200"), Decimal(1200)),
        (_candidate("1.2", scale="thousand"), _answer("1200"), Decimal(1200)),
        (_candidate("1.2", scale="million"), _answer("1200000"), Decimal(1200000)),
        (_candidate("0.0012", scale="billion"), _answer("1200000"), Decimal(1200000)),
        (_candidate("(42.50)"), _answer("-42.50"), Decimal("-42.50")),
        (_candidate("-$42.50", unit="usd"), _answer("-42.50", unit="USD"), Decimal("-42.50")),
        (_candidate("12.5%", unit="percent"), _answer("0.125", unit="%"), Decimal("0.125")),
        (_candidate("12.5", unit="percent", scale="percent"), _answer("0.125", unit="percent"), Decimal("0.125")),
        (_candidate("+3.40"), _answer("3.4"), Decimal("3.40")),
    ],
)
def test_scoring_normalizes_only_strict_candidate_numeric_values(candidate, answer, normalized):
    result = score_candidate(candidate, answer)

    assert result.correct is True
    assert result.candidate_value == normalized


@pytest.mark.parametrize(
    "value",
    [
        "Revenue was 19.5 in 2024",
        "19.5 2024",
        "approximately $1.2 million",
        "",
    ],
)
def test_scoring_never_extracts_first_or_last_number_from_prose(value):
    result = score_candidate(_candidate(value), _answer("19.5"))

    assert result.correct is False
    assert result.reason == "invalid_candidate_value"


def test_scoring_uses_decimal_tolerances_and_enforces_specified_compatibility():
    answer = _answer(
        "100",
        unit="USD",
        entity="Example Corp",
        period="FY 2024",
        absolute_tolerance="0.10",
        relative_tolerance="0.001",
    )

    assert score_candidate(
        _candidate("100.10", unit="$", entity=" example  corp ", period="fy 2024"),
        answer,
    ).correct
    assert not score_candidate(
        _candidate("100.11", unit="USD", entity="Example Corp", period="FY 2024"),
        answer,
    ).correct
    assert score_candidate(
        _candidate("100", unit="USD", entity="Example Corp", period="FY 2024"),
        answer,
    ).correct
    assert score_candidate(_candidate("100"), _answer("100")).correct
    assert score_candidate(
        _candidate("100", unit="EUR", entity="Example Corp", period="FY 2024"),
        answer,
    ).reason == "unit_mismatch"
    assert score_candidate(
        _candidate("100", unit="USD", entity="Other Corp", period="FY 2024"),
        answer,
    ).reason == "entity_mismatch"
    assert score_candidate(
        _candidate("100", unit="USD", entity="Example Corp", period="2023"),
        answer,
    ).reason == "period_mismatch"


def test_gold_oracle_serialization_requires_a_matching_hidden_label_join():
    public = PublicCase(
        case_id="case-1",
        dataset="finqa",
        document_id="doc-1",
        question="What was revenue?",
        evidence=(),
        stratum="headroom",
    )
    label = HiddenLabel(
        case_id="case-1",
        answer=_answer("1.20", unit="USD", scale="million", entity="Example", period="2024"),
        gold_derivation="1.2",
        gold_support_ids=("e1",),
        source_lineage=("snapshot", "doc-1", "q-1"),
    )

    oracle = serialize_gold_oracle(public, label)

    assert oracle == Candidate(
        value="1.20",
        unit="USD",
        scale="million",
        entity="Example",
        period="2024",
        expression="1.2",
        citations=("e1",),
    )
    with pytest.raises(ValueError, match="case_id"):
        serialize_gold_oracle(public, label.model_copy(update={"case_id": "case-2"}))
