import pytest

from budget_crossover.checking import check_candidate
from budget_crossover.models import Candidate, EvidenceItem


def _evidence() -> tuple[EvidenceItem, ...]:
    return (
        EvidenceItem(
            evidence_id="revenue",
            document_id="doc-1",
            kind="table_row",
            text="Revenue | 2024 | $100 million",
            headers=("Metric", "2024", "USD millions"),
            row_label="Revenue",
            unit="USD",
            scale="million",
            entity="Acme Corp",
            period="2024",
            ordinal=0,
        ),
        EvidenceItem(
            evidence_id="costs",
            document_id="doc-1",
            kind="table_row",
            text="Costs | 2024 | $40 million",
            headers=("Metric", "2024", "USD millions"),
            row_label="Costs",
            unit="USD",
            scale="million",
            entity="Acme Corp",
            period="2024",
            ordinal=1,
        ),
    )


def _candidate(**updates) -> Candidate:
    values = {
        "value": "60",
        "unit": "USD",
        "scale": "million",
        "entity": "Acme Corp",
        "period": "2024",
        "expression": "100 - 40",
        "citations": ("revenue", "costs"),
    }
    values.update(updates)
    return Candidate(**values)


def test_checker_approves_supported_safe_consistent_arithmetic_without_gold_judgment():
    result = check_candidate(_candidate(), _evidence())

    assert result.passed is True
    assert result.findings == ()
    assert str(result.evaluated_expression) == "60"
    assert not hasattr(result, "correct")
    assert not hasattr(result, "gold_value")


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        (_candidate(value="61"), "expression_mismatch"),
        (_candidate(citations=("revenue", "fabricated")), "fabricated_citation"),
        (_candidate(expression="100 - 41"), "unsupported_operand"),
        (_candidate(expression="100 / (40 - 40)"), "division_by_zero"),
        (_candidate(scale="billion"), "scale_mismatch"),
        (_candidate(unit="EUR"), "unit_mismatch"),
        (_candidate(entity="Other Corp"), "entity_mismatch"),
        (_candidate(period="2023"), "period_mismatch"),
        (_candidate(citations=()), "missing_citations"),
        (_candidate(expression=None), "missing_expression"),
    ],
)
def test_checker_rejects_invalid_arithmetic_provenance_or_metadata(candidate, code):
    result = check_candidate(candidate, _evidence())

    assert result.passed is False
    assert code in {finding.code for finding in result.findings}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "100 ** 2",
        "[100][0]",
        "(lambda: 1)()",
        "value + 1",
    ],
)
def test_checker_rejects_unsafe_ast_nodes_instead_of_executing_them(expression):
    result = check_candidate(_candidate(expression=expression), _evidence())

    assert result.passed is False
    assert "unsafe_expression" in {finding.code for finding in result.findings}


@pytest.mark.parametrize(
    ("evidence_text", "expression", "value", "expected_pass", "expected_code"),
    [
        ("Reported loss | -40", "40", "40", False, "unsupported_operand"),
        ("Reported loss | -40", "-40", "-40", True, None),
        ("Reported ratio | .5", ".5", ".5", True, None),
    ],
)
def test_checker_operand_provenance_preserves_signs_and_leading_decimals(
    evidence_text,
    expression,
    value,
    expected_pass,
    expected_code,
):
    evidence = EvidenceItem(
        evidence_id="operand",
        document_id="doc-1",
        kind="table_row",
        text=evidence_text,
        ordinal=0,
    )
    candidate = Candidate(
        value=value,
        unit=None,
        scale="ones",
        entity=None,
        period=None,
        expression=expression,
        citations=("operand",),
    )

    result = check_candidate(candidate, (evidence,))

    assert result.passed is expected_pass
    if expected_code is not None:
        assert expected_code in {finding.code for finding in result.findings}
    else:
        assert result.findings == ()


def test_checker_validates_explicit_counted_evidence_without_numeric_operands():
    evidence = (
        EvidenceItem(
            evidence_id="alpha",
            document_id="doc-1",
            kind="text",
            text="Reported countable alpha item.",
            ordinal=0,
        ),
        EvidenceItem(
            evidence_id="beta",
            document_id="doc-1",
            kind="text",
            text="Reported countable beta item.",
            ordinal=1,
        ),
    )
    valid = Candidate(
        value="2",
        unit=None,
        scale="ones",
        entity=None,
        period=None,
        expression='count("alpha", "beta")',
        citations=("alpha", "beta"),
    )
    mismatched = valid.model_copy(update={"citations": ("alpha",)})

    assert check_candidate(valid, evidence).passed is True
    result = check_candidate(mismatched, evidence)
    assert result.passed is False
    assert "count_evidence_mismatch" in {finding.code for finding in result.findings}
