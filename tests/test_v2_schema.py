import pytest

from budget_crossover.v2_schema import (
    canonical_decision,
    decisions_equal,
    gold_from_references,
    parse_response,
)


@pytest.mark.parametrize(
    ("task", "reference", "existing", "expected"),
    [
        (
            "Appetite Check",
            "No, the company is not in appetite for that LOB.",
            "auto",
            {"decision": "no"},
        ),
        (
            "Small Business Elibility Check",
            "This company does NOT qualify as a small business, it is a large business.",
            "auto",
            {"decision": "large"},
        ),
        (
            "Policy Limits",
            (
                "You should offer a per-incident policy limit of $3.0 million and an "
                "aggregate limit of $5.0 million."
            ),
            "cyber",
            {
                "applicable": True,
                "per_occurrence_usd": 3_000_000,
                "aggregate_usd": 5_000_000,
            },
        ),
        (
            "Deductibles",
            "You should offer a $5000.0 deductible.",
            "property",
            {"applicable": True, "deductible_usd": 5000},
        ),
        (
            "Product Recommendations",
            (
                "In addition to auto, the other LOBs in appetite for this company are "
                "property, bop, workers compensation, general liability, cyber."
            ),
            "auto",
            {
                "existing_lob": "auto",
                "recommended_lobs": [
                    "bop",
                    "cyber",
                    "general liability",
                    "property",
                    "workers compensation",
                ],
            },
        ),
    ],
)
def test_gold_reference_parsing(task, reference, existing, expected):
    assert gold_from_references(task, [reference], existing_lob=existing) == expected


def test_rationale_numbers_do_not_contaminate_policy_decision():
    parsed, rationale = parse_response(
        '{"applicable":true,"per_occurrence_usd":3000000,'
        '"aggregate_usd":5000000,"rationale":"The deductible is $500."}'
    )
    assert rationale == "The deductible is $500."
    assert decisions_equal(
        "Policy Limits",
        parsed,
        {
            "applicable": True,
            "per_occurrence_usd": 3_000_000,
            "aggregate_usd": 5_000_000,
        },
    )


def test_negated_existing_lob_is_not_in_structured_recommendations():
    candidate = {
        "existing_lob": "auto",
        "recommended_lobs": [
            "workers compensation",
            "property",
            "general liability",
            "cyber",
            "bop",
        ],
    }
    gold = gold_from_references(
        "Product Recommendations",
        [
            (
                "In addition to auto, the other LOBs in appetite for this company are "
                "property, bop, workers compensation, general liability, cyber."
            )
        ],
        existing_lob="auto",
    )
    assert decisions_equal("Product Recommendations", candidate, gold)


def test_invalid_schema_is_not_silently_coerced():
    assert canonical_decision(
        "Small Business Elibility Check",
        {"decision": "not a small business"},
    ) is None
