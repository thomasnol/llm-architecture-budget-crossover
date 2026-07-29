import pytest

from budget_crossover.evaluate import exact_evaluate


@pytest.mark.parametrize(
    ("task", "candidate", "reference"),
    [
        (
            "Appetite Check",
            "Yes, this company is in appetite for general liability.",
            "Yes, the company is in appetite for that LOB.",
        ),
        (
            "Small Business Elibility Check",
            "The company does not qualify and is a large business.",
            "This company does NOT qualify as a small business, it is a large business.",
        ),
        (
            "Policy Limits",
            "$3 million per incident and $5 million aggregate.",
            "You should offer a per-incident policy limit of $3.0 million and an aggregate limit of $5.0 million.",
        ),
        (
            "Deductibles",
            "Offer a $5,000 deductible.",
            "You should offer a $5000.0 deductible.",
        ),
        (
            "Business Classification",
            "The NAICS code is 522292.",
            "The company's NAICS code is 522292.",
        ),
        (
            "Product Recommendations",
            "Offer property, BOP, general liability, cyber, and auto.",
            "In addition to workers compensation, the other LOBs in appetite for this company are property, bop, general liability, cyber, auto.",
        ),
    ],
)
def test_task_specific_exact_matches(task, candidate, reference):
    assert exact_evaluate(task, candidate, [reference]).correct


def test_wrong_appetite_decision_fails():
    assert not exact_evaluate(
        "Appetite Check",
        "No, it is out of appetite.",
        ["Yes, the company is in appetite for that LOB."],
    ).correct
