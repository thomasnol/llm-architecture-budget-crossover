from budget_crossover.policy import PolicyInputs, adjudicate_policy, dti_category


def _inputs(**overrides):
    values = {
        "annual_income_usd": 120_000,
        "property_value_usd": 500_000,
        "loan_to_value_percent": 80.0,
        "debt_to_income_band": "36",
        "loan_term_months": 360,
        "conforming_limit_status": "C",
    }
    values.update(overrides)
    return PolicyInputs(**values)


def test_policy_precedence_and_reason_codes():
    assert adjudicate_policy(_inputs()) == ("approve", ["meets_policy"])
    assert adjudicate_policy(_inputs(debt_to_income_band="45", loan_to_value_percent=95)) == (
        "conditional_review",
        ["elevated_dti", "elevated_ltv"],
    )
    assert adjudicate_policy(
        _inputs(
            debt_to_income_band="50%-60%",
            loan_to_value_percent=101,
            conforming_limit_status="NC",
        )
    ) == ("deny", ["excessive_dti", "excessive_ltv"])
    assert adjudicate_policy(_inputs(annual_income_usd=None, debt_to_income_band=None)) == (
        "manual_review",
        ["missing_dti", "missing_income"],
    )


def test_dti_categories_cover_hmda_bands_and_exact_values():
    assert dti_category("<20%") == "below_43"
    assert dti_category("30%-<36%") == "below_43"
    assert dti_category("43") == "elevated"
    assert dti_category("49") == "elevated"
    assert dti_category("50%-60%") == "excessive"
    assert dti_category(">60%") == "excessive"
    assert dti_category(None) == "missing"
