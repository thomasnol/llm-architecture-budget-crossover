from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

DECISIONS = {"approve", "conditional_review", "deny", "manual_review"}
REASON_CODES = {
    "meets_policy",
    "missing_income",
    "missing_property_value",
    "missing_ltv",
    "missing_dti",
    "excessive_dti",
    "excessive_ltv",
    "term_exceeds_30_years",
    "elevated_dti",
    "elevated_ltv",
    "nonconforming_amount",
}

POLICY_TEXT = """RESEARCH POLICY ORACLE — NOT A REAL LENDING POLICY
Scope: conventional, first-lien, closed-end, consumer-purpose home-purchase
applications for owner-occupied, site-built properties with one to four units.
Inputs allowed for adjudication: verified annual income, debt-to-income ratio
(DTI), loan amount, property value, loan-to-value ratio (LTV), term, and
conforming-limit status. Demographic and neighborhood fields are monitoring-only
and must never affect the decision.

Apply rules in this order:
1. MANUAL REVIEW if income, property value, LTV, or DTI is not reported.
2. DENY if DTI is at least 50%, LTV exceeds 100%, or term exceeds 360 months.
3. CONDITIONAL REVIEW if DTI is 43–49%, LTV exceeds 90% through 100%, or the
   application is nonconforming. Include every applicable reason code at the
   controlling decision level.
4. APPROVE otherwise, with reason code meets_policy.

Allowed reason codes:
meets_policy, missing_income, missing_property_value, missing_ltv, missing_dti,
excessive_dti, excessive_ltv, term_exceeds_30_years, elevated_dti, elevated_ltv,
nonconforming_amount."""


@dataclass(frozen=True)
class PolicyInputs:
    annual_income_usd: float | None
    property_value_usd: float | None
    loan_to_value_percent: float | None
    debt_to_income_band: str | None
    loan_term_months: int | None
    conforming_limit_status: str | None


def _finite_float(value: Any, *, multiplier: float = 1.0) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "exempt"}:
        return None
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_float(value)
    return round(number) if number is not None else None


def policy_inputs_from_hmda(row: dict[str, Any]) -> PolicyInputs:
    return PolicyInputs(
        annual_income_usd=_finite_float(row.get("income"), multiplier=1000),
        property_value_usd=_finite_float(row.get("property_value")),
        loan_to_value_percent=_finite_float(row.get("loan_to_value_ratio")),
        debt_to_income_band=_normal_dti(row.get("debt_to_income_ratio")),
        loan_term_months=_integer(row.get("loan_term")),
        conforming_limit_status=_normal_optional(row.get("conforming_loan_limit")),
    )


def _normal_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"na", "nan", "exempt"} else text


def _normal_dti(value: Any) -> str | None:
    text = _normal_optional(value)
    if text is None:
        return None
    return re.sub(r"\s+", "", text)


def dti_category(value: str | None) -> str:
    if value is None:
        return "missing"
    compact = value.replace("%", "")
    if compact in {"<20", "20-<30", "30-<36"}:
        return "below_43"
    if compact == "50-60" or compact == ">60":
        return "excessive"
    try:
        number = float(compact)
    except ValueError:
        return "missing"
    if number >= 50:
        return "excessive"
    if number >= 43:
        return "elevated"
    return "below_43"


def adjudicate_policy(inputs: PolicyInputs) -> tuple[str, list[str]]:
    missing: list[str] = []
    if inputs.annual_income_usd is None:
        missing.append("missing_income")
    if inputs.property_value_usd is None:
        missing.append("missing_property_value")
    if inputs.loan_to_value_percent is None:
        missing.append("missing_ltv")
    dti = dti_category(inputs.debt_to_income_band)
    if dti == "missing":
        missing.append("missing_dti")
    if missing:
        return "manual_review", sorted(missing)

    hard_fail: list[str] = []
    if dti == "excessive":
        hard_fail.append("excessive_dti")
    if inputs.loan_to_value_percent is not None and inputs.loan_to_value_percent > 100:
        hard_fail.append("excessive_ltv")
    if inputs.loan_term_months is not None and inputs.loan_term_months > 360:
        hard_fail.append("term_exceeds_30_years")
    if hard_fail:
        return "deny", sorted(hard_fail)

    conditions: list[str] = []
    if dti == "elevated":
        conditions.append("elevated_dti")
    if inputs.loan_to_value_percent is not None and inputs.loan_to_value_percent > 90:
        conditions.append("elevated_ltv")
    if inputs.conforming_limit_status == "NC":
        conditions.append("nonconforming_amount")
    if conditions:
        return "conditional_review", sorted(conditions)
    return "approve", ["meets_policy"]


def canonical_decision(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    decision = str(value.get("decision", "")).strip().lower()
    reasons = value.get("reason_codes")
    if decision not in DECISIONS or not isinstance(reasons, list):
        return None
    normalized = sorted({str(reason).strip().lower() for reason in reasons})
    if not normalized or any(reason not in REASON_CODES for reason in normalized):
        return None
    return {"decision": decision, "reason_codes": normalized}
