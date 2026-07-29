from __future__ import annotations

import json
import math
import re
from typing import Any

KNOWN_LOBS = {
    "workers compensation",
    "property",
    "general liability",
    "auto",
    "cyber",
    "bop",
}
LOB_ALIASES = {
    "workers comp": "workers compensation",
    "worker's compensation": "workers compensation",
    "workers' compensation": "workers compensation",
    "commercial auto": "auto",
    "business owners policy": "bop",
    "business owner's policy": "bop",
}


def schema_for_task(task: str) -> dict[str, Any]:
    if task == "Appetite Check":
        return {
            "decision": "one of: yes, no, qualified, unknown",
            "rationale": "brief evidence-grounded string",
        }
    if task == "Small Business Elibility Check":
        return {
            "decision": "one of: small, large, unknown",
            "rationale": "brief evidence-grounded string",
        }
    if task == "Business Classification":
        return {
            "naics_code": "six-digit string or unknown",
            "rationale": "brief evidence-grounded string",
        }
    if task == "Deductibles":
        return {
            "applicable": "boolean",
            "deductible_usd": "integer or null",
            "rationale": "brief evidence-grounded string",
        }
    if task == "Policy Limits":
        return {
            "applicable": "boolean",
            "per_occurrence_usd": "integer or null",
            "aggregate_usd": "integer or null",
            "rationale": "brief evidence-grounded string",
        }
    if task == "Product Recommendations":
        return {
            "existing_lob": "canonical LOB string or unknown",
            "recommended_lobs": "array using only workers compensation, property, "
            "general liability, auto, cyber, bop",
            "rationale": "brief evidence-grounded string",
        }
    if task == "Multiple Choice Reasoning":
        return {
            "choice": "single uppercase option letter",
            "rationale": "brief explanation",
        }
    raise ValueError(f"unsupported task: {task}")


def _clean_json(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def parse_response(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = _clean_json(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None, ""
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, ""
    if not isinstance(payload, dict):
        return None, ""
    rationale = str(payload.pop("rationale", "")).strip()
    return payload, rationale


def _normal_lob(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).lower().replace("’", "'")).strip()
    text = LOB_ALIASES.get(text, text)
    return text if text in KNOWN_LOBS else "unknown"


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None
    return round(number) if math.isfinite(number) and number >= 0 else None


def canonical_decision(task: str, decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    if task == "Appetite Check":
        value = str(decision.get("decision", "")).lower().strip()
        return {"decision": value} if value in {"yes", "no", "qualified", "unknown"} else None
    if task == "Small Business Elibility Check":
        value = str(decision.get("decision", "")).lower().strip()
        return {"decision": value} if value in {"small", "large", "unknown"} else None
    if task == "Business Classification":
        value = str(decision.get("naics_code", "")).strip()
        return {"naics_code": value} if value == "unknown" or re.fullmatch(r"\d{6}", value) else None
    if task == "Deductibles":
        applicable = decision.get("applicable")
        if not isinstance(applicable, bool):
            return None
        amount = _integer(decision.get("deductible_usd"))
        return {"applicable": applicable, "deductible_usd": amount if applicable else None}
    if task == "Policy Limits":
        applicable = decision.get("applicable")
        if not isinstance(applicable, bool):
            return None
        per_occurrence = _integer(decision.get("per_occurrence_usd"))
        aggregate = _integer(decision.get("aggregate_usd"))
        return {
            "applicable": applicable,
            "per_occurrence_usd": per_occurrence if applicable else None,
            "aggregate_usd": aggregate if applicable else None,
        }
    if task == "Product Recommendations":
        existing = _normal_lob(decision.get("existing_lob", "unknown"))
        raw = decision.get("recommended_lobs")
        if not isinstance(raw, list):
            return None
        recommended = sorted({_normal_lob(value) for value in raw} - {"unknown"})
        return {"existing_lob": existing, "recommended_lobs": recommended}
    if task == "Multiple Choice Reasoning":
        value = str(decision.get("choice", "")).upper().strip()
        return {"choice": value} if re.fullmatch(r"[A-J]", value) else None
    return None


def decisions_equal(
    task: str,
    candidate: dict[str, Any] | None,
    gold: dict[str, Any],
) -> bool:
    return canonical_decision(task, candidate) == canonical_decision(task, gold)


def _money(text: str) -> list[int]:
    pattern = re.compile(
        r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|m|thousand|k)?\b",
        re.IGNORECASE,
    )
    values: list[int] = []
    for number, suffix in pattern.findall(text):
        raw = float(number.replace(",", ""))
        if suffix.lower() in {"million", "m"}:
            raw *= 1_000_000
        elif suffix.lower() in {"thousand", "k"}:
            raw *= 1_000
        integer = round(raw)
        if integer >= 100:
            values.append(integer)
    return values


def gold_from_references(
    task: str,
    references: list[str],
    *,
    existing_lob: str = "unknown",
) -> dict[str, Any]:
    text = references[0].strip()
    normal = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()
    if task == "Appetite Check":
        if "qualified" in normal:
            return {"decision": "qualified"}
        if any(term in normal for term in ("not in appetite", "out of appetite", "decline")):
            return {"decision": "no"}
        if "in appetite" in normal or normal.startswith("yes"):
            return {"decision": "yes"}
        raise ValueError(f"cannot parse appetite reference: {text}")
    if task == "Small Business Elibility Check":
        if "does not qualify" in normal or "large business" in normal:
            return {"decision": "large"}
        if "qualifies as a small business" in normal:
            return {"decision": "small"}
        raise ValueError(f"cannot parse eligibility reference: {text}")
    if task == "Business Classification":
        values = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
        if len(values) != 1:
            raise ValueError(f"cannot parse NAICS reference: {text}")
        return {"naics_code": values[0]}
    if task == "Deductibles":
        if "irrelevant" in normal or "out of appetite" in normal:
            return {"applicable": False, "deductible_usd": None}
        values = _money(text)
        if len(values) != 1:
            raise ValueError(f"cannot parse deductible reference: {text}")
        return {"applicable": True, "deductible_usd": values[0]}
    if task == "Policy Limits":
        if "irrelevant" in normal or "out of appetite" in normal:
            return {
                "applicable": False,
                "per_occurrence_usd": None,
                "aggregate_usd": None,
            }
        values = _money(text)
        if len(values) != 2:
            raise ValueError(f"cannot parse policy-limit reference: {text}")
        return {
            "applicable": True,
            "per_occurrence_usd": values[0],
            "aggregate_usd": values[1],
        }
    if task == "Product Recommendations":
        existing = _normal_lob(existing_lob)
        if "no other lobs" in normal:
            return {"existing_lob": existing, "recommended_lobs": []}
        found = {
            lob
            for lob in KNOWN_LOBS
            if re.search(rf"\b{re.escape(lob)}\b", normal)
        }
        found.discard(existing)
        return {"existing_lob": existing, "recommended_lobs": sorted(found)}
    if task == "Multiple Choice Reasoning":
        value = str(references[0]).upper().strip()
        if not re.fullmatch(r"[A-J]", value):
            raise ValueError(f"cannot parse multiple-choice reference: {text}")
        return {"choice": value}
    raise ValueError(f"unsupported task: {task}")
