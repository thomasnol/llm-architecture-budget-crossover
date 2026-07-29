from __future__ import annotations

import json
import re
from dataclasses import dataclass

KNOWN_LOBS = {
    "workers compensation": re.compile(
        r"workers[’']?\s+compensation|\bworkers\s+comp\b", re.IGNORECASE
    ),
    "property": re.compile(r"\bproperty\b", re.IGNORECASE),
    "general liability": re.compile(r"\bgeneral\s+liability\b", re.IGNORECASE),
    "auto": re.compile(r"\b(?:commercial\s+)?auto(?:mobile)?\b", re.IGNORECASE),
    "cyber": re.compile(r"\bcyber\b", re.IGNORECASE),
    "bop": re.compile(r"\bbop\b|business\s+owners?[’']?\s+policy", re.IGNORECASE),
}


@dataclass(frozen=True)
class ExactEvaluation:
    correct: bool
    candidate_value: str
    reference_values: tuple[str, ...]


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def _appetite_value(text: str) -> str:
    value = _normal(text)
    if any(term in value for term in ["not in appetite", "out of appetite", "decline"]):
        return "no"
    if "qualified" in value:
        return "qualified"
    if re.search(r"\b(?:yes|in appetite|within appetite|accept)\b", value):
        return "yes"
    return "unknown"


def _eligibility_value(text: str) -> str:
    value = _normal(text)
    if re.search(r"(?:does not|doesn't|not)\s+qualif|large business|ineligible", value):
        return "large"
    if re.search(r"\bqualif(?:y|ies|ied)\b|\bsmall business\b|\beligible\b", value):
        return "small"
    return "unknown"


def _money_values(text: str) -> tuple[int, ...]:
    pattern = re.compile(
        r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|m|thousand|k)?\b",
        re.IGNORECASE,
    )
    values: set[int] = set()
    for number, suffix in pattern.findall(text):
        raw = float(number.replace(",", ""))
        suffix = suffix.lower()
        if suffix in {"million", "m"}:
            raw *= 1_000_000
        elif suffix in {"thousand", "k"}:
            raw *= 1_000
        integer = round(raw)
        if integer >= 100:
            values.add(integer)
    return tuple(sorted(values))


def _naics_values(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"(?<!\d)\d{6}(?!\d)", text))))


def _lob_values(text: str) -> tuple[str, ...]:
    value = _normal(text)
    if re.search(r"\bno other\b|\bnone\b.*\bappetite\b", value):
        return ()
    found = {name for name, pattern in KNOWN_LOBS.items() if pattern.search(value)}
    existing_match = re.search(r"in addition to (.+?),\s*the other", value)
    if existing_match:
        existing_text = existing_match.group(1)
        for name, pattern in KNOWN_LOBS.items():
            if pattern.search(existing_text):
                found.discard(name)
    return tuple(sorted(found))


def _task_value(task: str, text: str) -> str:
    if task == "Appetite Check":
        return _appetite_value(text)
    if task == "Small Business Elibility Check":
        return _eligibility_value(text)
    if task in {"Policy Limits", "Deductibles"}:
        if "irrelevant" in _normal(text) or "out of appetite" in _normal(text):
            return "not_applicable"
        return json.dumps(_money_values(text))
    if task == "Business Classification":
        return json.dumps(_naics_values(text))
    if task == "Product Recommendations":
        return json.dumps(_lob_values(text))
    return _normal(text)


def exact_evaluate(task: str, candidate: str, references: list[str]) -> ExactEvaluation:
    candidate_value = _task_value(task, candidate)
    reference_values = tuple(sorted({_task_value(task, reference) for reference in references}))
    return ExactEvaluation(
        correct=candidate_value in reference_values,
        candidate_value=candidate_value,
        reference_values=reference_values,
    )
