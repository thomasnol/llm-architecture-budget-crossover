from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import Case

TOKEN_RE = re.compile(r"[a-z0-9]+")
DROP_TOOL_PREFIXES = (
    '["appetite_guide",',
    '{"naics": "table with',
    "Error: ToolException",
)


def _tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "company",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in stop}


def _json_list(content: str) -> list[Any] | None:
    if not content.lstrip().startswith("["):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def _candidate_naics(tool_contents: list[str], query_text: str) -> set[int]:
    query_tokens = _tokens(query_text)
    scored: list[tuple[float, int]] = []
    for content in tool_contents:
        rows = _json_list(content)
        if not rows or not isinstance(rows[0], dict):
            continue
        if not {"Code", "Title"}.issubset(rows[0]):
            continue
        for row in rows:
            try:
                code = int(row["Code"])
            except (KeyError, TypeError, ValueError):
                continue
            row_text = f"{row.get('Title', '')} {row.get('Description', '')}"
            overlap = len(query_tokens & _tokens(row_text))
            if overlap:
                scored.append((overlap / math.sqrt(max(1, len(_tokens(row_text)))), code))
    scored.sort(reverse=True)
    return {code for _, code in scored[:8]}


def _compact_structured_rows(
    rows: list[dict[str, Any]],
    *,
    state: str,
    query_text: str,
    candidate_naics: set[int],
) -> list[dict[str, Any]]:
    if len(rows) <= 16:
        return rows
    keys = set(rows[0])
    if "state" in keys:
        state_rows = [row for row in rows if str(row.get("state", "")).lower() == state.lower()]
        if state_rows:
            return state_rows[:16]
    if {"Code", "Title"}.issubset(keys):
        query_tokens = _tokens(query_text)

        def score(row: dict[str, Any]) -> tuple[float, int]:
            text = f"{row.get('Title', '')} {row.get('Description', '')}"
            overlap = len(query_tokens & _tokens(text))
            return (overlap / math.sqrt(max(1, len(_tokens(text)))), overlap)

        return sorted(rows, key=score, reverse=True)[:10]
    if "NAICS Codes" in keys and candidate_naics:
        matches = []
        for row in rows:
            try:
                code = int(row.get("NAICS Codes"))
            except (TypeError, ValueError):
                continue
            if code in candidate_naics:
                matches.append(row)
        if matches:
            return matches[:16]
    return rows[:8]


def compact_tool_evidence(
    contents: list[str],
    *,
    state: str,
    query_text: str,
    max_total_chars: int,
) -> list[str]:
    candidates = _candidate_naics(contents, query_text)
    evidence: list[str] = []
    used = 0
    for content in contents:
        content = content.strip()
        if not content or content in {"[]", "{}"} or content.startswith(DROP_TOOL_PREFIXES):
            continue
        rows = _json_list(content)
        if rows is not None and rows and isinstance(rows[0], dict):
            compact = _compact_structured_rows(
                rows,
                state=state,
                query_text=query_text,
                candidate_naics=candidates,
            )
            content = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        elif len(content) > 12000:
            content = content[:6000] + "\n[...broad tool output truncated...]\n" + content[-6000:]
        remaining = max_total_chars - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n[...case evidence cap reached...]"
        evidence.append(content)
        used += len(content)
    return evidence


def build_cases(
    parquet_path: Path,
    *,
    max_context_chars: int = 72000,
) -> list[Case]:
    frame = pd.read_parquet(parquet_path)
    cases: list[Case] = []
    for case_id, group in frame.groupby("company task id", sort=True):
        group = group.copy()
        group["_trace_chars"] = group["trace"].map(
            lambda trace: sum(len(str(message.get("content", ""))) for message in trace)
        )
        successful = group[group["correct"].astype(bool)]
        source_group = successful if not successful.empty else group
        source = source_group.sort_values(
            ["_trace_chars", "primary id"], ascending=[False, True]
        ).iloc[0]

        underwriter = [
            str(message.get("content", "")).strip()
            for message in source["trace"]
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        tool_contents = [
            str(message.get("content", ""))
            for message in source["trace"]
            if message.get("type") == "tool"
        ]
        metadata = {
            "annual_revenue_usd": int(source["annual revenue"]),
            "number_of_employees": int(source["number of employees"]),
            "total_payroll_usd": int(source["total payroll"]),
            "number_of_vehicles": int(source["number of vehicles"]),
            "building_construction": str(source["building construction"]),
            "state": str(source["state"]),
            "company_description": str(source["company description"]),
            "line_of_business": str(source["lob"]),
        }
        query_text = " ".join(
            [str(source["company name"]), metadata["company_description"], *underwriter]
        )
        evidence = compact_tool_evidence(
            tool_contents,
            state=metadata["state"],
            query_text=query_text,
            max_total_chars=max_context_chars - len(query_text) - 4000,
        )
        references = sorted({str(value).strip() for value in group["reference answer"]})
        counts = Counter(str(value).strip() for value in group["reference answer"])
        canonical = min(counts, key=lambda value: (-counts[value], value))
        case = Case(
            case_id=int(case_id),
            task=str(source["task"]),
            company_name=str(source["company name"]),
            company_metadata=metadata,
            underwriter_messages=underwriter,
            evidence=evidence,
            accepted_reference_answers=references,
            canonical_reference=canonical,
            evidence_chars=sum(map(len, evidence)),
            tool_evidence_count=len(evidence),
            source_primary_id=int(source["primary id"]),
            source_model=str(source["assistant model name"]),
        )
        reference_norm = re.sub(r"\s+", " ", canonical).strip().lower()
        packet_norm = re.sub(r"\s+", " ", " ".join(underwriter + evidence)).strip().lower()
        if reference_norm and reference_norm in packet_norm:
            raise ValueError(f"reference leakage detected for case {case_id}")
        cases.append(case)
    return cases


def _hamilton_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    ideals = {task: sample_size * count / total for task, count in counts.items()}
    quotas = {task: min(counts[task], max(1, math.floor(value))) for task, value in ideals.items()}
    while sum(quotas.values()) < sample_size:
        candidates = [task for task in counts if quotas[task] < counts[task]]
        task = max(candidates, key=lambda name: (ideals[name] - quotas[name], counts[name], name))
        quotas[task] += 1
    while sum(quotas.values()) > sample_size:
        candidates = [task for task in counts if quotas[task] > 1]
        task = min(candidates, key=lambda name: (ideals[name] - quotas[name], counts[name], name))
        quotas[task] -= 1
    return quotas


def sample_cases(
    cases: list[Case],
    *,
    sample_size: int,
    seed: int,
    task_quotas: dict[str, int] | None = None,
) -> list[Case]:
    if sample_size >= len(cases):
        return sorted(cases, key=lambda case: case.case_id)
    by_task: dict[str, list[Case]] = {}
    for case in cases:
        by_task.setdefault(case.task, []).append(case)
    counts = {task: len(values) for task, values in by_task.items()}
    quotas = task_quotas or _hamilton_quotas(counts, sample_size)
    if sum(quotas.values()) != sample_size:
        raise ValueError("task quotas must sum to sample_size")
    unknown = set(quotas) - set(by_task)
    if unknown:
        raise ValueError(f"task quotas contain unknown tasks: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    selected: list[Case] = []
    for task in sorted(by_task):
        quota = quotas.get(task, 0)
        if quota > len(by_task[task]):
            raise ValueError(f"quota {quota} exceeds {len(by_task[task])} cases for {task}")
        indices = rng.choice(len(by_task[task]), size=quota, replace=False)
        selected.extend(by_task[task][int(index)] for index in indices)
    return sorted(selected, key=lambda case: case.case_id)
