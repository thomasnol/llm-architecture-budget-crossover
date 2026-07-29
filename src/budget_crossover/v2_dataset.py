from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from .dataset import compact_tool_evidence
from .v2_models import V2Case
from .v2_schema import gold_from_references, schema_for_task

MMLU_PARQUET_URL = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
    "data/test-00000-of-00001.parquet"
)


def _message_content(message: dict[str, Any]) -> str:
    return str(message.get("content", "")).strip()


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def build_insurance_v2_cases(
    parquet_path: Path,
    *,
    evidence_condition: str,
    fixed_source_model: str,
    max_context_chars: int,
) -> list[V2Case]:
    frame = pd.read_parquet(parquet_path)
    cases: list[V2Case] = []
    for numeric_case_id, group in frame.groupby("company task id", sort=True):
        group = group.sort_values(["assistant model name", "primary id"]).copy()
        fixed = group[group["assistant model name"] == fixed_source_model]
        if fixed.empty:
            raise ValueError(
                f"case {numeric_case_id} has no trace from fixed model {fixed_source_model}"
            )
        source = fixed.sort_values("primary id").iloc[0]
        trace_rows = [source] if evidence_condition == "fixed" else list(group.itertuples())

        underwriter = _unique(
            [
                _message_content(message)
                for message in source["trace"]
                if message.get("role") == "user" and _message_content(message)
            ]
        )
        tool_contents: list[str] = []
        for row in trace_rows:
            trace = row["trace"] if isinstance(row, pd.Series) else row.trace
            tool_contents.extend(
                _message_content(message)
                for message in trace
                if message.get("type") == "tool" and _message_content(message)
            )
        tool_contents = _unique(tool_contents)

        metadata = {
            "company_name": str(source["company name"]),
            "annual_revenue_usd": int(source["annual revenue"]),
            "number_of_employees": int(source["number of employees"]),
            "total_payroll_usd": int(source["total payroll"]),
            "number_of_vehicles": int(source["number of vehicles"]),
            "building_construction": str(source["building construction"]),
            "state": str(source["state"]),
            "company_description": str(source["company description"]),
            "existing_lob": str(source["lob"]).lower(),
            "fixed_source_model": fixed_source_model,
        }
        query_text = " ".join(
            [metadata["company_name"], metadata["company_description"], *underwriter]
        )
        evidence = compact_tool_evidence(
            tool_contents,
            state=metadata["state"],
            query_text=query_text,
            max_total_chars=max_context_chars - len(query_text) - 5000,
        )
        task = str(source["task"])
        references = sorted({str(value).strip() for value in group["reference answer"]})
        gold = gold_from_references(
            task,
            references,
            existing_lob=metadata["existing_lob"],
        )
        question = (
            f"Company: {metadata['company_name']}\n"
            f"Task: {task}\n"
            + "\n".join(f"Underwriter {i + 1}: {text}" for i, text in enumerate(underwriter))
        )
        context = (
            "STRUCTURED COMPANY FACTS\n"
            + json.dumps(metadata, ensure_ascii=False, indent=2)
            + "\n\nTOOL EVIDENCE\n"
            + "\n\n".join(
                f"[Evidence {index + 1}]\n{value}" for index, value in enumerate(evidence)
            )
        )
        cases.append(
            V2Case(
                case_id=f"insurance-{int(numeric_case_id)}",
                dataset="insurance",
                task=task,
                question=question,
                context=context,
                output_schema=schema_for_task(task),
                gold_decision=gold,
                evidence_condition=evidence_condition,
                evidence_chars=sum(map(len, evidence)),
                metadata={
                    **metadata,
                    "reference_variants": references,
                    "historical_trace_count": len(group),
                    "tool_evidence_count": len(evidence),
                },
            )
        )
    return cases


def fetch_mmlu_pro(destination: Path, *, timeout_seconds: int = 120) -> list[dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return pd.read_parquet(destination).to_dict(orient="records")
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(MMLU_PARQUET_URL)
        response.raise_for_status()
        destination.write_bytes(response.content)
    return pd.read_parquet(destination).to_dict(orient="records")


def _answer_letter(answer: Any, options: list[str]) -> str:
    if isinstance(answer, int):
        return string.ascii_uppercase[answer]
    value = str(answer).upper().strip()
    if value in string.ascii_uppercase[: len(options)]:
        return value
    if str(answer) in options:
        return string.ascii_uppercase[options.index(str(answer))]
    raise ValueError(f"unsupported MMLU-Pro answer: {answer!r}")


def sample_mmlu_pro(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[V2Case]:
    frame = pd.DataFrame(rows)
    if sample_size > len(frame):
        raise ValueError(f"requested {sample_size} MMLU-Pro rows from {len(frame)}")
    rng = np.random.default_rng(seed)
    categories = sorted(frame["category"].astype(str).unique())
    base = sample_size // len(categories)
    remainder = sample_size % len(categories)
    selected: list[pd.Series] = []
    for index, category in enumerate(categories):
        group = frame[frame["category"].astype(str) == category]
        count = min(len(group), base + (1 if index < remainder else 0))
        choices = rng.choice(len(group), size=count, replace=False)
        selected.extend(group.iloc[int(choice)] for choice in choices)
    if len(selected) < sample_size:
        selected_ids = {str(row.get("question_id", row.name)) for row in selected}
        remaining = frame[
            ~frame.apply(
                lambda row: str(row.get("question_id", row.name)) in selected_ids,
                axis=1,
            )
        ]
        choices = rng.choice(len(remaining), size=sample_size - len(selected), replace=False)
        selected.extend(remaining.iloc[int(choice)] for choice in choices)

    cases: list[V2Case] = []
    for position, row in enumerate(selected):
        options = list(row["options"])
        option_text = "\n".join(
            f"{string.ascii_uppercase[index]}. {value}"
            for index, value in enumerate(options)
        )
        source_id = row.get("question_id", row.name)
        answer = _answer_letter(row["answer"], options)
        cases.append(
            V2Case(
                case_id=f"mmlu-pro-{source_id}",
                dataset="mmlu_pro",
                task="Multiple Choice Reasoning",
                question=f"{row['question']}\n\nOPTIONS\n{option_text}",
                context="Use the information in the question and general reasoning. "
                "Select exactly one option.",
                output_schema=schema_for_task("Multiple Choice Reasoning"),
                gold_decision={"choice": answer},
                evidence_chars=len(str(row["question"])),
                metadata={
                    "category": str(row["category"]),
                    "source_id": str(source_id),
                    "sample_position": position,
                },
            )
        )
    return sorted(cases, key=lambda case: case.case_id)


def stratified_sample_cases(
    cases: list[V2Case],
    *,
    sample_size: int,
    seed: int,
) -> list[V2Case]:
    if sample_size >= len(cases):
        return sorted(cases, key=lambda case: case.case_id)
    frame = pd.DataFrame(
        [{"index": index, "stratum": f"{case.dataset}:{case.task}"} for index, case in enumerate(cases)]
    )
    rng = np.random.default_rng(seed)
    groups = {name: group for name, group in frame.groupby("stratum")}
    counts = {name: len(group) for name, group in groups.items()}
    ideals = {name: sample_size * count / len(frame) for name, count in counts.items()}
    quotas = {name: max(1, int(np.floor(value))) for name, value in ideals.items()}
    while sum(quotas.values()) < sample_size:
        name = max(quotas, key=lambda item: ideals[item] - quotas[item])
        quotas[name] += 1
    while sum(quotas.values()) > sample_size:
        candidates = [name for name in quotas if quotas[name] > 1]
        name = min(candidates, key=lambda item: ideals[item] - quotas[item])
        quotas[name] -= 1
    selected: list[V2Case] = []
    for name, group in groups.items():
        indices = rng.choice(group["index"].to_numpy(), size=quotas[name], replace=False)
        selected.extend(cases[int(index)] for index in indices)
    return sorted(selected, key=lambda case: case.case_id)
