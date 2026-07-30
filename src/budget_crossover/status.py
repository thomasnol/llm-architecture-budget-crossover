from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .io import read_jsonl
from .records import FailureAttempt, Generation
from .runner import error_path, generation_path


def _counts(values: list[Any]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def summarize_run(
    *,
    repo: Path,
    config: ExperimentConfig,
    expected_cells: int,
) -> dict[str, Any]:
    generations = read_jsonl(generation_path(repo, config), Generation)
    attempts = read_jsonl(error_path(repo, config), FailureAttempt)
    scored = {(row.case_id, row.system, row.token_budget, row.repetition) for row in generations}
    failed = {
        (row.case_id, row.system, row.token_budget, row.repetition)
        for row in attempts
        if (row.case_id, row.system, row.token_budget, row.repetition) not in scored
    }
    return {
        "experiment": config.experiment_name,
        "expected_cells": expected_cells,
        "scored_cells": len(scored),
        "remaining_cells": max(0, expected_cells - len(scored)),
        "error_attempts": len(attempts),
        "unique_failed_cells": len(failed),
        "failures_by_system": _counts([row.system for row in attempts]),
        "failures_by_budget": _counts([row.token_budget for row in attempts]),
        "failures_by_model": _counts([row.attempted_model for row in attempts]),
        "failures_by_stage": _counts([row.stage for row in attempts]),
        "failures_by_http_status": _counts(
            [row.status_code if row.status_code is not None else "transport" for row in attempts]
        ),
        "failures_by_retryability": _counts(
            ["retryable" if row.retryable else "permanent" for row in attempts]
        ),
        "failures_by_signature": _counts([row.signature for row in attempts]),
    }
