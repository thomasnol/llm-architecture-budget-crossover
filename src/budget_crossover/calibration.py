from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import read_jsonl
from .manifest import run_dir
from .records import Generation
from .runner import generation_path


def _round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def recommend_budgets(
    *,
    trajectories: list[dict[str, Any]],
    architecture_minima: dict[str, int],
    round_to: int = 256,
) -> dict[str, Any]:
    """Recommend four logistically feasible budgets from observed trajectories."""
    if round_to < 1:
        raise ValueError("round_to must be positive")
    costs = np.asarray(
        [float(row["total_tokens"]) for row in trajectories if row.get("total_tokens") is not None],
        dtype=float,
    )
    if not len(costs):
        raise ValueError("at least one realized trajectory cost is required")
    if not architecture_minima:
        raise ValueError("architecture minima are required")
    floor = max(int(value) for value in architecture_minima.values())
    raw = np.quantile(costs, [0.25, 0.50, 0.75, 0.95]).tolist()
    budgets: list[int] = []
    for value in raw:
        candidate = _round_up(max(float(floor), float(value)), round_to)
        if budgets and candidate <= budgets[-1]:
            candidate = budgets[-1] + round_to
        budgets.append(candidate)
    return {
        "method": "pooled realized-token quantiles constrained by architecture minima",
        "quantiles": [0.25, 0.50, 0.75, 0.95],
        "round_to": round_to,
        "trajectory_count": len(costs),
        "architecture_minima": dict(sorted(architecture_minima.items())),
        "feasibility_floor": floor,
        "recommended_budgets": budgets,
    }


def calibrate_run(
    *,
    repo: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    generations = [
        row
        for row in read_jsonl(generation_path(repo, config), Generation)
        if row.status == "ok" and row.total_tokens is not None
    ]
    if not generations:
        raise RuntimeError("no successful calibration trajectories are available")
    trajectories = [{"system": row.system, "total_tokens": row.total_tokens} for row in generations]
    minima = {
        system: min(
            int(row.total_tokens)
            for row in generations
            if row.system == system and row.total_tokens is not None
        )
        for system in sorted({row.system for row in generations})
    }
    report = {
        "experiment": config.experiment_name,
        **recommend_budgets(
            trajectories=trajectories,
            architecture_minima=minima,
        ),
    }
    output = run_dir(repo, config) / "calibration.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
