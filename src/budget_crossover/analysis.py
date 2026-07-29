from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial

from .config import ExperimentConfig
from .evaluate import exact_evaluate
from .io import read_jsonl
from .judging import judge_path
from .models import Case, ExperimentResult, JudgeResult
from .runner import generation_path

ARCH_LABELS = {
    "direct": "Direct",
    "self_critique": "Self-critique",
    "debate": "Two-agent debate",
}
ARCH_COLORS = {
    "direct": "#1F4E79",
    "self_critique": "#2A9D8F",
    "debate": "#E76F51",
}


def _bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float]:
    if len(values) == 0:
        return (math.nan, math.nan)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def _crossing(budgets: list[int], differences: list[float]) -> float | None:
    if not differences or all(not np.isfinite(value) for value in differences):
        return None
    if np.isfinite(differences[0]) and differences[0] > 0:
        return float(budgets[0])
    for left, right, first, second in zip(
        budgets[:-1], budgets[1:], differences[:-1], differences[1:]
    ):
        if not (np.isfinite(first) and np.isfinite(second)):
            continue
        if first <= 0 < second:
            x1, x2 = math.log2(left), math.log2(right)
            x = x1 + (-first) * (x2 - x1) / (second - first)
            return float(2**x)
    return None


def _judgment_consensus(
    judgments: list[JudgeResult],
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[JudgeResult]] = {}
    for row in judgments:
        grouped.setdefault(row.run_id, []).append(row)
    consensus: dict[str, dict[str, Any]] = {}
    for run_id, rows in grouped.items():
        base = {row.judge_model: row for row in rows if row.judge_model in config.judge_models}
        adjudicator = next(
            (row for row in rows if row.judge_model == config.adjudicator_model), None
        )
        values = [base[model].correct for model in config.judge_models if model in base]
        disagreement = len(values) == len(config.judge_models) and len(set(values)) > 1
        if len(values) == len(config.judge_models) and not disagreement:
            correct: bool | None = values[0]
        elif disagreement and adjudicator is not None:
            correct = adjudicator.correct
        else:
            correct = None
        evidence_values = [row.evidence_score for row in base.values()]
        consensus[run_id] = {
            "judge_correct": correct,
            "judge_disagreement": disagreement,
            "judge_evidence_score": (
                float(np.mean(evidence_values)) if evidence_values else math.nan
            ),
            "judge_unsupported_rate": (
                float(np.mean([row.unsupported_claims for row in base.values()]))
                if base
                else math.nan
            ),
        }
    return consensus


def build_case_level_frame(
    *,
    generations: list[ExperimentResult],
    cases: list[Case],
    judgments: list[JudgeResult],
    config: ExperimentConfig,
) -> pd.DataFrame:
    case_map = {case.case_id: case for case in cases}
    consensus = _judgment_consensus(judgments, config)
    rows: list[dict[str, Any]] = []
    for generation in generations:
        if generation.status != "ok" or generation.case_id not in case_map:
            continue
        case = case_map[generation.case_id]
        exact = exact_evaluate(
            case.task,
            generation.parsed_answer,
            case.accepted_reference_answers,
        )
        complexity = math.log1p(case.evidence_chars) + 0.25 * case.tool_evidence_count
        row = {
            "run_id": generation.run_id,
            "case_id": generation.case_id,
            "task": case.task,
            "architecture": generation.architecture,
            "nominal_budget": generation.nominal_budget,
            "log2_budget": math.log2(generation.nominal_budget),
            "answer": generation.parsed_answer,
            "exact_correct": int(exact.correct),
            "candidate_value": exact.candidate_value,
            "reference_values": " | ".join(exact.reference_values),
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.total_tokens,
            "latency_seconds": generation.latency_seconds,
            "call_count": len(generation.calls),
            "evidence_chars": case.evidence_chars,
            "tool_evidence_count": case.tool_evidence_count,
            "complexity_raw": complexity,
            **consensus.get(
                generation.run_id,
                {
                    "judge_correct": None,
                    "judge_disagreement": False,
                    "judge_evidence_score": math.nan,
                    "judge_unsupported_rate": math.nan,
                },
            ),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        standard_deviation = frame["complexity_raw"].std(ddof=0)
        frame["complexity_z"] = (frame["complexity_raw"] - frame["complexity_raw"].mean()) / (
            standard_deviation if standard_deviation else 1.0
        )
    return frame


def _summary(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (architecture, budget), group in frame.groupby(["architecture", "nominal_budget"]):
        values = group["exact_correct"].to_numpy(dtype=float)
        low, high = _bootstrap_mean(values, rng, replicates)
        rows.append(
            {
                "architecture": architecture,
                "architecture_label": ARCH_LABELS[architecture],
                "nominal_budget": int(budget),
                "n": len(group),
                "accuracy": values.mean(),
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "judge_accuracy": pd.to_numeric(group["judge_correct"], errors="coerce").mean(),
                "evidence_score": group["judge_evidence_score"].mean(),
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_total_tokens": group["total_tokens"].mean(),
                "mean_latency_seconds": group["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["architecture", "nominal_budget"])


def _paired_differences(
    frame: pd.DataFrame,
    *,
    budgets: list[int],
    architectures: list[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, Any]] = []
    crossover_rows: list[dict[str, Any]] = []
    complex_architectures = [name for name in architectures if name != "direct"]
    for architecture in complex_architectures:
        difference_by_budget: list[float] = []
        paired_by_budget: dict[int, pd.Series] = {}
        for budget in budgets:
            subset = frame[
                (frame["nominal_budget"] == budget)
                & frame["architecture"].isin(["direct", architecture])
            ]
            pivot = subset.pivot_table(
                index="case_id",
                columns="architecture",
                values="exact_correct",
                aggfunc="first",
            ).dropna()
            paired = pivot[architecture] - pivot["direct"]
            paired_by_budget[budget] = paired
            difference = float(paired.mean()) if len(paired) else math.nan
            difference_by_budget.append(difference)
            low, high = _bootstrap_mean(paired.to_numpy(float), rng, replicates)
            rows.append(
                {
                    "architecture": architecture,
                    "architecture_label": ARCH_LABELS[architecture],
                    "nominal_budget": budget,
                    "paired_n": len(paired),
                    "accuracy_difference_vs_direct": difference,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                }
            )
        estimate = _crossing(budgets, difference_by_budget)
        shared_cases = (
            sorted(set.intersection(*(set(series.index) for series in paired_by_budget.values())))
            if paired_by_budget
            else []
        )
        bootstrap_crossings: list[float] = []
        if shared_cases:
            matrix = np.column_stack(
                [
                    paired_by_budget[budget].reindex(shared_cases).to_numpy(float)
                    for budget in budgets
                ]
            )
            for _ in range(replicates):
                sample = rng.integers(0, len(shared_cases), len(shared_cases))
                crossing = _crossing(budgets, matrix[sample].mean(axis=0).tolist())
                if crossing is not None:
                    bootstrap_crossings.append(crossing)
        if bootstrap_crossings:
            low, high = np.quantile(bootstrap_crossings, [0.025, 0.975]).tolist()
        else:
            low, high = math.nan, math.nan
        crossover_rows.append(
            {
                "architecture": architecture,
                "architecture_label": ARCH_LABELS[architecture],
                "crossover_budget": estimate,
                "crossover_ci_low": low,
                "crossover_ci_high": high,
                "bootstrap_crossing_probability": (
                    len(bootstrap_crossings) / replicates if replicates else math.nan
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(crossover_rows)


def _fit_model(frame: pd.DataFrame) -> tuple[pd.DataFrame, Any | None]:
    if frame.empty or frame["exact_correct"].nunique() < 2:
        return pd.DataFrame(), None
    model = smf.glm(
        "exact_correct ~ C(architecture, Treatment(reference='direct')) * "
        "log2_budget + C(task) + complexity_z",
        data=frame,
        family=Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["case_id"]})
    conf = model.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
        }
    )
    return coefficients, model


def _plot_accuracy(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(7.2, 3.9), dpi=180)
    for architecture in ["direct", "self_critique", "debate"]:
        group = summary[summary["architecture"] == architecture].sort_values("nominal_budget")
        if group.empty:
            continue
        axis.plot(
            group["nominal_budget"],
            group["accuracy"],
            marker="o",
            linewidth=2.2,
            label=ARCH_LABELS[architecture],
            color=ARCH_COLORS[architecture],
        )
        axis.fill_between(
            group["nominal_budget"],
            group["accuracy_ci_low"],
            group["accuracy_ci_high"],
            color=ARCH_COLORS[architecture],
            alpha=0.14,
            linewidth=0,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(summary["nominal_budget"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Per-case completion-token ceiling")
    axis.set_ylabel("Exact decision accuracy")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _plot_tradeoff(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=180)
    for architecture in ["direct", "self_critique", "debate"]:
        group = summary[summary["architecture"] == architecture].sort_values("nominal_budget")
        if group.empty:
            continue
        axes[0].plot(
            group["mean_total_tokens"],
            group["accuracy"],
            marker="o",
            color=ARCH_COLORS[architecture],
            label=ARCH_LABELS[architecture],
        )
        axes[1].plot(
            group["mean_latency_seconds"],
            group["accuracy"],
            marker="o",
            color=ARCH_COLORS[architecture],
            label=ARCH_LABELS[architecture],
        )
    axes[0].set_xlabel("Mean measured total tokens")
    axes[0].set_ylabel("Exact decision accuracy")
    axes[1].set_xlabel("Mean summed call latency (s)")
    axes[1].set_ylabel("")
    for axis in axes:
        axis.set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def analyze_run(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
) -> dict[str, Any]:
    generations = read_jsonl(generation_path(repo, config), ExperimentResult)
    judgments = read_jsonl(judge_path(repo, config), JudgeResult)
    frame = build_case_level_frame(
        generations=generations,
        cases=cases,
        judgments=judgments,
        config=config,
    )
    if frame.empty:
        raise RuntimeError("no successful generations are available to analyze")
    output_dir = repo / "outputs"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_dir / "case_level_results.csv", index=False)
    summary = _summary(
        frame,
        replicates=config.bootstrap_replicates,
        seed=config.seed,
    )
    summary.to_csv(table_dir / "architecture_by_budget.csv", index=False)
    differences, crossovers = _paired_differences(
        frame,
        budgets=config.budgets,
        architectures=config.architectures,
        replicates=config.bootstrap_replicates,
        seed=config.seed,
    )
    differences.to_csv(table_dir / "paired_differences.csv", index=False)
    crossovers.to_csv(table_dir / "crossovers.csv", index=False)
    coefficients, model = _fit_model(frame)
    coefficients.to_csv(table_dir / "adjusted_logistic_model.csv", index=False)
    _plot_accuracy(summary, figure_dir / "accuracy_by_budget.png")
    if summary["mean_total_tokens"].notna().any():
        _plot_tradeoff(summary, figure_dir / "resource_tradeoff.png")

    task_summary = frame.groupby(["task", "architecture", "nominal_budget"], as_index=False).agg(
        n=("exact_correct", "size"), accuracy=("exact_correct", "mean")
    )
    task_summary.to_csv(table_dir / "task_breakdown.csv", index=False)
    judge_disagreement_rate = float(frame["judge_disagreement"].mean())
    report = {
        "experiment": config.experiment_name,
        "successful_generations": len(frame),
        "expected_generations": int(
            config.sample_size * len(config.architectures) * len(config.budgets)
        ),
        "unique_cases": int(frame["case_id"].nunique()),
        "judge_disagreement_rate": judge_disagreement_rate,
        "crossovers": crossovers.where(pd.notna(crossovers), None).to_dict(orient="records"),
        "model_converged": bool(model.converged) if model is not None else None,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
