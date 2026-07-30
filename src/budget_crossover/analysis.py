from __future__ import annotations

"""Statistical analysis and artifact generation."""

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

from .config import ExperimentConfig
from .dataset import case_set_profile
from .io import read_jsonl
from .manifest import ensure_manifest, record_phase, run_dir
from .policy import canonical_decision
from .records import Case, Generation
from .runner import generation_path

SYSTEM_LABELS = {
    "monolith": "Monolithic full-context",
    "retrieval": "Plan-and-retrieve",
    "committee": "Specialist committee",
    "guardrail": "Underwriter + compliance guardrail",
    "adaptive": "Adaptive guarded routing",
    "always_primary": "Always-primary monolith",
    "always_supervisor": "Always-supervisor monolith",
    "selective_supervisor": "Selective supervisor routing",
}


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _bootstrap_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    indexes = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indexes].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def _estimated_cost(generation: Generation, config: ExperimentConfig) -> float | None:
    total = 0.0
    priced = False
    for call in generation.calls:
        model = call.response.model
        price = config.model_prices_per_million.get(model)
        if price is None:
            candidates = [
                value
                for key, value in config.model_prices_per_million.items()
                if model.startswith(key)
            ]
            price = candidates[0] if len(candidates) == 1 else None
        prompt = call.response.usage.prompt_tokens
        completion = call.response.usage.completion_tokens
        if price is None or prompt is None or completion is None:
            continue
        total += (prompt * float(price["input"]) + completion * float(price["output"])) / 1_000_000
        priced = True
    return total if priced else None


def build_frame(
    *,
    cases: list[Case],
    generations: list[Generation],
    config: ExperimentConfig,
) -> pd.DataFrame:
    case_map = {case.case_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    for generation in generations:
        case = case_map.get(generation.case_id)
        if case is None:
            continue
        candidate = canonical_decision(generation.parsed_decision)
        gold = canonical_decision(case.gold_decision)
        decision_correct = (
            candidate is not None and gold is not None and candidate["decision"] == gold["decision"]
        )
        policy_complete = candidate == gold
        action_analogue = (
            "originated"
            if candidate is not None and candidate["decision"] == "approve"
            else (
                "denied"
                if candidate is not None and candidate["decision"] == "deny"
                else "nonfinal_review"
            )
        )
        actual_total = generation.total_tokens
        overrun = bool(generation.diagnostics.get("budget_overrun", False))
        rows.append(
            {
                "run_id": generation.run_id,
                "case_id": case.case_id,
                "pair_id": case.pair_id,
                "counterfactual_variant": case.counterfactual_variant,
                "state": case.state,
                "historical_action": case.historical_action,
                "policy_decision": case.policy_decision,
                "complexity": case.complexity,
                "changed_protected_attribute": case.changed_protected_attribute,
                "system": generation.system,
                "system_label": SYSTEM_LABELS[generation.system],
                "generator_model": generation.model,
                "supervisor_model": generation.supervisor_model,
                "models_used": json.dumps(
                    sorted({call.response.model for call in generation.calls})
                ),
                "credential_slots_used": json.dumps(
                    sorted(
                        {
                            call.response.credential_slot
                            for call in generation.calls
                            if call.response.credential_slot is not None
                        }
                    )
                ),
                "token_budget": generation.token_budget,
                "repetition": generation.repetition,
                "status": generation.status,
                "schema_valid": int(candidate is not None),
                "decision_correct": int(decision_correct),
                "policy_complete": int(policy_complete),
                "candidate_decision": (candidate["decision"] if candidate is not None else None),
                "historical_action_analogue": action_analogue,
                "historical_action_concordant": int(action_analogue == case.historical_action),
                "candidate_policy_json": (
                    json.dumps(candidate, sort_keys=True) if candidate is not None else None
                ),
                "gold_policy_json": json.dumps(gold, sort_keys=True),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": actual_total,
                "call_count": len(generation.calls),
                "wall_time_seconds": generation.wall_time_seconds,
                "summed_api_latency_seconds": (generation.summed_api_latency_seconds),
                "budget_exhausted": generation.status == "budget_exhausted",
                "budget_overrun": overrun,
                "budget_compliant": (
                    not overrun
                    and actual_total is not None
                    and actual_total <= generation.token_budget
                ),
                "escalated": bool(generation.diagnostics.get("escalated", False)),
                "confidence": generation.confidence,
                "estimated_cost_usd": _estimated_cost(generation, config),
            }
        )
    return pd.DataFrame(rows)


def _pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (pair_id, system, budget, repetition), group in frame.groupby(
        ["pair_id", "system", "token_budget", "repetition"]
    ):
        observed = group[group["counterfactual_variant"] == "observed"]
        decisions = group["candidate_policy_json"].dropna().tolist()
        candidate_labels = group["candidate_decision"].dropna().tolist()
        complete_pair = len(group) == 2
        rows.append(
            {
                "pair_id": pair_id,
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "token_budget": budget,
                "repetition": repetition,
                "complete_pair": complete_pair,
                "pair_decision_accuracy": group["decision_correct"].mean(),
                "pair_policy_accuracy": group["policy_complete"].mean(),
                "both_decisions_correct": (
                    int(complete_pair and group["decision_correct"].eq(1).all())
                ),
                "both_policies_complete": (
                    int(complete_pair and group["policy_complete"].eq(1).all())
                ),
                "counterfactual_decision_consistent": int(
                    complete_pair and len(candidate_labels) == 2 and len(set(candidate_labels)) == 1
                ),
                "counterfactual_policy_consistent": int(
                    complete_pair and len(decisions) == 2 and len(set(decisions)) == 1
                ),
                "historical_action": (
                    observed.iloc[0]["historical_action"]
                    if len(observed)
                    else group.iloc[0]["historical_action"]
                ),
                "policy_decision": group.iloc[0]["policy_decision"],
                "complexity": group.iloc[0]["complexity"],
                "state": group.iloc[0]["state"],
            }
        )
    return pd.DataFrame(rows)


def _summary(
    frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
    case_count: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    expected_cells = case_count * config.repetitions
    for system in config.systems:
        for budget in config.token_budgets:
            group = frame[(frame["system"] == system) & (frame["token_budget"] == budget)]
            pairs = pair_frame[
                (pair_frame["system"] == system) & (pair_frame["token_budget"] == budget)
            ]
            pair_clusters = (
                pairs.groupby("pair_id")["pair_decision_accuracy"].mean()
                if not pairs.empty
                else pd.Series(dtype=float)
            )
            low, high = _bootstrap_mean(
                pair_clusters.to_numpy(float),
                rng=rng,
                replicates=config.bootstrap_replicates,
            )
            completed = group[group["status"] == "ok"]
            completed_accuracy = (
                completed["decision_correct"].mean() if len(completed) else math.nan
            )
            itt_accuracy = (
                group["decision_correct"].sum() / expected_cells if expected_cells else math.nan
            )
            consistency = (
                pairs["counterfactual_decision_consistent"].mean() if len(pairs) else math.nan
            )
            rows.append(
                {
                    "system": system,
                    "system_label": SYSTEM_LABELS[system],
                    "token_budget": budget,
                    "cases": len(group),
                    "expected_cells": expected_cells,
                    "coverage_rate": len(group) / expected_cells,
                    "completed_decision_rate": len(completed) / expected_cells,
                    "infrastructure_missing_rate": (expected_cells - len(group)) / expected_cells,
                    "pairs": len(pairs),
                    "decision_accuracy": itt_accuracy,
                    "intention_to_treat_accuracy": itt_accuracy,
                    "conditional_decision_accuracy": completed_accuracy,
                    "decision_accuracy_ci_low": low,
                    "decision_accuracy_ci_high": high,
                    "policy_complete_accuracy": (group["policy_complete"].sum() / expected_cells),
                    "historical_action_concordance": group["historical_action_concordant"].mean(),
                    "both_twins_decision_correct": pairs["both_decisions_correct"].mean(),
                    "counterfactual_decision_consistency": consistency,
                    "counterfactual_flip_rate": (
                        1 - consistency if not pd.isna(consistency) else math.nan
                    ),
                    "counterfactual_policy_consistency": pairs[
                        "counterfactual_policy_consistent"
                    ].mean(),
                    "schema_validity": group["schema_valid"].mean(),
                    "resource_abstention_rate": (group["budget_exhausted"].sum() / expected_cells),
                    "budget_exhaustion_rate": group["budget_exhausted"].mean(),
                    "budget_overrun_rate": group["budget_overrun"].mean(),
                    "budget_compliance_rate": group["budget_compliant"].mean(),
                    "mean_prompt_tokens": group["prompt_tokens"].mean(),
                    "mean_completion_tokens": group["completion_tokens"].mean(),
                    "mean_total_tokens": group["total_tokens"].mean(),
                    "mean_call_count": group["call_count"].mean(),
                    "mean_wall_time_seconds": group["wall_time_seconds"].mean(),
                    "mean_summed_api_latency_seconds": group["summed_api_latency_seconds"].mean(),
                    "mean_estimated_cost_usd": group["estimated_cost_usd"].mean(),
                    "escalation_rate": group["escalated"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["token_budget", "system"])


def _paired_comparisons(
    pair_frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1)
    rows: list[dict[str, Any]] = []
    baseline = "monolith" if config.study_kind == "architecture" else "always_primary"
    for budget in config.token_budgets:
        budget_frame = pair_frame[pair_frame["token_budget"] == budget]
        for system in sorted(set(budget_frame["system"]) - {baseline}):
            subset = budget_frame[budget_frame["system"].isin([baseline, system])]
            mean_pivot = subset.pivot_table(
                index="pair_id",
                columns="system",
                values="both_decisions_correct",
                aggfunc="mean",
            ).dropna()
            exact_pivot = subset.pivot_table(
                index="pair_id",
                columns="system",
                values="both_decisions_correct",
                aggfunc="min",
            ).dropna()
            if (
                baseline not in mean_pivot
                or system not in mean_pivot
                or baseline not in exact_pivot
                or system not in exact_pivot
            ):
                continue
            paired = mean_pivot[system] - mean_pivot[baseline]
            low, high = _bootstrap_mean(
                paired.to_numpy(float),
                rng=rng,
                replicates=config.bootstrap_replicates,
            )
            improved = int(((exact_pivot[baseline] == 0) & (exact_pivot[system] == 1)).sum())
            regressed = int(((exact_pivot[baseline] == 1) & (exact_pivot[system] == 0)).sum())
            discordant = improved + regressed
            p_value = (
                float(
                    binomtest(
                        min(improved, regressed),
                        n=discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            discordance = discordant / len(exact_pivot) if len(exact_pivot) else math.nan
            mde = (
                math.sqrt((1.959964 + 0.841621) ** 2 * discordance / len(exact_pivot))
                if len(exact_pivot)
                else math.nan
            )
            rows.append(
                {
                    "token_budget": budget,
                    "system": system,
                    "system_label": SYSTEM_LABELS[system],
                    "baseline": baseline,
                    "paired_applications": len(mean_pivot),
                    "accuracy_difference_vs_baseline": paired.mean(),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "improved_applications": improved,
                    "regressed_applications": regressed,
                    "mcnemar_exact_p_value": p_value,
                    "mcnemar_success_rule": "all_repetitions_both_twins_correct",
                    "discordance_rate": discordance,
                    "approx_mde_80pct": mde,
                }
            )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["mcnemar_holm_p_value"] = math.nan
    for indexes in output.groupby("system").groups.values():
        ordered = sorted(
            indexes,
            key=lambda index: float(output.loc[index, "mcnemar_exact_p_value"]),
        )
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0,
                (count - rank) * float(output.loc[index, "mcnemar_exact_p_value"]),
            )
            running = max(running, adjusted)
            output.loc[index, "mcnemar_holm_p_value"] = running
    return output


def _crossing_budget(
    budgets: list[int],
    differences: list[float],
) -> float | None:
    previous_budget: int | None = None
    previous_difference: float | None = None
    for budget, difference in zip(budgets, differences, strict=True):
        if not math.isfinite(difference):
            continue
        if difference == 0:
            if previous_difference is not None:
                return float(budget)
            continue
        if (
            previous_budget is not None
            and previous_difference is not None
            and previous_difference * difference < 0
        ):
            fraction = -previous_difference / (difference - previous_difference)
            return float(previous_budget + fraction * (budget - previous_budget))
        previous_budget = budget
        previous_difference = difference
    return None


def _crossover_estimates(
    pair_frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
) -> pd.DataFrame:
    baseline = "monolith" if config.study_kind == "architecture" else "always_primary"
    aggregated = (
        pair_frame.groupby(["pair_id", "system", "token_budget"])["both_decisions_correct"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 2)
    for system in sorted(set(aggregated["system"]) - {baseline}):
        subset = aggregated[aggregated["system"].isin([baseline, system])]
        pivot = subset.pivot_table(
            index="pair_id",
            columns=["system", "token_budget"],
            values="both_decisions_correct",
            aggfunc="mean",
        )
        budgets = [
            budget
            for budget in config.token_budgets
            if (baseline, budget) in pivot.columns and (system, budget) in pivot.columns
        ]
        if not budgets:
            continue
        differences = pd.DataFrame(
            {budget: pivot[(system, budget)] - pivot[(baseline, budget)] for budget in budgets}
        ).dropna()
        if differences.empty:
            continue
        estimate = _crossing_budget(
            budgets,
            differences.mean(axis=0).tolist(),
        )
        indexes = rng.integers(
            0,
            len(differences),
            size=(config.bootstrap_replicates, len(differences)),
        )
        bootstrap_means = differences.to_numpy(float)[indexes].mean(axis=1)
        bootstrap_crossings = [
            crossing
            for crossing in (_crossing_budget(budgets, row.tolist()) for row in bootstrap_means)
            if crossing is not None
        ]
        if bootstrap_crossings:
            low, high = np.quantile(
                bootstrap_crossings,
                [0.025, 0.975],
            ).tolist()
        else:
            low, high = math.nan, math.nan
        rows.append(
            {
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "baseline": baseline,
                "paired_applications": len(differences),
                "crossover_detected": estimate is not None,
                "crossover_budget": estimate,
                "crossover_ci_low": low,
                "crossover_ci_high": high,
                "bootstrap_crossover_support_rate": (
                    len(bootstrap_crossings) / config.bootstrap_replicates
                ),
            }
        )
    return pd.DataFrame(rows)


def _mechanisms(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, budget, complexity), group in frame.groupby(
        ["system", "token_budget", "complexity"]
    ):
        rows.append(
            {
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "token_budget": budget,
                "complexity": complexity,
                "cases": len(group),
                "decision_accuracy": group["decision_correct"].mean(),
                "policy_complete_accuracy": group["policy_complete"].mean(),
                "historical_action_concordance": group["historical_action_concordant"].mean(),
                "schema_validity": group["schema_valid"].mean(),
                "budget_exhaustion_rate": group["budget_exhausted"].mean(),
                "mean_total_tokens": group["total_tokens"].mean(),
                "escalation_rate": group["escalated"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _frontier(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget, group in summary.groupby("token_budget"):
        for _, candidate in group.iterrows():
            if pd.isna(candidate["mean_total_tokens"]):
                dominated = True
            else:
                dominated = any(
                    other["decision_accuracy"] >= candidate["decision_accuracy"]
                    and other["mean_total_tokens"] <= candidate["mean_total_tokens"]
                    and (
                        other["decision_accuracy"] > candidate["decision_accuracy"]
                        or other["mean_total_tokens"] < candidate["mean_total_tokens"]
                    )
                    for _, other in group.drop(candidate.name).iterrows()
                    if not pd.isna(other["mean_total_tokens"])
                )
            rows.append(
                {
                    "token_budget": budget,
                    "system": candidate["system"],
                    "system_label": candidate["system_label"],
                    "decision_accuracy": candidate["decision_accuracy"],
                    "mean_total_tokens": candidate["mean_total_tokens"],
                    "pareto_efficient_within_budget": not dominated,
                }
            )
    return pd.DataFrame(rows)


def _write_figures(
    summary: pd.DataFrame,
    output: Path,
    *,
    diagnostic: bool,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    output.mkdir(parents=True, exist_ok=True)
    palette = sns.color_palette("colorblind", n_colors=summary["system"].nunique())
    labels = list(dict.fromkeys(summary["system"]))
    colors = dict(zip(labels, palette, strict=True))
    markers = dict(zip(labels, ["o", "s", "^", "D", "P", "X"][: len(labels)], strict=True))

    def mark(fig: plt.Figure) -> None:
        if diagnostic:
            fig.text(
                0.5,
                0.5,
                "INCOMPLETE DIAGNOSTIC",
                ha="center",
                va="center",
                fontsize=20,
                color="crimson",
                alpha=0.24,
                rotation=24,
            )

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for system, group in summary.groupby("system"):
        group = group.sort_values("token_budget")
        ax.plot(
            group["token_budget"],
            group["decision_accuracy"],
            marker=markers[system],
            label=SYSTEM_LABELS[system],
            color=colors[system],
        )
        ax.fill_between(
            group["token_budget"],
            group["decision_accuracy_ci_low"],
            group["decision_accuracy_ci_high"],
            alpha=0.14,
            color=colors[system],
        )
    ax.set(
        xlabel="Nominal total-token budget",
        ylabel="Decision accuracy",
        ylim=(-0.02, 1.02),
        xticks=sorted(summary["token_budget"].unique()),
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    mark(fig)
    fig.savefig(output / "decision_accuracy_by_budget.png", dpi=220)
    fig.savefig(output / "decision_accuracy_by_budget.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for system, group in summary.groupby("system"):
        group = group.sort_values("token_budget")
        ax.plot(
            group["token_budget"],
            group["counterfactual_flip_rate"],
            marker=markers[system],
            label=SYSTEM_LABELS[system],
            color=colors[system],
        )
    ax.set(
        xlabel="Nominal total-token budget",
        ylabel="Counterfactual decision flip rate (lower is better)",
        ylim=(-0.02, 1.02),
        xticks=sorted(summary["token_budget"].unique()),
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    mark(fig)
    fig.savefig(output / "counterfactual_flip_rate_by_budget.png", dpi=220)
    fig.savefig(output / "counterfactual_flip_rate_by_budget.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for system, group in summary.groupby("system"):
        ax.scatter(
            group["mean_total_tokens"],
            group["decision_accuracy"],
            label=SYSTEM_LABELS[system],
            color=colors[system],
            marker=markers[system],
            s=44,
        )
    ax.set(
        xlabel="Mean realized total tokens",
        ylabel="Decision accuracy",
        ylim=(-0.02, 1.02),
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    mark(fig)
    fig.savefig(output / "tokens_vs_accuracy.png", dpi=220)
    fig.savefig(output / "tokens_vs_accuracy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for system, group in summary.groupby("system"):
        group = group.sort_values("token_budget")
        ax.plot(
            group["token_budget"],
            group["coverage_rate"],
            marker=markers[system],
            label=SYSTEM_LABELS[system],
            color=colors[system],
        )
    ax.set(
        xlabel="Nominal total-token budget",
        ylabel="Scored-grid coverage",
        ylim=(-0.02, 1.02),
        xticks=sorted(summary["token_budget"].unique()),
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    mark(fig)
    fig.savefig(output / "coverage_by_budget.png", dpi=220)
    fig.savefig(output / "coverage_by_budget.pdf")
    plt.close(fig)


def analyze_run(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
    diagnostic: bool = False,
) -> dict[str, Any]:
    ensure_manifest(repo, config, cases)
    generations = read_jsonl(generation_path(repo, config), Generation)
    if not generations:
        raise RuntimeError("no generations are available to analyze")
    expected_grid = {
        (case.case_id, system, budget, repetition)
        for case in cases
        for system in config.systems
        for budget in config.token_budgets
        for repetition in range(config.repetitions)
    }
    observed_grid = [
        (row.case_id, row.system, row.token_budget, row.repetition) for row in generations
    ]
    missing = expected_grid - set(observed_grid)
    extra = set(observed_grid) - expected_grid
    duplicates = len(observed_grid) - len(set(observed_grid))
    incomplete = bool(missing or extra or duplicates)
    if incomplete and not diagnostic:
        raise RuntimeError(
            "incomplete generation grid: "
            f"{len(missing)} missing, {len(extra)} extra, "
            f"{duplicates} duplicates"
        )
    frame = build_frame(cases=cases, generations=generations, config=config)
    pair_frame = _pair_frame(frame)
    summary = _summary(
        frame,
        pair_frame,
        config=config,
        case_count=len(cases),
    )
    comparisons = _paired_comparisons(pair_frame, config=config)
    crossovers = _crossover_estimates(pair_frame, config=config)
    mechanisms = _mechanisms(frame)
    frontier = _frontier(summary)

    analysis_dir = run_dir(repo, config) / "analysis"
    tables = analysis_dir / "tables"
    figures = analysis_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "case_results.csv", index=False)
    pair_frame.to_csv(tables / "pair_results.csv", index=False)
    summary.to_csv(tables / "system_summary.csv", index=False)
    summary[
        [
            "system",
            "token_budget",
            "expected_cells",
            "cases",
            "coverage_rate",
            "completed_decision_rate",
            "infrastructure_missing_rate",
            "resource_abstention_rate",
        ]
    ].to_csv(tables / "coverage.csv", index=False)
    comparisons.to_csv(tables / "paired_comparisons.csv", index=False)
    crossovers.to_csv(tables / "crossover_estimates.csv", index=False)
    mechanisms.to_csv(tables / "mechanisms.csv", index=False)
    frontier.to_csv(tables / "pareto_frontier.csv", index=False)
    _write_figures(summary, figures, diagnostic=diagnostic)

    expected = len(expected_grid)
    report = {
        "experiment": config.experiment_name,
        "case_profile": case_set_profile(cases),
        "expected_generations": expected,
        "observed_generations": len(generations),
        "generation_completion_rate": len(generations) / expected if expected else 0,
        "diagnostic": diagnostic,
        "incomplete": incomplete,
        "missing_generation_cells": len(missing),
        "extra_generation_cells": len(extra),
        "duplicate_generation_cells": duplicates,
        "unique_generation_cells": int(
            frame[["case_id", "system", "token_budget", "repetition"]].drop_duplicates().shape[0]
        ),
        "primary_comparison": (
            "adaptive versus monolith within each token budget"
            if config.study_kind == "architecture"
            else "routing policies versus always-primary within each token budget"
        ),
        "primary_unit": "HMDA source application (counterfactual pair)",
        "results_are_empirical": config.execution_mode == "gateway",
    }
    report_path = analysis_dir / "analysis.json"
    report_path.write_text(json.dumps(_json_native(report), indent=2, sort_keys=True))
    record_phase(repo, config, phase="analysis", counters=report)
    return report
