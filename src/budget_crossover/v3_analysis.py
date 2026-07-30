from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

from .io import read_jsonl
from .v3_config import V3Config
from .v3_dataset import case_set_profile
from .v3_manifest import ensure_v3_manifest, record_v3_phase, v3_run_dir
from .v3_models import V3Case, V3Generation
from .v3_policy import canonical_decision
from .v3_runner import v3_generation_path

SYSTEM_LABELS = {
    "monolith": "Monolithic full-context",
    "strong_monolith": "Strong-model monolith",
    "retrieval": "Plan-and-retrieve",
    "committee": "Specialist committee",
    "guardrail": "Underwriter + compliance guardrail",
    "adaptive": "Adaptive guarded routing",
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


def _estimated_cost(generation: V3Generation, config: V3Config) -> float | None:
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
        total += (
            prompt * float(price["input"]) + completion * float(price["output"])
        ) / 1_000_000
        priced = True
    return total if priced else None


def build_v3_frame(
    *,
    cases: list[V3Case],
    generations: list[V3Generation],
    config: V3Config,
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
            candidate is not None
            and gold is not None
            and candidate["decision"] == gold["decision"]
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
                "status": generation.status,
                "schema_valid": int(candidate is not None),
                "decision_correct": int(decision_correct),
                "policy_complete": int(policy_complete),
                "candidate_decision": (
                    candidate["decision"] if candidate is not None else None
                ),
                "historical_action_analogue": action_analogue,
                "historical_action_concordant": int(
                    action_analogue == case.historical_action
                ),
                "candidate_policy_json": (
                    json.dumps(candidate, sort_keys=True)
                    if candidate is not None
                    else None
                ),
                "gold_policy_json": json.dumps(gold, sort_keys=True),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": actual_total,
                "call_count": len(generation.calls),
                "wall_time_seconds": generation.wall_time_seconds,
                "summed_api_latency_seconds": (
                    generation.summed_api_latency_seconds
                ),
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
    for (pair_id, system, budget), group in frame.groupby(
        ["pair_id", "system", "token_budget"]
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
                    complete_pair
                    and len(candidate_labels) == 2
                    and len(set(candidate_labels)) == 1
                ),
                "counterfactual_policy_consistent": int(
                    complete_pair
                    and len(decisions) == 2
                    and len(set(decisions)) == 1
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
    config: V3Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    for (system, budget), group in frame.groupby(["system", "token_budget"]):
        pairs = pair_frame[
            (pair_frame["system"] == system)
            & (pair_frame["token_budget"] == budget)
        ]
        low, high = _bootstrap_mean(
            pairs["pair_decision_accuracy"].to_numpy(float),
            rng=rng,
            replicates=config.bootstrap_replicates,
        )
        rows.append(
            {
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "token_budget": budget,
                "cases": len(group),
                "pairs": len(pairs),
                "decision_accuracy": group["decision_correct"].mean(),
                "decision_accuracy_ci_low": low,
                "decision_accuracy_ci_high": high,
                "policy_complete_accuracy": group["policy_complete"].mean(),
                "historical_action_concordance": group[
                    "historical_action_concordant"
                ].mean(),
                "both_twins_decision_correct": pairs[
                    "both_decisions_correct"
                ].mean(),
                "counterfactual_decision_consistency": pairs[
                    "counterfactual_decision_consistent"
                ].mean(),
                "counterfactual_policy_consistency": pairs[
                    "counterfactual_policy_consistent"
                ].mean(),
                "schema_validity": group["schema_valid"].mean(),
                "budget_exhaustion_rate": group["budget_exhausted"].mean(),
                "budget_overrun_rate": group["budget_overrun"].mean(),
                "budget_compliance_rate": group["budget_compliant"].mean(),
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_total_tokens": group["total_tokens"].mean(),
                "mean_call_count": group["call_count"].mean(),
                "mean_wall_time_seconds": group["wall_time_seconds"].mean(),
                "mean_summed_api_latency_seconds": group[
                    "summed_api_latency_seconds"
                ].mean(),
                "mean_estimated_cost_usd": group["estimated_cost_usd"].mean(),
                "escalation_rate": group["escalated"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["token_budget", "system"])


def _paired_comparisons(
    pair_frame: pd.DataFrame,
    *,
    config: V3Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1)
    rows: list[dict[str, Any]] = []
    for budget in config.token_budgets:
        budget_frame = pair_frame[pair_frame["token_budget"] == budget]
        for system in sorted(set(budget_frame["system"]) - {"monolith"}):
            subset = budget_frame[
                budget_frame["system"].isin(["monolith", system])
            ]
            pivot = subset.pivot_table(
                index="pair_id",
                columns="system",
                values="both_decisions_correct",
                aggfunc="first",
            ).dropna()
            if "monolith" not in pivot or system not in pivot:
                continue
            paired = pivot[system] - pivot["monolith"]
            low, high = _bootstrap_mean(
                paired.to_numpy(float),
                rng=rng,
                replicates=config.bootstrap_replicates,
            )
            improved = int(
                ((pivot["monolith"] == 0) & (pivot[system] == 1)).sum()
            )
            regressed = int(
                ((pivot["monolith"] == 1) & (pivot[system] == 0)).sum()
            )
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
            discordance = discordant / len(pivot) if len(pivot) else math.nan
            mde = (
                math.sqrt(
                    (1.959964 + 0.841621) ** 2 * discordance / len(pivot)
                )
                if len(pivot)
                else math.nan
            )
            rows.append(
                {
                    "token_budget": budget,
                    "system": system,
                    "system_label": SYSTEM_LABELS[system],
                    "paired_applications": len(pivot),
                    "both_twins_accuracy_difference_vs_monolith": paired.mean(),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "improved_applications": improved,
                    "regressed_applications": regressed,
                    "mcnemar_exact_p_value": p_value,
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
                (count - rank)
                * float(output.loc[index, "mcnemar_exact_p_value"]),
            )
            running = max(running, adjusted)
            output.loc[index, "mcnemar_holm_p_value"] = running
    return output


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
                "historical_action_concordance": group[
                    "historical_action_concordant"
                ].mean(),
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


def _write_figures(summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    output.mkdir(parents=True, exist_ok=True)
    palette = sns.color_palette("colorblind", n_colors=summary["system"].nunique())
    labels = list(dict.fromkeys(summary["system"]))
    colors = dict(zip(labels, palette, strict=True))
    markers = dict(
        zip(labels, ["o", "s", "^", "D", "P", "X"][: len(labels)], strict=True)
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
    fig.savefig(output / "decision_accuracy_by_budget.png", dpi=220)
    fig.savefig(output / "decision_accuracy_by_budget.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for system, group in summary.groupby("system"):
        group = group.sort_values("token_budget")
        ax.plot(
            group["token_budget"],
            group["counterfactual_decision_consistency"],
            marker=markers[system],
            label=SYSTEM_LABELS[system],
            color=colors[system],
        )
    ax.set(
        xlabel="Nominal total-token budget",
        ylabel="Counterfactual decision consistency",
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
    fig.savefig(output / "counterfactual_consistency_by_budget.png", dpi=220)
    fig.savefig(output / "counterfactual_consistency_by_budget.pdf")
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
    fig.savefig(output / "tokens_vs_accuracy.png", dpi=220)
    fig.savefig(output / "tokens_vs_accuracy.pdf")
    plt.close(fig)


def analyze_v3(
    *,
    repo: Path,
    config: V3Config,
    cases: list[V3Case],
) -> dict[str, Any]:
    ensure_v3_manifest(repo, config, cases)
    generations = read_jsonl(v3_generation_path(repo, config), V3Generation)
    if not generations:
        raise RuntimeError("no v3 generations are available to analyze")
    frame = build_v3_frame(cases=cases, generations=generations, config=config)
    pair_frame = _pair_frame(frame)
    summary = _summary(frame, pair_frame, config=config)
    comparisons = _paired_comparisons(pair_frame, config=config)
    mechanisms = _mechanisms(frame)
    frontier = _frontier(summary)

    analysis_dir = v3_run_dir(repo, config) / "analysis"
    tables = analysis_dir / "tables"
    figures = analysis_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "case_results.csv", index=False)
    pair_frame.to_csv(tables / "pair_results.csv", index=False)
    summary.to_csv(tables / "system_summary.csv", index=False)
    comparisons.to_csv(tables / "paired_comparisons.csv", index=False)
    mechanisms.to_csv(tables / "mechanisms.csv", index=False)
    frontier.to_csv(tables / "pareto_frontier.csv", index=False)
    _write_figures(summary, figures)

    expected = len(cases) * len(config.systems) * len(config.token_budgets)
    report = {
        "experiment": config.experiment_name,
        "case_profile": case_set_profile(cases),
        "expected_generations": expected,
        "observed_generations": len(generations),
        "generation_completion_rate": len(generations) / expected if expected else 0,
        "unique_generation_cells": int(
            frame[["case_id", "system", "token_budget"]]
            .drop_duplicates()
            .shape[0]
        ),
        "primary_comparison": "adaptive versus monolith within each token budget",
        "primary_unit": "HMDA source application (counterfactual pair)",
        "results_are_empirical": config.execution_mode == "gateway",
    }
    report_path = analysis_dir / "analysis.json"
    report_path.write_text(json.dumps(_json_native(report), indent=2, sort_keys=True))
    record_v3_phase(repo, config, phase="analysis", counters=report)
    return report
