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
from .v2_config import V2Config
from .v2_judging import select_judging_generations, v2_judgment_path
from .v2_manifest import ensure_run_manifest
from .v2_models import V2Case, V2Generation, V2Judgment
from .v2_runner import v2_generation_path
from .v2_schema import decisions_equal

SYSTEM_LABELS = {
    "direct": "Direct",
    "checklist": "Single-call checklist",
    "strong_direct": "Strong-model direct",
    "self_critique": "Same-model self-critique",
    "external_verify": "Strong external critique",
    "best_of_2": "Best-of-2 + verifier",
    "best_of_4": "Best-of-4 + verifier",
    "adaptive": "Adaptive verify/escalate",
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
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def _cost(generation: V2Generation, config: V2Config) -> float | None:
    total = 0.0
    priced = False
    for call in generation.calls:
        model = call.response.model
        price = config.model_prices_per_million.get(model)
        if price is None:
            # Gateways sometimes return a dated deployment name. Permit an exact
            # configured request-model prefix without inventing public prices.
            candidates = [
                value
                for key, value in config.model_prices_per_million.items()
                if model.startswith(key)
            ]
            price = candidates[0] if len(candidates) == 1 else None
        if price is None:
            continue
        prompt = call.response.usage.prompt_tokens
        completion = call.response.usage.completion_tokens
        if prompt is None or completion is None:
            continue
        total += (
            prompt * float(price["input"]) + completion * float(price["output"])
        ) / 1_000_000
        priced = True
    return total if priced else None


def build_v2_frame(
    *,
    cases: list[V2Case],
    generations: list[V2Generation],
    judgments: list[V2Judgment],
    config: V2Config,
) -> pd.DataFrame:
    case_map = {case.case_id: case for case in cases}
    judge_map: dict[str, list[V2Judgment]] = {}
    for judgment in judgments:
        if judgment.status == "ok":
            judge_map.setdefault(judgment.run_id, []).append(judgment)
    rows: list[dict[str, Any]] = []
    for generation in generations:
        if generation.status != "ok" or generation.case_id not in case_map:
            continue
        case = case_map[generation.case_id]
        exact = decisions_equal(case.task, generation.parsed_decision, case.gold_decision)
        initial = generation.diagnostics.get("initial_decision")
        initial_correct = (
            decisions_equal(case.task, initial, case.gold_decision)
            if initial is not None
            else None
        )
        judge_rows = judge_map.get(generation.run_id, [])
        judge_values = [row.semantically_correct for row in judge_rows]
        rows.append(
            {
                "run_id": generation.run_id,
                "case_id": generation.case_id,
                "dataset": case.dataset,
                "task": case.task,
                "system": generation.system,
                "system_label": SYSTEM_LABELS[generation.system],
                "schema_valid": int(generation.parsed_decision is not None),
                "exact_correct": int(exact),
                "initial_correct": (
                    int(initial_correct) if initial_correct is not None else math.nan
                ),
                "final_decision": json.dumps(
                    generation.parsed_decision, sort_keys=True
                ),
                "gold_decision": json.dumps(case.gold_decision, sort_keys=True),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
                "summed_api_latency_seconds": generation.latency_seconds,
                "wall_time_seconds": generation.wall_time_seconds,
                "call_count": len(generation.calls),
                "estimated_cost_usd": _cost(generation, config),
                "escalated": bool(generation.diagnostics.get("escalated", False)),
                "candidate_disagreement": generation.diagnostics.get(
                    "candidate_disagreement"
                ),
                "verifier_accept": generation.diagnostics.get("verifier_accept"),
                "verifier_confidence": generation.diagnostics.get(
                    "verifier_confidence"
                ),
                "verifier_error_type": generation.diagnostics.get(
                    "verifier_error_type"
                ),
                "evidence_chars": case.evidence_chars,
                "judge_count": len(judge_rows),
                "judge_accuracy": (
                    float(np.mean(judge_values)) if judge_values else math.nan
                ),
                "judge_disagreement": (
                    len(judge_values) == len(config.judge_models)
                    and len(set(judge_values)) > 1
                ),
                "judge_evidence_score": (
                    float(np.mean([row.evidence_score for row in judge_rows]))
                    if judge_rows
                    else math.nan
                ),
                "judge_unsupported_rate": (
                    float(np.mean([row.unsupported_claims for row in judge_rows]))
                    if judge_rows
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _system_summary(
    frame: pd.DataFrame,
    *,
    config: V2Config,
    expected_judged_run_ids: set[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    for (dataset, system), group in frame.groupby(["dataset", "system"]):
        values = group["exact_correct"].to_numpy(float)
        low, high = _bootstrap_mean(
            values, rng=rng, replicates=config.bootstrap_replicates
        )
        task_means = group.groupby("task")["exact_correct"].mean()
        expected_judgments = (
            group["run_id"].isin(expected_judged_run_ids).sum()
            * len(config.judge_models)
        )
        row = {
            "dataset": dataset,
            "system": system,
            "system_label": SYSTEM_LABELS[system],
            "n": len(group),
            "unique_cases": group["case_id"].nunique(),
            "accuracy": values.mean(),
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "task_macro_accuracy": task_means.mean(),
            "schema_validity": group["schema_valid"].mean(),
            "mean_prompt_tokens": group["prompt_tokens"].mean(),
            "mean_completion_tokens": group["completion_tokens"].mean(),
            "mean_total_tokens": group["total_tokens"].mean(),
            "mean_call_count": group["call_count"].mean(),
            "mean_wall_time_seconds": group["wall_time_seconds"].mean(),
            "mean_summed_api_latency_seconds": group[
                "summed_api_latency_seconds"
            ].mean(),
            "mean_cost_usd": group["estimated_cost_usd"].mean(),
            "judge_sampled_generations": group["run_id"]
            .isin(expected_judged_run_ids)
            .sum(),
            "judge_coverage": (
                group["judge_count"].sum() / expected_judgments
                if expected_judgments
                else math.nan
            ),
            "judge_accuracy": group["judge_accuracy"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "mean_total_tokens", "system"])


def _paired_comparisons(
    frame: pd.DataFrame,
    *,
    config: V2Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1)
    rows: list[dict[str, Any]] = []
    for dataset, dataset_frame in frame.groupby("dataset"):
        for system in sorted(set(dataset_frame["system"]) - {"direct"}):
            subset = dataset_frame[dataset_frame["system"].isin(["direct", system])]
            pivot = subset.pivot_table(
                index="case_id",
                columns="system",
                values="exact_correct",
                aggfunc="first",
            ).dropna()
            if "direct" in pivot and system in pivot:
                paired = pivot[system] - pivot["direct"]
            else:
                continue
            low, high = _bootstrap_mean(
                paired.to_numpy(float),
                rng=rng,
                replicates=config.bootstrap_replicates,
            )
            improved = int(((pivot["direct"] == 0) & (pivot[system] == 1)).sum())
            regressed = int(((pivot["direct"] == 1) & (pivot[system] == 0)).sum())
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
            discordance_rate = discordant / len(pivot) if len(pivot) else math.nan
            # Approximate two-sided 5% / 80%-power paired minimum detectable
            # difference using the observed discordance rate.
            mde = (
                math.sqrt((1.959964 + 0.841621) ** 2 * discordance_rate / len(pivot))
                if len(pivot)
                else math.nan
            )
            rows.append(
                {
                    "dataset": dataset,
                    "system": system,
                    "system_label": SYSTEM_LABELS[system],
                    "paired_n": len(pivot),
                    "accuracy_difference_vs_direct": paired.mean(),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "improved_cases": improved,
                    "regressed_cases": regressed,
                    "mcnemar_exact_p_value": p_value,
                    "discordance_rate": discordance_rate,
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
    eligible = frame[frame["initial_correct"].notna()]
    for (dataset, system), group in eligible.groupby(["dataset", "system"]):
        initial_wrong = group["initial_correct"] == 0
        initial_right = group["initial_correct"] == 1
        final_right = group["exact_correct"] == 1
        rejected = group["verifier_accept"] == False
        rows.append(
            {
                "dataset": dataset,
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "n": len(group),
                "initial_accuracy": group["initial_correct"].mean(),
                "final_accuracy": group["exact_correct"].mean(),
                "correction_rate_given_initial_wrong": (
                    (initial_wrong & final_right).sum() / initial_wrong.sum()
                    if initial_wrong.sum()
                    else math.nan
                ),
                "regression_rate_given_initial_right": (
                    (initial_right & ~final_right).sum() / initial_right.sum()
                    if initial_right.sum()
                    else math.nan
                ),
                "verifier_recall_on_initial_errors": (
                    (rejected & initial_wrong).sum() / initial_wrong.sum()
                    if initial_wrong.sum()
                    else math.nan
                ),
                "verifier_precision_when_rejecting": (
                    (rejected & initial_wrong).sum() / rejected.sum()
                    if rejected.sum()
                    else math.nan
                ),
                "escalation_rate": group["escalated"].mean(),
                "candidate_disagreement_rate": pd.to_numeric(
                    group["candidate_disagreement"], errors="coerce"
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def _pilot_gates(frame: pd.DataFrame, config: V2Config) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset"):
        direct = group[group["system"] == "direct"]
        decisions = group.pivot_table(
            index="case_id",
            columns="system",
            values="final_decision",
            aggfunc="first",
        )
        disagreement = (
            decisions.nunique(axis=1).gt(1).mean() if not decisions.empty else 0.0
        )
        validity = group["schema_valid"].mean()
        accuracy = direct["exact_correct"].mean()
        checks = {
            "schema_validity": {
                "value": validity,
                "threshold": f">={config.pilot_min_schema_validity}",
                "pass": validity >= config.pilot_min_schema_validity,
            },
            "direct_accuracy": {
                "value": accuracy,
                "threshold": (
                    f"{config.pilot_min_direct_accuracy}.."
                    f"{config.pilot_max_direct_accuracy}"
                ),
                "pass": (
                    config.pilot_min_direct_accuracy
                    <= accuracy
                    <= config.pilot_max_direct_accuracy
                ),
            },
            "system_disagreement": {
                "value": disagreement,
                "threshold": f">={config.pilot_min_system_disagreement}",
                "pass": disagreement >= config.pilot_min_system_disagreement,
            },
        }
        gates.append(
            {
                "dataset": dataset,
                "checks": checks,
                "pass": all(value["pass"] for value in checks.values()),
            }
        )
    return gates


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in summary.groupby("dataset"):
        for _, row in group.iterrows():
            dominated = bool(
                (
                    (group["accuracy"] >= row["accuracy"])
                    & (group["mean_total_tokens"] <= row["mean_total_tokens"])
                    & (
                        (group["accuracy"] > row["accuracy"])
                        | (group["mean_total_tokens"] < row["mean_total_tokens"])
                    )
                ).any()
            )
            rows.append(
                {
                    "dataset": dataset,
                    "system": row["system"],
                    "pareto_efficient": not dominated,
                }
            )
    return pd.DataFrame(rows)


def _budget_policy(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the empirical system selected at each observed token budget.

    These operating points are descriptive. They avoid interpolating a smooth
    threshold through a small number of noisy architecture means.
    """

    rows: list[dict[str, Any]] = []
    for dataset, group in summary.groupby("dataset"):
        group = group.dropna(subset=["mean_total_tokens", "accuracy"]).copy()
        previous: str | None = None
        for budget in sorted(group["mean_total_tokens"].unique()):
            feasible = group[group["mean_total_tokens"] <= budget].sort_values(
                ["accuracy", "mean_total_tokens", "system"],
                ascending=[False, True, True],
            )
            if feasible.empty:
                continue
            best = feasible.iloc[0]
            if best["system"] == previous:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "minimum_observed_budget_tokens": float(budget),
                    "selected_system": best["system"],
                    "system_label": best["system_label"],
                    "accuracy": float(best["accuracy"]),
                    "mean_total_tokens": float(best["mean_total_tokens"]),
                }
            )
            previous = str(best["system"])
    return pd.DataFrame(rows)


def _plots(summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    for dataset, group in summary.groupby("dataset"):
        figure, axis = plt.subplots(figsize=(7.0, 4.1), dpi=200)
        ordered = group.sort_values("mean_total_tokens")
        axis.scatter(
            ordered["mean_total_tokens"],
            ordered["accuracy"],
            s=58,
            color="#1F4E79",
            zorder=3,
        )
        for _, row in ordered.iterrows():
            axis.annotate(
                row["system_label"],
                (row["mean_total_tokens"], row["accuracy"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7.5,
            )
        axis.set_xlabel("Mean realized total tokens per case")
        axis.set_ylabel("Exact structured-decision accuracy")
        axis.set_ylim(0, 1)
        axis.set_title(f"Accuracy-cost frontier: {dataset}")
        figure.tight_layout()
        figure.savefig(output / f"accuracy_cost_{dataset}.png", bbox_inches="tight")
        plt.close(figure)


def analyze_v2(
    *,
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
) -> dict[str, Any]:
    ensure_run_manifest(repo, config, cases)
    generations = read_jsonl(v2_generation_path(repo, config), V2Generation)
    judgments = read_jsonl(v2_judgment_path(repo, config), V2Judgment)
    frame = build_v2_frame(
        cases=cases,
        generations=generations,
        judgments=judgments,
        config=config,
    )
    if frame.empty:
        raise RuntimeError("no successful Version 2 generations to analyze")
    expected_generation_keys = {
        (case.case_id, system, 0)
        for case in cases
        for system in config.systems
    }
    successful_generation_keys = [
        (row.case_id, row.system, row.replicate)
        for row in generations
        if row.status == "ok"
    ]
    if (
        len(successful_generation_keys) != len(set(successful_generation_keys))
        or set(successful_generation_keys) != expected_generation_keys
    ):
        raise RuntimeError(
            "generation grid must be complete and duplicate-free before analysis"
        )
    selected_for_judging = select_judging_generations(
        [row for row in generations if row.status == "ok"],
        config,
    )
    expected_judged_run_ids = {row.run_id for row in selected_for_judging}
    output = repo / "experiments" / "runs" / config.experiment_name / "analysis"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary = _system_summary(
        frame,
        config=config,
        expected_judged_run_ids=expected_judged_run_ids,
    )
    paired = _paired_comparisons(frame, config=config)
    mechanisms = _mechanisms(frame)
    pareto = _pareto(summary)
    budget_policy = _budget_policy(summary)
    task_summary = (
        frame.groupby(["dataset", "task", "system"], as_index=False)
        .agg(n=("case_id", "size"), accuracy=("exact_correct", "mean"))
        .sort_values(["dataset", "task", "system"])
    )
    frame.to_csv(tables / "case_level.csv", index=False)
    summary.to_csv(tables / "system_summary.csv", index=False)
    paired.to_csv(tables / "paired_comparisons.csv", index=False)
    mechanisms.to_csv(tables / "mechanisms.csv", index=False)
    pareto.to_csv(tables / "pareto.csv", index=False)
    budget_policy.to_csv(tables / "budget_policy.csv", index=False)
    task_summary.to_csv(tables / "task_breakdown.csv", index=False)
    _plots(summary, figures)
    successful_expected = len(cases) * len(config.systems)
    expected_judgment_keys = {
        (run_id, model)
        for run_id in expected_judged_run_ids
        for model in config.judge_models
    }
    successful_judgment_keys = {
        (judgment.run_id, judgment.judge_model)
        for judgment in judgments
        if judgment.status == "ok"
    }
    judge_successes = len(expected_judgment_keys & successful_judgment_keys)
    judge_expected = len(expected_judged_run_ids) * len(config.judge_models)
    pilot_gate_path = (
        repo
        / "experiments"
        / "runs"
        / config.pilot_experiment_name
        / "analysis"
        / "analysis_summary.json"
    )
    if config.experiment_name == config.pilot_experiment_name:
        eligible = set(frame["dataset"].unique())
    elif pilot_gate_path.exists():
        pilot_report = json.loads(pilot_gate_path.read_text())
        eligible = {
            gate["dataset"]
            for gate in pilot_report.get("pilot_gates", [])
            if gate.get("pass") is True
        }
    else:
        eligible = set()
    if paired.empty:
        adaptive = pd.DataFrame(
            columns=[
                "dataset",
                "system",
                "difference_ci_low",
                "mcnemar_holm_p_value",
                "pareto_efficient",
                "hypothesis_supported",
            ]
        )
    else:
        adaptive = paired[paired["system"] == "adaptive"].merge(
            pareto[pareto["system"] == "adaptive"],
            on=["dataset", "system"],
            how="left",
        )
        adaptive["pilot_gate_pass"] = adaptive["dataset"].isin(eligible)
        adaptive["practically_meaningful"] = (
            adaptive["accuracy_difference_vs_direct"]
            >= config.sesoi_accuracy_difference
        )
        adaptive["hypothesis_supported"] = (
            adaptive["pilot_gate_pass"]
            & adaptive["practically_meaningful"]
            & (adaptive["difference_ci_low"] > 0)
            & (adaptive["mcnemar_holm_p_value"] < 0.05)
            & adaptive["pareto_efficient"].fillna(False)
        )
    supported_datasets = adaptive.loc[
        adaptive["hypothesis_supported"], "dataset"
    ].tolist()
    report = {
        "experiment": config.experiment_name,
        "successful_generations": len(frame),
        "expected_generations": successful_expected,
        "generation_completion_rate": len(frame) / successful_expected,
        "unique_cases": frame["case_id"].nunique(),
        "datasets": sorted(frame["dataset"].unique().tolist()),
        "judge_successes": judge_successes,
        "judge_expected": judge_expected,
        "judge_coverage": judge_successes / judge_expected if judge_expected else 0.0,
        "judge_disagreement_rate_among_fully_covered": (
            frame.loc[
                frame["judge_count"] == len(config.judge_models),
                "judge_disagreement",
            ].mean()
            if (frame["judge_count"] == len(config.judge_models)).any()
            else None
        ),
        "pilot_gates": _pilot_gates(frame, config),
        "primary_hypothesis": (
            "Adaptive verify/escalate improves structured-decision accuracy over "
            "direct generation by at least the preregistered smallest effect of "
            "practical interest while remaining on the realized-token Pareto frontier."
        ),
        "sesoi_accuracy_difference": config.sesoi_accuracy_difference,
        "confirmatory_eligible_datasets": sorted(eligible),
        "primary_comparisons": adaptive.where(pd.notna(adaptive), None).to_dict(
            orient="records"
        ),
        "primary_supported_datasets": supported_datasets,
        "primary_replicated_across_datasets": (
            len(supported_datasets) == frame["dataset"].nunique()
        ),
        "budget_policy_rows": len(budget_policy),
    }
    report = _json_native(report)
    (output / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report
