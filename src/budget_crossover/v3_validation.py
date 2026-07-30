from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_jsonl
from .v3_config import V3Config
from .v3_dataset import case_set_profile, validate_hmda_source
from .v3_manifest import v3_run_dir
from .v3_models import V3Case, V3Generation
from .v3_runner import v3_error_path, v3_generation_path


def validate_v3_cases(
    *,
    repo: Path,
    config: V3Config,
    cases: list[V3Case],
) -> dict[str, Any]:
    source = repo / "data" / "raw" / config.hmda_raw_filename
    validate_hmda_source(source, config)
    issues: list[str] = []
    expected_cases = config.base_application_count * 2
    if len(cases) != expected_cases:
        issues.append(f"expected {expected_cases} cases, found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        issues.append("case IDs are not unique")
    pair_counts = Counter(case.pair_id for case in cases)
    if any(value != 2 for value in pair_counts.values()):
        issues.append("every source application must have exactly two counterfactual cases")

    for pair_id in pair_counts:
        pair = [case for case in cases if case.pair_id == pair_id]
        if len(pair) != 2:
            continue
        observed, counterfactual = sorted(
            pair, key=lambda case: case.counterfactual_variant, reverse=True
        )
        if {case.counterfactual_variant for case in pair} != {
            "observed",
            "counterfactual",
        }:
            issues.append(f"{pair_id} has invalid counterfactual variant labels")
        if observed.gold_decision != counterfactual.gold_decision:
            issues.append(f"{pair_id} changes the policy gold decision")
        for name in ("application", "collateral", "credit", "quality_control"):
            if observed.documents[name] != counterfactual.documents[name]:
                issues.append(f"{pair_id} changes non-monitoring document {name}")
        changed = [
            key
            for key in observed.protected_attributes
            if observed.protected_attributes[key]
            != counterfactual.protected_attributes[key]
        ]
        if changed != [observed.changed_protected_attribute]:
            issues.append(
                f"{pair_id} should change only {observed.changed_protected_attribute}; "
                f"changed {changed}"
            )

    for case in cases:
        financial = "\n".join(
            case.documents[name]
            for name in ("application", "collateral", "credit", "quality_control")
        ).lower()
        if case.historical_action.lower() in financial:
            issues.append(f"{case.case_id} leaks its historical action")
        for name, value in case.protected_attributes.items():
            if (
                name != "age_band"
                and value.lower() not in {"not reported", "joint"}
                and value.lower() in financial
            ):
                issues.append(f"{case.case_id} leaks protected field {name}")
    profile = case_set_profile(cases)
    return {
        "pass": not issues,
        "issues": sorted(set(issues)),
        "profile": profile,
        "source_sha256": config.hmda_source_sha256,
        "post_decision_fields_supplied_to_models": False,
        "historical_action_used_as_gold": False,
        "counterfactual_pairs_valid": not any(
            "counterfactual" in issue or "changes" in issue for issue in issues
        ),
    }


def _pilot_gate_path(repo: Path, config: V3Config) -> Path:
    return (
        repo
        / "experiments"
        / "runs"
        / config.pilot_experiment_name
        / "analysis"
        / "tables"
        / "system_summary.csv"
    )


def _pilot_validation_path(repo: Path, config: V3Config) -> Path:
    return (
        repo
        / "experiments"
        / "runs"
        / config.pilot_experiment_name
        / "validation.json"
    )


def pilot_gate_status(repo: Path, config: V3Config) -> dict[str, Any]:
    path = _pilot_gate_path(repo, config)
    if not path.exists():
        return {"pass": False, "issues": [f"pilot summary is missing: {path}"]}
    summary = pd.read_csv(path)
    high = summary[summary["token_budget"] == max(config.token_budgets)]
    issues: list[str] = []
    validation_path = _pilot_validation_path(repo, config)
    if not validation_path.exists():
        issues.append(f"pilot validation is missing: {validation_path}")
    else:
        validation = json.loads(validation_path.read_text())
        if not validation.get("pass"):
            issues.append("pilot validation did not pass")
        if (
            validation.get("requirements", {}).get("require_generations")
            is not True
        ):
            issues.append("pilot validation did not require the full generation grid")
    if set(high["system"]) != set(config.systems):
        issues.append("pilot high-budget system grid is incomplete")
    if (
        not high.empty
        and high["schema_validity"].min()
        < config.minimum_high_budget_schema_validity
    ):
        issues.append("pilot high-budget schema validity is below threshold")
    if (
        not summary.empty
        and summary["budget_overrun_rate"].max()
        > config.maximum_budget_overrun_rate
    ):
        issues.append("pilot token-budget overrun rate is above threshold")
    if (
        not summary.empty
        and summary.groupby("token_budget")["decision_accuracy"]
        .nunique()
        .max()
        < 2
    ):
        issues.append("pilot shows no accuracy disagreement across architectures")
    return {"pass": not issues, "issues": issues}


def assert_v3_pilot_gate(repo: Path, config: V3Config) -> None:
    status = pilot_gate_status(repo, config)
    if not status["pass"]:
        raise RuntimeError("v3 pilot gate failed: " + "; ".join(status["issues"]))


def validate_v3_run(
    *,
    repo: Path,
    config: V3Config,
    cases: list[V3Case],
    require_generations: bool = True,
    require_pilot_gate: bool = True,
) -> dict[str, Any]:
    case_validation = validate_v3_cases(repo=repo, config=config, cases=cases)
    issues = list(case_validation["issues"])
    generations = read_jsonl(v3_generation_path(repo, config), V3Generation)
    error_attempts = read_jsonl(v3_error_path(repo, config), V3Generation)
    expected = {
        (case.case_id, system, budget)
        for case in cases
        for system in config.systems
        for budget in config.token_budgets
    }
    observed = [
        (row.case_id, row.system, row.token_budget) for row in generations
    ]
    if require_generations:
        if set(observed) != expected:
            missing = len(expected - set(observed))
            extra = len(set(observed) - expected)
            issues.append(f"generation grid mismatch: {missing} missing, {extra} extra")
        if len(observed) != len(set(observed)):
            issues.append("generation grid contains duplicate cells")
        errors = sum(row.status == "error" for row in generations)
        if errors:
            issues.append(f"generation grid contains {errors} execution errors")
        invalid_statuses = sorted(
            {
                row.status
                for row in generations
                if row.status not in {"ok", "budget_exhausted"}
            }
        )
        if invalid_statuses:
            issues.append(f"generation grid contains invalid statuses: {invalid_statuses}")
        missing_usage_calls = 0
        inconsistent_usage_calls = 0
        accounting_mismatches = 0
        empty_successes = 0
        for row in generations:
            if row.status == "ok" and not row.calls:
                empty_successes += 1
            observed_total = 0
            complete_usage = True
            for call in row.calls:
                usage = call.response.usage
                if (
                    usage.prompt_tokens is None
                    or usage.completion_tokens is None
                    or usage.total_tokens is None
                ):
                    missing_usage_calls += 1
                    complete_usage = False
                    continue
                if usage.total_tokens < (
                    usage.prompt_tokens + usage.completion_tokens
                ):
                    inconsistent_usage_calls += 1
                observed_total += usage.total_tokens
            if complete_usage and row.calls:
                accounted = row.diagnostics.get("accounted_tokens")
                if accounted is None or int(accounted) != observed_total:
                    accounting_mismatches += 1
        if empty_successes:
            issues.append(f"{empty_successes} successful cells contain no API calls")
        if missing_usage_calls:
            issues.append(
                f"{missing_usage_calls} API calls are missing gateway usage fields"
            )
        if inconsistent_usage_calls:
            issues.append(
                f"{inconsistent_usage_calls} API calls report total tokens below "
                "prompt plus completion tokens"
            )
        if accounting_mismatches:
            issues.append(
                f"{accounting_mismatches} cells disagree with the authoritative "
                "gateway token totals"
            )
        overruns = sum(
            bool(row.diagnostics.get("budget_overrun", False))
            or (row.total_tokens is not None and row.total_tokens > row.token_budget)
            for row in generations
        )
        overrun_rate = overruns / len(generations) if generations else 1.0
        if overrun_rate > config.maximum_budget_overrun_rate:
            issues.append(
                f"budget overrun rate {overrun_rate:.3f} exceeds "
                f"{config.maximum_budget_overrun_rate:.3f}"
            )
    else:
        overrun_rate = None
        missing_usage_calls = None
        inconsistent_usage_calls = None
        accounting_mismatches = None
        empty_successes = None
    pilot_status = {"pass": True, "issues": []}
    if require_pilot_gate and config.experiment_name != config.pilot_experiment_name:
        pilot_status = pilot_gate_status(repo, config)
        if not pilot_status["pass"]:
            issues.extend(f"pilot: {issue}" for issue in pilot_status["issues"])
    result = {
        "pass": not issues,
        "issues": sorted(set(issues)),
        "requirements": {
            "require_generations": require_generations,
            "require_pilot_gate": require_pilot_gate,
        },
        "case_validation": case_validation,
        "generation_cells_expected": len(expected),
        "generation_cells_observed": len(observed),
        "retryable_error_attempts_observed": len(error_attempts),
        "missing_usage_calls": missing_usage_calls,
        "inconsistent_usage_calls": inconsistent_usage_calls,
        "token_accounting_mismatches": accounting_mismatches,
        "successful_cells_without_calls": empty_successes,
        "budget_overrun_rate": overrun_rate,
        "pilot_gate": pilot_status,
    }
    output = v3_run_dir(repo, config) / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
