from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .v2_config import V2Config
from .v2_judging import select_judging_generations, v2_judgment_path
from .v2_manifest import ensure_run_manifest, run_manifest_path
from .v2_models import V2Case, V2Generation, V2Judgment
from .v2_runner import v2_generation_path

EXPECTED_CALL_COUNTS: dict[str, set[int]] = {
    "direct": {1},
    "checklist": {1},
    "strong_direct": {1},
    "self_critique": {3},
    "external_verify": {3},
    "best_of_2": {3},
    "best_of_4": {5},
    "adaptive": {2, 5},
}
TRUNCATION_REASONS = {"length", "max_tokens", "max_output_tokens"}


def _check(value: bool, *, observed: Any, expected: Any) -> dict[str, Any]:
    return {"pass": bool(value), "observed": observed, "expected": expected}


def _pilot_gate_report(repo: Path, config: V2Config) -> dict[str, Any] | None:
    path = (
        repo
        / "experiments"
        / "runs"
        / config.pilot_experiment_name
        / "analysis"
        / "analysis_summary.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text())


def assert_pilot_gate(repo: Path, config: V2Config) -> None:
    report = _pilot_gate_report(repo, config)
    if report is None:
        raise RuntimeError(
            "pilot analysis is missing; run pilot-v2 and analyze-v2 with the "
            "pilot config before the main sweep"
        )
    if report.get("successful_generations") != report.get("expected_generations"):
        raise RuntimeError(
            "pilot generation grid is incomplete; resume pilot-v2 before the main sweep"
        )
    passed = [
        gate.get("dataset")
        for gate in report.get("pilot_gates", [])
        if gate.get("pass") is True
    ]
    if not passed:
        raise RuntimeError(
            "no dataset passed the preregistered pilot gates; inspect the pilot "
            "analysis instead of spending the main-study budget"
        )


def validate_v2_run(
    *,
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
    require_judgments: bool,
    require_pilot_gate: bool,
) -> dict[str, Any]:
    ensure_run_manifest(repo, config, cases)
    generations = read_jsonl(v2_generation_path(repo, config), V2Generation)
    successes = [row for row in generations if row.status == "ok"]
    expected_keys = {
        (case.case_id, system, 0)
        for case in cases
        for system in config.systems
    }
    success_keys = [
        (row.case_id, row.system, row.replicate)
        for row in successes
    ]
    success_counts = Counter(success_keys)
    duplicate_successes = sorted(
        key for key, count in success_counts.items() if count > 1
    )
    missing = sorted(expected_keys - set(success_keys))
    unexpected = sorted(set(success_keys) - expected_keys)

    missing_usage: list[str] = []
    inconsistent_usage: list[str] = []
    unexpected_calls: list[dict[str, Any]] = []
    truncated: list[dict[str, Any]] = []
    missing_wall_time: list[str] = []
    invalid_schema: list[str] = []
    for generation in successes:
        if generation.wall_time_seconds is None or generation.wall_time_seconds <= 0:
            missing_wall_time.append(generation.run_id)
        if generation.parsed_decision is None:
            invalid_schema.append(generation.run_id)
        allowed = EXPECTED_CALL_COUNTS.get(generation.system, set())
        if len(generation.calls) not in allowed:
            unexpected_calls.append(
                {
                    "run_id": generation.run_id,
                    "system": generation.system,
                    "observed": len(generation.calls),
                    "expected": sorted(allowed),
                }
            )
        for call_index, call in enumerate(generation.calls):
            usage = call.response.usage
            call_key = f"{generation.run_id}:call-{call_index + 1}"
            if (
                usage.prompt_tokens is None
                or usage.completion_tokens is None
                or usage.total_tokens is None
            ):
                missing_usage.append(call_key)
            elif usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
                inconsistent_usage.append(call_key)
            reason = str(call.response.raw_finish_reason or "").lower()
            if reason in TRUNCATION_REASONS:
                truncated.append(
                    {
                        "call": call_key,
                        "finish_reason": call.response.raw_finish_reason,
                        "completion_tokens": usage.completion_tokens,
                        "token_cap": call.token_cap,
                    }
                )

    selected = select_judging_generations(successes, config)
    expected_judgment_keys = {
        (generation.run_id, model)
        for generation in selected
        for model in config.judge_models
    }
    judgments = read_jsonl(v2_judgment_path(repo, config), V2Judgment)
    successful_judgments = [row for row in judgments if row.status == "ok"]
    judgment_keys = [(row.run_id, row.judge_model) for row in successful_judgments]
    judgment_counts = Counter(judgment_keys)
    duplicate_judgments = sorted(
        key for key, count in judgment_counts.items() if count > 1
    )
    missing_judgments = sorted(expected_judgment_keys - set(judgment_keys))
    unexpected_judgments = sorted(set(judgment_keys) - expected_judgment_keys)

    checks = {
        "run_manifest_matches": _check(
            run_manifest_path(repo, config).exists(),
            observed=str(run_manifest_path(repo, config)),
            expected="matching immutable run manifest",
        ),
        "generation_grid_complete": _check(
            not missing and not unexpected and not duplicate_successes,
            observed={
                "successful": len(successes),
                "missing": len(missing),
                "unexpected": len(unexpected),
                "duplicates": len(duplicate_successes),
            },
            expected={"successful": len(expected_keys), "missing": 0, "unexpected": 0, "duplicates": 0},
        ),
        "gateway_usage_complete": _check(
            not missing_usage and not inconsistent_usage,
            observed={
                "missing": len(missing_usage),
                "inconsistent_totals": len(inconsistent_usage),
            },
            expected={"missing": 0, "inconsistent_totals": 0},
        ),
        "call_contracts": _check(
            not unexpected_calls,
            observed=len(unexpected_calls),
            expected=0,
        ),
        "truncation_rate_within_limit": _check(
            len(truncated) / len(successes) <= config.maximum_truncation_rate
            if successes
            else False,
            observed=len(truncated) / len(successes) if successes else None,
            expected=f"<={config.maximum_truncation_rate}",
        ),
        "wall_time_complete": _check(
            not missing_wall_time,
            observed=len(missing_wall_time),
            expected=0,
        ),
        "schema_validity_above_floor": _check(
            1 - len(invalid_schema) / len(successes)
            >= config.minimum_schema_validity
            if successes
            else False,
            observed=(
                1 - len(invalid_schema) / len(successes) if successes else None
            ),
            expected=f">={config.minimum_schema_validity}",
        ),
        "judge_sample_complete": _check(
            (
                not require_judgments
                or (
                    not missing_judgments
                    and not unexpected_judgments
                    and not duplicate_judgments
                )
            ),
            observed={
                "successful": len(successful_judgments),
                "missing": len(missing_judgments),
                "unexpected": len(unexpected_judgments),
                "duplicates": len(duplicate_judgments),
            },
            expected={
                "successful": len(expected_judgment_keys),
                "missing": 0,
                "unexpected": 0,
                "duplicates": 0,
            },
        ),
    }

    pilot_report = _pilot_gate_report(repo, config) if require_pilot_gate else None
    if require_pilot_gate:
        pilot_complete = bool(
            pilot_report
            and pilot_report.get("successful_generations")
            == pilot_report.get("expected_generations")
        )
        passed_datasets = (
            [
                row.get("dataset")
                for row in pilot_report.get("pilot_gates", [])
                if row.get("pass") is True
            ]
            if pilot_report
            else []
        )
        checks["pilot_gate"] = _check(
            pilot_complete and bool(passed_datasets),
            observed={
                "report_present": pilot_report is not None,
                "complete": pilot_complete,
                "passed_datasets": passed_datasets,
            },
            expected="complete pilot and at least one passing dataset",
        )
    judge_manifest = (
        run_manifest_path(repo, config).parent / "judge_sample_manifest.json"
    )
    if require_judgments:
        checks["judge_sample_frozen"] = _check(
            judge_manifest.exists(),
            observed=str(judge_manifest) if judge_manifest.exists() else None,
            expected="frozen judge_sample_manifest.json",
        )

    report = {
        "experiment": config.experiment_name,
        "pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
        "details": {
            "missing_generations": missing,
            "unexpected_generations": unexpected,
            "duplicate_successes": duplicate_successes,
            "missing_usage": missing_usage,
            "inconsistent_usage": inconsistent_usage,
            "unexpected_call_counts": unexpected_calls,
            "truncated_calls": truncated,
            "missing_wall_time": missing_wall_time,
            "invalid_schema": invalid_schema,
            "missing_judgments": missing_judgments,
            "unexpected_judgments": unexpected_judgments,
            "duplicate_judgments": duplicate_judgments,
        },
        "expected_output_contract": [
            str(v2_generation_path(repo, config)),
            str(v2_judgment_path(repo, config)),
            str(
                repo
                / "experiments"
                / "runs"
                / config.experiment_name
                / "analysis"
                / "analysis_summary.json"
            ),
            str(
                repo
                / "experiments"
                / "runs"
                / config.experiment_name
                / "analysis"
                / "tables"
                / "system_summary.csv"
            ),
        ],
    }
    output = repo / "experiments" / "runs" / config.experiment_name / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
