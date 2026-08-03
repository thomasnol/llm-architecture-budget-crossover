from __future__ import annotations

"""Fail-closed linear workflow for the preregistered conditional crossover study."""

import asyncio
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

from .analysis import (
    PairedCaseOutcome,
    SystemCaseMetric,
    cluster_bootstrap_crossover,
    confirm_crossover,
    pareto_dominance_probabilities,
    score_itt_results,
)
from .calibration import (
    DevelopmentFitObservation,
    freeze_calibration,
    select_calibration_ceiling,
)
from .config import ExperimentConfig, SourceSnapshotConfig
from .dataset import (
    DatasetSnapshot,
    SplitQuotas,
    prepare_primary_datasets,
)
from .diagnostics import (
    FinanceComplexSnapshot,
    adapt_financecomplex_snapshot,
    audit_evidence_lineage_and_leakage,
    build_financecomplex_boundary_report,
    export_oracle_evidence_cases,
    scorer_oracle_boundary,
)
from .gateway import GatewayClient, GatewayCompletionClient, GatewayPromptTokenizer
from .io import read_jsonl, write_jsonl
from .manifest import (
    ensure_run_manifest,
    manifest_path,
    run_dir,
    sha256_path,
    update_run_state,
    verify_run_manifest,
)
from .models import (
    AnswerSpec,
    CellResult,
    DescriptiveMetadata,
    EvidenceItem,
    GatewayResponse,
    HiddenLabel,
    PublicCase,
    Usage,
)
from .preflight import run_preflight
from .runner import CellKey, build_cell_grid, execute_cells
from .validation import OperationalGateInputs, evaluate_operational_gates

STAGES = (
    "prepare",
    "diagnose-finance-complex",
    "preflight",
    "develop",
    "pilot",
    "gate",
    "run",
    "validate",
    "analyze",
    "build-paper",
)


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _prepared_dir(repo: Path, config: ExperimentConfig) -> Path:
    return _resolve(repo, config.prepared_data_dir)


def _receipt_path(repo: Path, config: ExperimentConfig, stage: str) -> Path:
    return run_dir(repo, config) / "stages" / f"{stage}.json"


def _config_sha256(config: ExperimentConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"invalid {description}: expected a JSON object")
    return payload


def _write_receipt(
    repo: Path,
    config: ExperimentConfig,
    stage: str,
    *,
    outputs: Mapping[str, Path],
    upstream_stages: Sequence[str] = (),
    passed: bool = True,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    upstream = {
        name: sha256_path(_receipt_path(repo, config, name)) for name in upstream_stages
    }
    receipt = {
        "schema_version": 1,
        "stage": stage,
        "created_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "resolved_config_sha256": _config_sha256(config),
        "non_empirical_fixture": config.execution_mode == "offline_fixture",
        "upstream_receipt_hashes": upstream,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in sorted(outputs.items())
        },
        "details": dict(details or {}),
    }
    _write_json(_receipt_path(repo, config, stage), receipt)
    return receipt


def _verify_receipt(
    repo: Path,
    config: ExperimentConfig,
    stage: str,
    *,
    require_pass: bool = True,
    _trail: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if stage in _trail:
        raise RuntimeError(f"stage receipt dependency cycle: {stage}")
    path = _receipt_path(repo, config, stage)
    receipt = _read_json(path, f"{stage} stage receipt")
    if receipt.get("stage") != stage:
        raise RuntimeError(f"{stage} receipt has the wrong stage identity")
    if receipt.get("resolved_config_sha256") != _config_sha256(config):
        raise RuntimeError(f"{stage} receipt resolved configuration hash mismatch")
    if receipt.get("non_empirical_fixture") is not (
        config.execution_mode == "offline_fixture"
    ):
        raise RuntimeError(f"{stage} receipt execution-mode mismatch")
    if require_pass and receipt.get("passed") is not True:
        raise RuntimeError(f"upstream stage {stage} did not pass")
    trail = _trail | {stage}
    for upstream, expected in receipt.get("upstream_receipt_hashes", {}).items():
        if sha256_path(_receipt_path(repo, config, upstream)) != expected:
            raise RuntimeError(f"{stage} upstream receipt hash mismatch: {upstream}")
        _verify_receipt(repo, config, upstream, _trail=trail)
    for name, artifact in receipt.get("outputs", {}).items():
        artifact_path = Path(artifact["path"])
        if sha256_path(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"{stage} output hash mismatch: {name}")
    return receipt


def _prepared_paths(repo: Path, config: ExperimentConfig) -> dict[str, Path]:
    root = _prepared_dir(repo, config)
    paths: dict[str, Path] = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                paths[path.relative_to(root).as_posix()] = path
    return paths


def _split_paths(
    repo: Path, config: ExperimentConfig, split: str
) -> tuple[Path, Path]:
    root = _prepared_dir(repo, config)
    return root / "public" / f"{split}.jsonl", root / "hidden" / f"{split}.jsonl"


def _load_split(
    repo: Path, config: ExperimentConfig, split: str
) -> tuple[tuple[PublicCase, ...], tuple[HiddenLabel, ...]]:
    public_path, hidden_path = _split_paths(repo, config, split)
    cases = tuple(read_jsonl(public_path, PublicCase))
    labels = tuple(read_jsonl(hidden_path, HiddenLabel))
    if not cases or len(cases) != len(labels):
        raise RuntimeError(f"prepared {split} split is missing or does not join one-to-one")
    if {case.case_id for case in cases} != {label.case_id for label in labels}:
        raise RuntimeError(f"prepared {split} public and hidden case IDs do not match")
    return cases, labels


def _offline_case(split: str, index: int) -> tuple[PublicCase, HiddenLabel]:
    dataset = "finqa" if index % 2 == 0 else "tatqa"
    case_id = f"offline-{split}-{index:03d}"
    document_id = f"offline-document-{split}-{index:03d}"
    evidence_id = f"{case_id}-evidence"
    public = PublicCase(
        case_id=case_id,
        dataset=dataset,
        document_id=document_id,
        question="What value is reported in the fixture evidence?",
        evidence=(
            EvidenceItem(
                evidence_id=evidence_id,
                document_id=document_id,
                kind="text",
                text="The fixture value is 10.",
                ordinal=0,
            ),
        ),
        stratum="easy_control" if split == "easy_reserve" else "headroom",
        metadata=DescriptiveMetadata(tags=("NON_EMPIRICAL_OFFLINE_FIXTURE",)),
    )
    hidden = HiddenLabel(
        case_id=case_id,
        answer=AnswerSpec(
            value=Decimal(10),
            unit=None,
            entity=None,
            period=None,
            absolute_tolerance=Decimal("0.000001"),
            relative_tolerance=Decimal("0.000001"),
        ),
        gold_derivation="10",
        gold_support_ids=(evidence_id,),
        source_lineage=("NON_EMPIRICAL_OFFLINE_FIXTURE", document_id, case_id),
    )
    return public, hidden


def prepare_stage(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    if _receipt_path(repo, config, "prepare").is_file():
        return _verify_receipt(repo, config, "prepare")
    output_dir = _prepared_dir(repo, config)
    if config.execution_mode == "offline_fixture":
        counts = {
            "development": config.development_cases,
            "operational_pilot": config.operational_pilot_cases,
            "main": config.main_cases,
            "easy_reserve": config.easy_reserve_cases,
        }
        lineage: list[dict[str, Any]] = []
        for split, count in counts.items():
            pairs = tuple(_offline_case(split, index) for index in range(count))
            public_path, hidden_path = _split_paths(repo, config, split)
            write_jsonl(public_path, (public for public, _hidden in pairs))
            write_jsonl(hidden_path, (hidden for _public, hidden in pairs))
            lineage.extend(
                {
                    "case_id": public.case_id,
                    "split": split,
                    "dataset": public.dataset,
                    "document_id": public.document_id,
                    "stratum": public.stratum,
                }
                for public, _hidden in pairs
            )
        _write_json(
            output_dir / "profile.json",
            {
                "schema_version": 1,
                "status": "NON_EMPIRICAL_OFFLINE_FIXTURE",
                "counts": counts,
                "document_disjoint": True,
                "lineage": lineage,
            },
        )
        (output_dir / "rejections.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "rejections.jsonl").write_text("", encoding="utf-8")
    else:
        prepare_primary_datasets(
            (
                DatasetSnapshot(
                    dataset="finqa",
                    path=_resolve(repo, config.finqa_snapshot.path),
                    expected_sha256=config.finqa_snapshot.sha256,
                ),
                DatasetSnapshot(
                    dataset="tatqa",
                    path=_resolve(repo, config.tatqa_snapshot.path),
                    expected_sha256=config.tatqa_snapshot.sha256,
                ),
            ),
            output_dir=output_dir,
            quotas=SplitQuotas(
                development=config.development_cases,
                operational_pilot=config.operational_pilot_cases,
                main=config.main_cases,
                easy_reserve=config.easy_reserve_cases,
            ),
            seed=config.seed,
        )
    paths = {
        name: path
        for name, path in _prepared_paths(repo, config).items()
        if name != "workflow_hashes.json"
    }
    hashes = {name: sha256_path(path) for name, path in paths.items()}
    hash_path = output_dir / "workflow_hashes.json"
    _write_json(
        hash_path,
        {
            "source_hashes": {
                "finqa": config.finqa_snapshot.sha256,
                "tatqa": config.tatqa_snapshot.sha256,
            },
            "artifact_hashes": hashes,
            "non_empirical_fixture": config.execution_mode == "offline_fixture",
        },
    )
    return _write_receipt(
        repo,
        config,
        "prepare",
        outputs={**_prepared_paths(repo, config)},
        details={"split_counts": {
            split: len(_load_split(repo, config, split)[0])
            for split in ("development", "operational_pilot", "main", "easy_reserve")
        }},
    )


def diagnose_finance_complex_stage(
    *, repo: Path, config: ExperimentConfig
) -> dict[str, Any]:
    if _receipt_path(repo, config, "diagnose-finance-complex").is_file():
        return _verify_receipt(repo, config, "diagnose-finance-complex")
    _verify_receipt(repo, config, "prepare")
    output_dir = run_dir(repo, config) / "finance_complex_diagnostic"
    report_path = output_dir / "boundary_report.json"
    if config.execution_mode == "offline_fixture":
        report = {
            "schema_version": 1,
            "domain_role": "exploratory_only",
            "confirmation_pool_eligible": False,
            "exploratory_system_run_gate": False,
            "status": "NON_EMPIRICAL_OFFLINE_FIXTURE",
            "failures": ["model_with_oracle_evidence", "retrieval", "orchestration"],
        }
        _write_json(report_path, report)
    else:
        if config.finance_complex_snapshot is None:
            raise RuntimeError("FinanceComplexQA snapshot is required for diagnostics")
        adapted = adapt_financecomplex_snapshot(
            FinanceComplexSnapshot(
                path=_resolve(repo, config.finance_complex_snapshot.path),
                expected_sha256=config.finance_complex_snapshot.sha256,
            ),
            output_dir=output_dir,
            expected_count=config.expected_finance_complex_cases,
        )
        scorer = scorer_oracle_boundary(adapted.cases)
        lineage = audit_evidence_lineage_and_leakage(adapted.cases)
        export_oracle_evidence_cases(adapted.cases, output_dir / "oracle_evidence.jsonl")
        build_financecomplex_boundary_report(
            scorer=scorer,
            lineage_leakage=lineage,
            oracle_evidence_model={"pass": False, "status": "pending_model_run"},
            retrieval={},
            orchestration={"pass": False, "status": "pending_model_run"},
            output_path=report_path,
        )
    return _write_receipt(
        repo,
        config,
        "diagnose-finance-complex",
        outputs={"boundary_report": report_path},
        upstream_stages=("prepare",),
        details={"domain_role": "exploratory_only"},
    )


def preflight_stage(
    *,
    repo: Path,
    config: ExperimentConfig,
    client: Any | None = None,
) -> dict[str, Any]:
    path = run_dir(repo, config) / "preflight.json"
    if path.is_file() and _receipt_path(repo, config, "preflight").is_file():
        existing = _read_json(path, "preflight")
        if existing.get("pass") is True:
            _verify_receipt(repo, config, "preflight")
            expected_identity = {
                "experiment": config.experiment_name,
                "requested_model": config.model,
                "tokenizer_id": config.tokenizer_id,
                "tokenizer_sha256": config.tokenizer_sha256,
                "non_empirical_fixture": config.execution_mode == "offline_fixture",
            }
            if any(existing.get(key) != value for key, value in expected_identity.items()):
                raise RuntimeError("frozen preflight identity does not match configuration")
            return existing
    _verify_receipt(repo, config, "prepare")
    _verify_receipt(repo, config, "diagnose-finance-complex")
    report = asyncio.run(run_preflight(repo=repo, config=config, client=client))
    _write_receipt(
        repo,
        config,
        "preflight",
        outputs={"preflight": path},
        upstream_stages=("prepare", "diagnose-finance-complex"),
        passed=bool(report["pass"]),
    )
    return report


def _development_observations(
    repo: Path, config: ExperimentConfig, tier: str, case_ids: Sequence[str]
) -> tuple[DevelopmentFitObservation, ...]:
    if config.execution_mode == "offline_fixture":
        mandatory = {"low": 4000, "middle": 8000, "high": 16000}[tier]
        return tuple(
            DevelopmentFitObservation(case_id=case_id, mandatory_tokens=mandatory)
            for case_id in case_ids
        )
    path = _resolve(repo, config.development_fit_path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("tier") == tier]
    if {row.get("case_id") for row in selected} != set(case_ids):
        raise RuntimeError(f"development fit observations do not cover {tier} cases exactly")
    forbidden = {"correct", "accuracy", "answer", "system_difference", "outcome"}
    if any(forbidden & set(row) for row in selected):
        raise RuntimeError("development calibration contains forbidden outcome fields")
    return tuple(
        DevelopmentFitObservation(
            case_id=str(row["case_id"]), mandatory_tokens=int(row["mandatory_tokens"])
        )
        for row in selected
    )


def develop_stage(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    _verify_receipt(repo, config, "prepare")
    _verify_receipt(repo, config, "preflight")
    cases, _labels = _load_split(repo, config, "development")
    selections = tuple(
        select_calibration_ceiling(
            tier,
            _development_observations(
                repo, config, tier, tuple(case.case_id for case in cases)
            ),
        )
        for tier in config.tiers
    )
    output = run_dir(repo, config) / "calibration.json"
    if output.exists():
        existing = _read_json(output, "calibration")
        expected = {
            "schema_version": 1,
            "frozen_before_pilot": True,
            "outcome_fields_available": False,
            "selections": [selection.model_dump(mode="json") for selection in selections],
        }
        if existing != expected:
            raise RuntimeError("frozen calibration mismatch")
    else:
        freeze_calibration(selections, output_path=output, pilot_started=False)
    return _write_receipt(
        repo,
        config,
        "develop",
        outputs={"calibration": output},
        upstream_stages=("prepare", "preflight"),
        details={"outcome_fields_available": False},
    )


def _gateway_adapter(config: ExperimentConfig) -> GatewayCompletionClient:
    gateway = GatewayClient(timeout_seconds=config.request_timeout_seconds)
    tokenizer = GatewayPromptTokenizer(
        gateway,
        tokenizer_id=config.tokenizer_id,
        tokenizer_sha256=config.tokenizer_sha256,
    )
    return GatewayCompletionClient(
        gateway=gateway,
        tokenizer=tokenizer,
        model=config.model,
        tokenizer_id=config.tokenizer_id,
        tokenizer_sha256=config.tokenizer_sha256,
    )


async def _execute(
    *,
    config: ExperimentConfig,
    cases: Sequence[PublicCase],
    client: Any,
    results_path: Path,
    attempts_path: Path,
) -> Any:
    try:
        return await execute_cells(
            cases=cases,
            systems=config.systems,
            tiers=config.tiers,
            repetitions=config.repetitions,
            model=config.model,
            client=client,
            results_path=results_path,
            attempts_path=attempts_path,
            max_concurrency=config.max_concurrency,
        )
    finally:
        gateway = getattr(client, "gateway", None)
        if isinstance(gateway, GatewayClient):
            await gateway.close()


def pilot_stage(
    *, repo: Path, config: ExperimentConfig, client: Any | None = None
) -> dict[str, Any]:
    _verify_receipt(repo, config, "develop")
    cases, _labels = _load_split(repo, config, "operational_pilot")
    results_path = run_dir(repo, config) / "pilot_results.jsonl"
    attempts_path = run_dir(repo, config) / "pilot_attempts.jsonl"
    active = client or _gateway_adapter(config)
    summary = asyncio.run(
        _execute(
            config=config,
            cases=cases,
            client=active,
            results_path=results_path,
            attempts_path=attempts_path,
        )
    )
    if not attempts_path.exists():
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        attempts_path.write_text("", encoding="utf-8")
    receipt = _write_receipt(
        repo,
        config,
        "pilot",
        outputs={"results": results_path, "attempts": attempts_path},
        upstream_stages=("prepare", "preflight", "develop"),
        passed=summary.remaining == 0,
        details=summary.model_dump(mode="json"),
    )
    return receipt


def _key(result: CellResult) -> CellKey:
    return CellKey(
        case_id=result.case_id,
        system=result.system,
        tier=result.tier,
        repetition=result.repetition,
    )


def _authoritative(result: CellResult) -> bool:
    return bool(result.trace.call_events) and all(
        event.usage is not None
        and event.usage.prompt_tokens is not None
        and event.usage.completion_tokens is not None
        and not event.protocol_violation
        for event in result.trace.call_events
    )


def gate_stage(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    _verify_receipt(repo, config, "pilot")
    cases, _labels = _load_split(repo, config, "operational_pilot")
    results = tuple(read_jsonl(run_dir(repo, config) / "pilot_results.jsonl", CellResult))
    expected = build_cell_grid(
        cases=cases,
        systems=config.systems,
        tiers=config.tiers,
        repetitions=config.repetitions,
    )
    observed = tuple(_key(result) for result in results)
    expected_mechanisms = dict(
        sorted(Counter(f"{key.system}:{key.tier}" for key in expected).items())
    )
    observed_mechanisms = dict(
        sorted(Counter(f"{key.system}:{key.tier}" for key in observed).items())
    )
    if config.execution_mode == "offline_fixture":
        metrics = {
            "matched_blocks_total": 0,
            "unresolved_external_matched_blocks": 0,
            "verified_search_median_tokens": {"low": 100.0, "middle": 120.0, "high": 144.0},
            "easy_monolith_correct": 9,
            "easy_monolith_total": 10,
            "hard_monolith_correct": 5,
            "hard_monolith_total": 10,
            "checker_true_negatives": 95,
            "checker_actual_negatives": 100,
            "checker_true_positives": 60,
            "checker_actual_positives": 100,
            "correct_first_drafts_repaired": 1,
            "correct_to_wrong_repairs": 0,
            "checker_detected_wrong_first_drafts": 10,
            "wrong_first_drafts_corrected": 2,
        }
    else:
        metrics = _read_json(
            _resolve(repo, config.pilot_gate_metrics_path), "blinded pilot metrics"
        )
        if metrics.pop("provenance", None) != "independent_blinded_pilot_audit_v1":
            raise RuntimeError("pilot gate metrics lack independent blinded provenance")
    low_results = [result for result in results if result.tier == "low"]
    inputs = OperationalGateInputs(
        expected_cell_keys=expected,
        observed_cell_keys=observed,
        authoritative_usage_cells=sum(_authoritative(result) for result in results),
        label_leakage_count=0,
        budget_overrun_count=sum(
            event.protocol_violation
            for result in results
            for event in result.trace.call_events
        ),
        schema_valid_cells=sum(
            result.status == "ok" and result.candidate is not None for result in results
        ),
        expected_mechanism_counts=expected_mechanisms,
        observed_mechanism_counts=observed_mechanisms,
        low_tier_cases=len(low_results),
        low_tier_feasible_cases=sum(
            result.trace.exit_reason != "budget_exhausted" for result in low_results
        ),
        **metrics,
    )
    output = run_dir(repo, config) / "pilot_gate.json"
    artifact = evaluate_operational_gates(inputs, output_path=output)
    _write_receipt(
        repo,
        config,
        "gate",
        outputs={"pilot_gate": output},
        upstream_stages=("pilot",),
        passed=artifact.passed,
        details={"failed_components": list(artifact.failed_components)},
    )
    return artifact.model_dump(mode="json")


def _run_manifest_inputs(
    repo: Path, config: ExperimentConfig
) -> tuple[tuple[PublicCase, ...], dict[str, Path], Path, Path]:
    cases, _labels = _load_split(repo, config, "main")
    return (
        cases,
        _prepared_paths(repo, config),
        run_dir(repo, config) / "preflight.json",
        run_dir(repo, config) / "pilot_gate.json",
    )


def _verify_prepared_hashes_before_parse(repo: Path, config: ExperimentConfig) -> None:
    """Check serialized artifacts before parsing any possibly tampered row."""
    frozen = _read_json(manifest_path(repo, config), "run manifest")
    expected = frozen.get("identity", {}).get("artifact_hashes", {})
    observed_paths = _prepared_paths(repo, config)
    observed = {name: sha256_path(path) for name, path in observed_paths.items()}
    if expected != observed:
        raise RuntimeError("run manifest mismatch; refusing changed inputs: identity.artifact_hashes")


def run_stage(
    *, repo: Path, config: ExperimentConfig, client: Any | None = None
) -> dict[str, Any]:
    _verify_receipt(repo, config, "gate")
    cases, artifacts, preflight, pilot_gate = _run_manifest_inputs(repo, config)
    ensure_run_manifest(
        repo=repo,
        config=config,
        cases=cases,
        artifact_paths=artifacts,
        preflight_path=preflight,
        pilot_gate_path=pilot_gate,
    )
    results_path = run_dir(repo, config) / "results.jsonl"
    attempts_path = run_dir(repo, config) / "attempts.jsonl"
    active = client or _gateway_adapter(config)
    summary = asyncio.run(
        _execute(
            config=config,
            cases=cases,
            client=active,
            results_path=results_path,
            attempts_path=attempts_path,
        )
    )
    if not attempts_path.exists():
        attempts_path.write_text("", encoding="utf-8")
    update_run_state(
        repo=repo,
        config=config,
        stage="run",
        counters=summary.model_dump(mode="json"),
    )
    receipt = _write_receipt(
        repo,
        config,
        "run",
        outputs={"results": results_path, "attempts": attempts_path},
        upstream_stages=("gate",),
        passed=summary.remaining == 0,
        details=summary.model_dump(mode="json"),
    )
    return receipt


def validate_stage(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    _verify_prepared_hashes_before_parse(repo, config)
    cases, artifacts, preflight, pilot_gate = _run_manifest_inputs(repo, config)
    verify_run_manifest(
        repo=repo,
        config=config,
        cases=cases,
        artifact_paths=artifacts,
        preflight_path=preflight,
        pilot_gate_path=pilot_gate,
    )
    results_path = run_dir(repo, config) / "results.jsonl"
    results = tuple(read_jsonl(results_path, CellResult))
    expected = build_cell_grid(
        cases=cases,
        systems=config.systems,
        tiers=config.tiers,
        repetitions=config.repetitions,
    )
    expected_keys = {(key.case_id, key.system, key.tier, key.repetition) for key in expected}
    observed = [(row.case_id, row.system, row.tier, row.repetition) for row in results]
    observed_keys = set(observed)
    complete = observed_keys == expected_keys
    unique = len(observed) == len(observed_keys)
    authoritative = all(_authoritative(result) for result in results) and bool(results)
    protocol_violations = sum(
        event.protocol_violation
        for result in results
        for event in result.trace.call_events
    )
    validation = {
        "schema_version": 1,
        "pass": complete and unique and authoritative and protocol_violations == 0,
        "complete_grid": complete,
        "unique_grid": unique,
        "authoritative_usage": authoritative,
        "protocol_violation_count": protocol_violations,
        "expected_cells": len(expected),
        "observed_cells": len(results),
        "unique_cells": len(observed_keys),
        "missing_cells": len(expected_keys - observed_keys),
        "unexpected_cells": len(observed_keys - expected_keys),
        "source_manifest_sha256": sha256_path(manifest_path(repo, config)),
        "pilot_gate_sha256": sha256_path(pilot_gate),
        "non_empirical_fixture": config.execution_mode == "offline_fixture",
    }
    output = run_dir(repo, config) / "validation.json"
    _write_json(output, validation)
    _write_receipt(
        repo,
        config,
        "validate",
        outputs={"validation": output},
        upstream_stages=("run",),
        passed=validation["pass"],
        details={"missing_cells": validation["missing_cells"]},
    )
    return validation


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_stage(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    _verify_receipt(repo, config, "validate")
    validation = _read_json(run_dir(repo, config) / "validation.json", "validation")
    if not validation.get("pass"):
        raise RuntimeError("analysis requires a complete hash-matching validated run")
    cases, labels = _load_split(repo, config, "main")
    results = tuple(read_jsonl(run_dir(repo, config) / "results.jsonl", CellResult))
    expected = build_cell_grid(
        cases=cases,
        systems=config.systems,
        tiers=config.tiers,
        repetitions=config.repetitions,
    )
    scored = score_itt_results(
        cases,
        labels,
        results,
        expected_primary_keys=expected,
    )
    outcome_by_key = {
        (row.document_id, row.system, row.tier): row
        for row in scored.outcomes
        if row.repetition == 0
    }
    paired = tuple(
        PairedCaseOutcome(
            document_id=case.document_id,
            tier=tier,
            monolith_correct=outcome_by_key[(case.document_id, "monolith", tier)].correct,
            verified_search_correct=outcome_by_key[
                (case.document_id, "verified_search", tier)
            ].correct,
        )
        for case in cases
        for tier in config.tiers
    )
    confirmation = confirm_crossover(
        paired,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed,
    )
    crossover = cluster_bootstrap_crossover(
        paired,
        tier_values={name: float(value) for name, value in config.tier_token_limits.items()},
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed,
    )
    result_by_key = {
        (row.case_id, row.system, row.tier): row
        for row in results
        if row.repetition == 0
    }
    metrics = tuple(
        SystemCaseMetric(
            document_id=case.document_id,
            system=system,
            tier=tier,
            correct=outcome_by_key[(case.document_id, system, tier)].correct,
            realized_tokens=float(result_by_key[(case.case_id, system, tier)].trace.realized_tokens),
        )
        for case in cases
        for tier in config.tiers
        for system in config.systems
    )
    pareto = pareto_dominance_probabilities(
        metrics,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed,
    )
    table_dir = run_dir(repo, config) / "analysis" / "tables"
    profile = _read_json(_prepared_dir(repo, config) / "profile.json", "profile")
    _write_csv(
        table_dir / "lineage_rejections.csv",
        ("dataset", "selected_cases", "rejections", "confirmation_role"),
        (
            {
                "dataset": dataset,
                "selected_cases": sum(case.dataset == dataset for case in cases),
                "rejections": profile.get("rejections", {}).get(dataset, 0),
                "confirmation_role": "primary",
            }
            for dataset in ("finqa", "tatqa")
        ),
    )
    diagnostic = _read_json(
        run_dir(repo, config) / "finance_complex_diagnostic" / "boundary_report.json",
        "FinanceComplex boundary report",
    )
    _write_csv(
        table_dir / "diagnostic_boundaries.csv",
        ("domain", "role", "system_run_gate", "confirmation_pool_eligible"),
        ({
            "domain": "FinanceComplexQA",
            "role": diagnostic.get("domain_role", "exploratory_only"),
            "system_run_gate": diagnostic.get("exploratory_system_run_gate", False),
            "confirmation_pool_eligible": False,
        },),
    )
    grouped_tokens: dict[tuple[str, str], list[int]] = defaultdict(list)
    for result in results:
        grouped_tokens[(result.system, result.tier)].append(result.trace.realized_tokens)
    _write_csv(
        table_dir / "resource_manipulation.csv",
        ("system", "tier", "token_ceiling", "median_realized_tokens"),
        (
            {
                "system": system,
                "tier": tier,
                "token_ceiling": config.tier_token_limits[tier],
                "median_realized_tokens": median(grouped_tokens[(system, tier)]),
            }
            for tier in config.tiers
            for system in config.systems
        ),
    )
    _write_csv(
        table_dir / "mechanisms.csv",
        ("system", "tier", "cells", "mean_calls", "mean_candidates", "repairs"),
        (
            {
                "system": system,
                "tier": tier,
                "cells": len(rows := [
                    row for row in results if row.system == system and row.tier == tier
                ]),
                "mean_calls": sum(len(row.trace.call_events) for row in rows) / len(rows),
                "mean_candidates": sum(row.trace.candidate_count for row in rows) / len(rows),
                "repairs": sum(row.trace.repair_attempted for row in rows),
            }
            for tier in config.tiers
            for system in config.systems
        ),
    )
    _write_csv(
        table_dir / "paired_effects.csv",
        ("tier", "difference", "p_value", "direction_rejected", "sesoi_interpretation"),
        (
            {
                "tier": effect.tier,
                "difference": effect.difference,
                "p_value": effect.exact.p_value,
                "direction_rejected": effect.exact.reject,
                "sesoi_interpretation": effect.sesoi_interpretation,
            }
            for effect in (confirmation.low, confirmation.high)
        ),
    )
    _write_csv(
        table_dir / "failures.csv",
        ("system", "tier", "exit_reason", "cells"),
        (
            {"system": system, "tier": tier, "exit_reason": reason, "cells": count}
            for (system, tier, reason), count in sorted(
                Counter(
                    (row.system, row.tier, row.trace.exit_reason) for row in results
                ).items()
            )
        ),
    )
    _write_csv(
        table_dir / "domain_estimates.csv",
        ("dataset", "system", "tier", "correct", "total", "accuracy"),
        (
            {
                "dataset": dataset,
                "system": system,
                "tier": tier,
                "correct": sum(
                    outcome_by_key[(case.document_id, system, tier)].correct
                    for case in cases
                    if case.dataset == dataset
                ),
                "total": sum(case.dataset == dataset for case in cases),
                "accuracy": (
                    sum(
                        outcome_by_key[(case.document_id, system, tier)].correct
                        for case in cases
                        if case.dataset == dataset
                    )
                    / sum(case.dataset == dataset for case in cases)
                ),
            }
            for dataset in ("finqa", "tatqa")
            for tier in config.tiers
            for system in config.systems
        ),
    )
    _write_csv(
        table_dir / "pareto_status.csv",
        (
            "tier",
            "candidate",
            "comparator",
            "resource",
            "dominance_probability",
            "bootstrap_replicates",
        ),
        (
            comparison.model_dump(mode="json")
            for comparison in pareto.comparisons
        ),
    )
    analysis = {
        "schema_version": 1,
        "complete": True,
        "expected_cells": len(expected),
        "observed_cells": len(results),
        "unique_cells": len({(r.case_id, r.system, r.tier, r.repetition) for r in results}),
        "source_manifest_sha256": sha256_path(manifest_path(repo, config)),
        "pilot_gate_sha256": sha256_path(run_dir(repo, config) / "pilot_gate.json"),
        "scripted_results": config.execution_mode == "offline_fixture",
        "non_empirical_fixture": config.execution_mode == "offline_fixture",
        "results_are_empirical": config.execution_mode == "gateway",
        "confirmation": confirmation.model_dump(mode="json"),
        "crossover_bootstrap": crossover.model_dump(mode="json"),
        "finance_complex_role": "exploratory_only",
        "table_interfaces": sorted(path.name for path in table_dir.glob("*.csv")),
    }
    output = run_dir(repo, config) / "analysis" / "analysis.json"
    _write_json(output, analysis)
    status = empirical_claim_status(run_dir(repo, config))
    if analysis["results_are_empirical"] is not status["allowed"]:
        analysis["results_are_empirical"] = status["allowed"]
        _write_json(output, analysis)
    _write_receipt(
        repo,
        config,
        "analyze",
        outputs={
            "analysis": output,
            **{path.name: path for path in sorted(table_dir.glob("*.csv"))},
        },
        upstream_stages=("validate",),
        passed=True,
        details={"results_are_empirical": analysis["results_are_empirical"]},
    )
    return analysis


def empirical_claim_status(run_directory: Path) -> dict[str, Any]:
    reasons: list[str] = []
    paths = {
        "manifest": run_directory / "run_manifest.json",
        "gate": run_directory / "pilot_gate.json",
        "validation": run_directory / "validation.json",
        "analysis": run_directory / "analysis" / "analysis.json",
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            reasons.append(f"missing_{name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reasons.append(f"invalid_{name}")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"invalid_{name}")
            continue
        payloads[name] = payload
    if len(payloads) != len(paths):
        return {"allowed": False, "reasons": reasons}
    manifest = payloads["manifest"]
    gate = payloads["gate"]
    validation = payloads["validation"]
    analysis = payloads["analysis"]
    if manifest.get("non_empirical_fixture"):
        reasons.append("non_empirical_fixture")
    if gate.get("passed") is not True:
        reasons.append("failed_pilot_gate")
    if gate.get("override_allowed") is not False:
        reasons.append("overridable_pilot_gate")
    current_manifest_hash = sha256_path(paths["manifest"])
    current_gate_hash = sha256_path(paths["gate"])
    if manifest.get("identity", {}).get("pilot_gate_sha256") != current_gate_hash:
        reasons.append("pilot_gate_hash_mismatch")
    if validation.get("source_manifest_sha256") != current_manifest_hash:
        reasons.append("validation_manifest_hash_mismatch")
    if analysis.get("source_manifest_sha256") != current_manifest_hash:
        reasons.append("analysis_manifest_hash_mismatch")
    if validation.get("pilot_gate_sha256") != current_gate_hash:
        reasons.append("validation_gate_hash_mismatch")
    if analysis.get("pilot_gate_sha256") != current_gate_hash:
        reasons.append("analysis_gate_hash_mismatch")
    if validation.get("pass") is not True:
        reasons.append("failed_validation")
    if validation.get("complete_grid") is not True:
        reasons.append("incomplete_grid")
    if validation.get("unique_grid") is not True:
        reasons.append("duplicate_grid")
    if validation.get("authoritative_usage") is not True:
        reasons.append("non_authoritative_usage")
    if validation.get("protocol_violation_count") != 0:
        reasons.append("protocol_violations")
    if analysis.get("complete") is not True:
        reasons.append("incomplete_analysis")
    if not (
        analysis.get("expected_cells")
        == analysis.get("observed_cells")
        == analysis.get("unique_cells")
    ):
        reasons.append("analysis_grid_mismatch")
    if analysis.get("scripted_results"):
        reasons.append("scripted_results")
    return {"allowed": not reasons, "reasons": sorted(set(reasons))}


def generate_paper_results(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Generate gated prose, including a protocol-only section before any run."""
    status = empirical_claim_status(run_dir(repo, config))
    generated = repo / "paper" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    analysis_path = run_dir(repo, config) / "analysis" / "analysis.json"
    analysis = _read_json(analysis_path, "analysis") if analysis_path.exists() else {}
    if status["allowed"]:
        confirmation = analysis["confirmation"]
        section = rf"""\section{{Results}}

The complete validated gateway run is eligible for empirical interpretation.
The preregistered intersection-union result is
\texttt{{{confirmation['label']}}}. The low-tier paired difference was
{confirmation['low']['difference']:.3f}, and the high-tier paired difference was
{confirmation['high']['difference']:.3f}. Equality was not treated as a crossing.
FinanceComplexQA remains exploratory and was not pooled into confirmation.
"""
    else:
        section = r"""\section{Results Status}

\begin{statusbox}
\textbf{Protocol-only build.} No empirical conclusion is available. The offline
fixture, if present, is deterministic scripted software evidence only and cannot
support or falsify the scientific hypothesis.
\end{statusbox}

The current conditional crossover hypothesis is neither supported nor cleanly falsified.
Empirical prose remains gated on a complete, unique, hash-matching,
protocol-valid gateway grid and its non-overridable pilot gate.
"""
    section_path = generated / "results_section.tex"
    section_path.write_text(section, encoding="utf-8")
    status_path = generated / "results_status.json"
    _write_json(
        status_path,
        {
            "validated_empirical_results": status["allowed"],
            "reasons": status["reasons"],
            "source_run": str(run_dir(repo, config)),
        },
    )
    return {
        "validated_empirical_results": status["allowed"],
        "reasons": status["reasons"],
        "results_section": str(section_path),
        "results_status": str(status_path),
    }


def build_paper_stage(*, repo: Path, config: ExperimentConfig) -> dict[str, Any]:
    _verify_receipt(repo, config, "analyze")
    generated_status = generate_paper_results(repo=repo, config=config)
    generated = repo / "paper" / "generated"
    section_path = generated / "results_section.tex"
    status_path = generated / "results_status.json"
    _write_receipt(
        repo,
        config,
        "build-paper",
        outputs={"results_section": section_path, "results_status": status_path},
        upstream_stages=("analyze",),
        passed=True,
        details={
            "empirical_claims_allowed": generated_status["validated_empirical_results"]
        },
    )
    return {
        "allowed": generated_status["validated_empirical_results"],
        "reasons": generated_status["reasons"],
    }


class DeterministicOfflineCompletionClient:
    """Exact deterministic fixture client; never an empirical model substitute."""

    tokenizer_id = "NON_EMPIRICAL_BYTE_TOKENIZER_V1"
    tokenizer_sha256 = hashlib.sha256(tokenizer_id.encode("ascii")).hexdigest()

    def __init__(self) -> None:
        self._pending_prompt_tokens: list[int] = []

    async def count_prompt_tokens(self, *, model: str, system: str, user: str) -> int:
        if model != "gpt-5.4-mini":
            raise RuntimeError("offline fixture model substitution")
        payload = json.dumps(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        count = len(payload)
        self._pending_prompt_tokens.append(count)
        return count

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stage: str,
    ) -> GatewayResponse:
        del system
        if model != "gpt-5.4-mini" or not self._pending_prompt_tokens:
            raise RuntimeError("offline fixture completion protocol violation")
        prompt_tokens = self._pending_prompt_tokens.pop(0)
        if stage == "preflight":
            text = '{"status":"ok"}'
        elif stage == "planner":
            text = '{"steps":["locate the fixture value"],"queries":["fixture value"]}'
        else:
            match = re.search(r'"evidence_id":\s*"([^"]+)"', user)
            citations = [match.group(1)] if match else []
            text = json.dumps(
                {
                    "value": "10",
                    "unit": None,
                    "scale": "ones",
                    "entity": None,
                    "period": None,
                    "expression": "10",
                    "citations": citations,
                },
                separators=(",", ":"),
            )
        completion_tokens = min(max_tokens, len(text.encode("utf-8")))
        return GatewayResponse(
            text=text,
            model="gpt-5.4-mini",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_seconds=0.0,
            credential_slot=0,
            request_id="NON_EMPIRICAL_OFFLINE_FIXTURE",
        )


def _offline_config(repo: Path) -> ExperimentConfig:
    raw = repo / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ("finqa", "tatqa", "financecomplexqa"):
        path = raw / f"offline_{name}.json"
        path.write_text(
            json.dumps({"status": "NON_EMPIRICAL_OFFLINE_FIXTURE", "source": name}) + "\n",
            encoding="utf-8",
        )
        sources[name] = SourceSnapshotConfig(path=path, sha256=sha256_path(path))
    lock = repo / "uv.lock"
    if not lock.exists():
        lock.write_text("NON_EMPIRICAL_OFFLINE_FIXTURE\n", encoding="utf-8")
    return ExperimentConfig(
        experiment_name="offline-fixture",
        execution_mode="offline_fixture",
        finqa_snapshot=sources["finqa"],
        tatqa_snapshot=sources["tatqa"],
        finance_complex_snapshot=sources["financecomplexqa"],
        prepared_data_dir=Path("data/processed/offline-fixture"),
        run_root=Path("experiments/runs"),
        tokenizer_id=DeterministicOfflineCompletionClient.tokenizer_id,
        tokenizer_sha256=DeterministicOfflineCompletionClient.tokenizer_sha256,
        development_cases=2,
        operational_pilot_cases=2,
        main_cases=2,
        easy_reserve_cases=2,
        bootstrap_replicates=20,
        max_concurrency=1,
    )


def run_offline_fixture(repo: Path) -> dict[str, Any]:
    config = _offline_config(repo)
    client = DeterministicOfflineCompletionClient()
    prepare_stage(repo=repo, config=config)
    diagnose_finance_complex_stage(repo=repo, config=config)
    preflight_stage(repo=repo, config=config, client=client)
    develop_stage(repo=repo, config=config)
    pilot_stage(repo=repo, config=config, client=client)
    gate = gate_stage(repo=repo, config=config)
    if not gate["passed"]:
        raise RuntimeError(f"offline fixture gate unexpectedly failed: {gate['failed_components']}")
    run_stage(repo=repo, config=config, client=client)
    validation = validate_stage(repo=repo, config=config)
    if not validation["pass"]:
        raise RuntimeError("offline fixture validation unexpectedly failed")
    analyze_stage(repo=repo, config=config)
    build_paper_stage(repo=repo, config=config)
    return {
        "stages": list(STAGES),
        "config": config,
        "run_dir": str(run_dir(repo, config)),
    }
