from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .v2_config import V2Config
from .v2_models import V2Case
from .v2_systems import PROMPT_VERSION

MANIFEST_SCHEMA_VERSION = 1


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.exists() else None


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _source_hash(repo: Path) -> str:
    paths = sorted((repo / "src" / "budget_crossover").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(repo: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_manifest_path(repo: Path, config: V2Config) -> Path:
    return (
        repo
        / "experiments"
        / "runs"
        / config.experiment_name
        / "run_manifest.json"
    )


def _immutable_payload(
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
) -> dict[str, Any]:
    config_payload = config.model_dump(mode="json")
    case_payload = [case.model_dump(mode="json") for case in cases]
    git_commit, git_dirty = _git_state(repo)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": config.experiment_name,
        "prompt_version": PROMPT_VERSION,
        "config": config_payload,
        "config_sha256": _canonical_hash(config_payload),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "cases_sha256": _canonical_hash(case_payload),
        "datasets_sha256": {
            "insurance": _sha256_file(repo / "data" / "raw" / "train.parquet"),
            "mmlu_pro": _sha256_file(
                repo / "data" / "raw" / "mmlu_pro_test.parquet"
            ),
        },
        "source_sha256": _source_hash(repo),
        "git_commit": git_commit,
        "git_dirty_at_start": git_dirty,
        "models": {
            "generator": config.generator_model,
            "verifier": config.verifier_model,
            "judges": config.judge_models,
        },
        "seed": config.seed,
    }


def ensure_run_manifest(
    repo: Path,
    config: V2Config,
    cases: list[V2Case],
) -> dict[str, Any]:
    path = run_manifest_path(repo, config)
    current = _immutable_payload(repo, config, cases)
    if path.exists():
        existing = json.loads(path.read_text())
        mismatches = [
            key for key, value in current.items() if existing.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "run manifest mismatch; refusing to mix outputs across changed "
                f"code, config, data, or cases: {', '.join(mismatches)}"
            )
        return existing
    generation_path = path.parent / "generations.jsonl"
    if generation_path.exists() and generation_path.stat().st_size:
        raise RuntimeError(
            "generation outputs exist without a run manifest; move the old run "
            "directory aside before starting this frozen experiment"
        )
    payload = {
        **current,
        "created_at": datetime.now(UTC).isoformat(),
        "phases": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
    return payload


def record_phase(
    repo: Path,
    config: V2Config,
    *,
    phase: str,
    counters: dict[str, int | float],
) -> None:
    path = run_manifest_path(repo, config)
    payload = json.loads(path.read_text())
    payload.setdefault("phases", {})[phase] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "counters": counters,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def ensure_judge_sample_manifest(
    *,
    repo: Path,
    config: V2Config,
    selected_run_ids: list[str],
) -> dict[str, Any]:
    path = run_manifest_path(repo, config).parent / "judge_sample_manifest.json"
    current = {
        "experiment": config.experiment_name,
        "seed": config.seed,
        "judge_models": config.judge_models,
        "selected_run_ids": sorted(selected_run_ids),
        "sample_sha256": _canonical_hash(sorted(selected_run_ids)),
    }
    if path.exists():
        existing = json.loads(path.read_text())
        mismatches = [
            key for key, value in current.items() if existing.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "judge sample manifest mismatch; refusing to change the audit "
                f"sample after judging began: {', '.join(mismatches)}"
            )
        return existing
    payload = {**current, "created_at": datetime.now(UTC).isoformat()}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
    return payload
