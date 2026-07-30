from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .v3_config import V3Config
from .v3_models import V3Case
from .v3_systems import PROMPT_VERSION

MANIFEST_SCHEMA_VERSION = 1


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _source_hash(repo: Path) -> str:
    source = repo / "src" / "budget_crossover"
    paths = sorted(
        [
            *source.glob("v3_*.py"),
            source / "gateway.py",
            source / "models.py",
            source / "io.py",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
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


def v3_run_dir(repo: Path, config: V3Config) -> Path:
    return repo / "experiments" / "runs" / config.experiment_name


def v3_manifest_path(repo: Path, config: V3Config) -> Path:
    return v3_run_dir(repo, config) / "run_manifest.json"


def _immutable_payload(
    repo: Path,
    config: V3Config,
    cases: list[V3Case],
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
        "hmda_source_sha256": config.hmda_source_sha256,
        "source_sha256": _source_hash(repo),
        "git_commit": git_commit,
        "git_dirty_at_start": git_dirty,
        "generator_model": config.generator_model,
        "supervisor_model": config.supervisor_model,
        "seed": config.seed,
    }


def ensure_v3_manifest(
    repo: Path,
    config: V3Config,
    cases: list[V3Case],
) -> dict[str, Any]:
    path = v3_manifest_path(repo, config)
    current = _immutable_payload(repo, config, cases)
    if path.exists():
        existing = json.loads(path.read_text())
        mismatches = [
            key for key, value in current.items() if existing.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "v3 run manifest mismatch; refusing to combine outputs across "
                f"changed code, config, data, or cases: {', '.join(mismatches)}"
            )
        return existing
    generation_path = path.parent / "generations.jsonl"
    if generation_path.exists() and generation_path.stat().st_size:
        raise RuntimeError(
            "v3 generations exist without a manifest; move the old run directory "
            "aside before starting this frozen experiment"
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


def record_v3_phase(
    repo: Path,
    config: V3Config,
    *,
    phase: str,
    counters: dict[str, Any],
) -> None:
    path = v3_manifest_path(repo, config)
    payload = json.loads(path.read_text())
    payload.setdefault("phases", {})[phase] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "counters": counters,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
