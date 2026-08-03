from __future__ import annotations

"""Immutable run identity and separate mutable progress state."""

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .models import PublicCase
from .runner import CellKey, build_cell_grid
from .systems import CANDIDATE_SCHEMA, CORE_INSTRUCTIONS, PROMPT_REVISION

MANIFEST_SCHEMA_VERSION = 3


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required immutable artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def run_dir(repo: Path, config: ExperimentConfig) -> Path:
    return _resolve(repo, config.run_root) / config.experiment_name


def manifest_path(repo: Path, config: ExperimentConfig) -> Path:
    return run_dir(repo, config) / "run_manifest.json"


def run_state_path(repo: Path, config: ExperimentConfig) -> Path:
    return run_dir(repo, config) / "run_state.json"


def _git_identity(repo: Path) -> tuple[str | None, bool | None]:
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
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, not bool(status.strip())


def _package_root(repo: Path) -> Path:
    candidate = repo / "src" / "budget_crossover"
    return candidate if candidate.is_dir() else Path(__file__).resolve().parent


def _hash_files(paths: Sequence[Path], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _version_hashes(repo: Path) -> dict[str, str]:
    package = _package_root(repo)
    return {
        "prompts": sha256_bytes(
            canonical_json_bytes(
                {
                    "revision": PROMPT_REVISION,
                    "core_instructions": CORE_INSTRUCTIONS,
                    "candidate_schema": CANDIDATE_SCHEMA,
                }
            )
        ),
        "systems": sha256_path(package / "systems.py"),
        "checker": sha256_path(package / "checking.py"),
        "retriever": sha256_path(package / "retrieval.py"),
    }


def _code_hash(repo: Path) -> str:
    package = _package_root(repo)
    return _hash_files(tuple(package.glob("*.py")), relative_to=package)


def _credential_patterns() -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {}
    defaults = {
        1: "gpt-5.4-mini",
        2: "gpt-5.4-mini",
    }
    for index in (1, 2):
        configured = bool(
            os.getenv(f"LLM_GATEWAY_API_KEY_{index}")
            or (
                os.getenv(f"LLM_GATEWAY_CLIENT_ID_{index}")
                and os.getenv(f"LLM_GATEWAY_CLIENT_SECRET_{index}")
            )
        )
        if configured:
            values = os.getenv(f"LLM_GATEWAY_MODELS_{index}", defaults[index])
            patterns[str(index)] = [
                value.strip() for value in values.split(",") if value.strip()
            ]
    return patterns


def _gateway_protocol() -> dict[str, Any]:
    return {
        "base_url": os.getenv("LLM_GATEWAY_BASE_URL", "").rstrip("/"),
        "chat_path": os.getenv("LLM_GATEWAY_CHAT_PATH", "/chat/completions"),
        "tokenizer_path": os.getenv("LLM_GATEWAY_TOKENIZER_PATH", ""),
        "max_tokens_field": os.getenv("LLM_GATEWAY_MAX_TOKENS_FIELD", "max_tokens"),
        "api_key_header": os.getenv("LLM_GATEWAY_API_KEY_HEADER", "Authorization"),
        "oauth_basic_auth": os.getenv("LLM_GATEWAY_OAUTH_BASIC_AUTH", "").lower()
        in {"1", "true", "yes", "on"},
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description} artifact: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"invalid {description} artifact: expected a JSON object")
    return payload


def _cell_key_text(key: CellKey) -> str:
    return f"{key.case_id}:{key.system}:{key.tier}:{key.repetition}"


def _source_hashes(repo: Path, config: ExperimentConfig) -> dict[str, str]:
    snapshots = {
        "finqa": config.finqa_snapshot,
        "tatqa": config.tatqa_snapshot,
    }
    if config.finance_complex_snapshot is not None:
        snapshots["finance_complex"] = config.finance_complex_snapshot
    hashes: dict[str, str] = {}
    for name, snapshot in snapshots.items():
        observed = sha256_path(_resolve(repo, snapshot.path))
        if observed != snapshot.sha256:
            raise RuntimeError(f"{name} source hash does not match frozen configuration")
        hashes[name] = observed
    return hashes


def _expected_identity(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: Sequence[PublicCase],
    artifact_paths: Mapping[str, Path],
    preflight_path: Path,
    pilot_gate_path: Path,
) -> dict[str, Any]:
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("manifest case IDs must be unique")
    preflight = _read_json(preflight_path, "preflight")
    pilot_gate = _read_json(pilot_gate_path, "pilot gate")
    if not preflight.get("pass"):
        raise RuntimeError("preflight did not pass; refusing to freeze a run manifest")
    if config.execution_mode == "gateway" and not preflight.get(
        "eligible_for_empirical_run"
    ):
        raise RuntimeError("preflight is not eligible for an empirical gateway run")
    if not pilot_gate.get("passed"):
        raise RuntimeError("pilot gate did not pass; refusing to freeze a run manifest")
    if pilot_gate.get("override_allowed") is not False:
        raise RuntimeError("pilot gate artifact must be explicitly non-overridable")
    commit, clean = _git_identity(repo)
    if config.execution_mode == "gateway" and (commit is None or clean is not True):
        raise RuntimeError("an empirical run requires a clean Git commit")
    dependency_lock = repo / "uv.lock"
    if config.execution_mode == "gateway" and not dependency_lock.is_file():
        raise RuntimeError("an empirical run requires the frozen uv.lock dependency lock")
    grid = build_cell_grid(
        cases=cases,
        systems=config.systems,
        tiers=config.tiers,
        repetitions=config.repetitions,
    )
    return {
        "resolved_config_sha256": sha256_bytes(
            canonical_json_bytes(config.model_dump(mode="json"))
        ),
        "source_hashes": _source_hashes(repo, config),
        "artifact_hashes": {
            name: sha256_path(path) for name, path in sorted(artifact_paths.items())
        },
        "case_inventory": [
            {
                "case_id": case.case_id,
                "dataset": case.dataset,
                "document_id": case.document_id,
                "stratum": case.stratum,
            }
            for case in cases
        ],
        "expected_cell_keys": [_cell_key_text(key) for key in grid],
        "version_hashes": _version_hashes(repo),
        "code_sha256": _code_hash(repo),
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": sha256_path(dependency_lock) if dependency_lock.is_file() else None,
        },
        "model": config.model,
        "deployment": config.deployment,
        "tokenizer": {
            "id": config.tokenizer_id,
            "sha256": config.tokenizer_sha256,
        },
        "retry_policy": config.retry_policy.model_dump(mode="json"),
        "failure_semantics": config.failure_semantics.model_dump(mode="json"),
        "credential_model_patterns": _credential_patterns(),
        "gateway_protocol": _gateway_protocol(),
        "git": {"commit": commit, "clean": clean},
        "preflight_sha256": sha256_path(preflight_path),
        "pilot_gate_sha256": sha256_path(pilot_gate_path),
    }


def _manifest_without_timestamp(
    config: ExperimentConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.experiment_name,
        "non_empirical_fixture": config.execution_mode == "offline_fixture",
        "resolved_config": config.model_dump(mode="json"),
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical_json_bytes(identity)),
    }


def ensure_run_manifest(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: Sequence[PublicCase],
    artifact_paths: Mapping[str, Path],
    preflight_path: Path,
    pilot_gate_path: Path,
) -> dict[str, Any]:
    identity = _expected_identity(
        repo=repo,
        config=config,
        cases=cases,
        artifact_paths=artifact_paths,
        preflight_path=preflight_path,
        pilot_gate_path=pilot_gate_path,
    )
    expected = _manifest_without_timestamp(config, identity)
    path = manifest_path(repo, config)
    if path.exists():
        return verify_run_manifest(
            repo=repo,
            config=config,
            cases=cases,
            artifact_paths=artifact_paths,
            preflight_path=preflight_path,
            pilot_gate_path=pilot_gate_path,
        )
    payload = {**expected, "created_at": datetime.now(UTC).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return payload


def verify_run_manifest(
    *,
    repo: Path,
    config: ExperimentConfig,
    cases: Sequence[PublicCase],
    artifact_paths: Mapping[str, Path],
    preflight_path: Path,
    pilot_gate_path: Path,
) -> dict[str, Any]:
    path = manifest_path(repo, config)
    existing = _read_json(path, "run manifest")
    identity = _expected_identity(
        repo=repo,
        config=config,
        cases=cases,
        artifact_paths=artifact_paths,
        preflight_path=preflight_path,
        pilot_gate_path=pilot_gate_path,
    )
    expected = _manifest_without_timestamp(config, identity)
    mismatches = [
        f"identity.{key}"
        for key, value in identity.items()
        if not isinstance(existing.get("identity"), dict)
        or existing["identity"].get(key) != value
    ]
    mismatches.extend(
        key
        for key in ("schema_version", "run_id", "non_empirical_fixture", "resolved_config")
        if existing.get(key) != expected[key]
    )
    if existing.get("identity_sha256") != expected["identity_sha256"]:
        mismatches.append("identity_sha256")
    if mismatches:
        raise RuntimeError(
            "run manifest mismatch; refusing changed inputs: " + ", ".join(mismatches)
        )
    if not isinstance(existing.get("created_at"), str):
        raise TypeError("run manifest mismatch; missing immutable creation timestamp")
    return existing


def update_run_state(
    *,
    repo: Path,
    config: ExperimentConfig,
    stage: str,
    counters: Mapping[str, Any],
) -> dict[str, Any]:
    if not manifest_path(repo, config).is_file():
        raise RuntimeError("run state cannot exist before the immutable run manifest")
    path = run_state_path(repo, config)
    if path.exists():
        payload = _read_json(path, "run state")
    else:
        payload = {"schema_version": 1, "run_id": config.experiment_name, "stages": {}}
    payload.setdefault("stages", {})[stage] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "counters": dict(counters),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return payload


def ensure_manifest(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError(
        "legacy ensure_manifest was removed; use ensure_run_manifest with exact artifact hashes"
    )


def record_phase(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("manifest mutation was removed; write mutable counters to run_state.json")
