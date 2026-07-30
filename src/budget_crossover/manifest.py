from __future__ import annotations

"""Immutable experiment manifest management."""

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .records import Case
from .systems import PROMPT_REVISION

MANIFEST_SCHEMA_VERSION = 2


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _dependency_lock(repo: Path) -> dict[str, str | None]:
    path = repo / "uv.lock"
    return {
        "path": "uv.lock",
        "sha256": _sha256_bytes(path.read_bytes()) if path.exists() else None,
    }


def _source_hash(repo: Path) -> str:
    source = repo / "src" / "budget_crossover"
    paths = sorted(
        [
            *source.glob("*.py"),
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


def run_dir(repo: Path, config: ExperimentConfig) -> Path:
    return repo / "experiments" / "runs" / config.experiment_name


def manifest_path(repo: Path, config: ExperimentConfig) -> Path:
    return run_dir(repo, config) / "run_manifest.json"


def _immutable_payload(
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
) -> dict[str, Any]:
    config_payload = config.model_dump(mode="json")
    case_payload = [case.model_dump(mode="json") for case in cases]
    git_commit, git_dirty = _git_state(repo)
    default_models = {
        1: ("gpt-5.4,gpt-5.4-mini,gpt-5.4-nano,claude-opus-4-6,claude-sonnet-4-6"),
        2: "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano",
    }
    credential_patterns: dict[str, list[str]] = {}
    for index in (1, 2):
        configured = bool(
            os.getenv(f"LLM_GATEWAY_API_KEY_{index}")
            or (
                os.getenv(f"LLM_GATEWAY_CLIENT_ID_{index}")
                and os.getenv(f"LLM_GATEWAY_CLIENT_SECRET_{index}")
            )
        )
        if configured:
            raw = os.getenv(
                f"LLM_GATEWAY_MODELS_{index}",
                default_models[index],
            )
            credential_patterns[str(index)] = [
                item.strip() for item in raw.split(",") if item.strip()
            ]
    gateway_protocol = {
        "base_url": os.getenv("LLM_GATEWAY_BASE_URL", "").rstrip("/"),
        "token_url": os.getenv("LLM_GATEWAY_TOKEN_URL", ""),
        "chat_path": os.getenv(
            "LLM_GATEWAY_CHAT_PATH",
            "/chat/completions",
        ),
        "max_tokens_field": os.getenv(
            "LLM_GATEWAY_MAX_TOKENS_FIELD",
            "max_tokens",
        ),
        "oauth_basic_auth": os.getenv(
            "LLM_GATEWAY_OAUTH_BASIC_AUTH",
            "",
        ).lower()
        in {"1", "true", "yes", "on"},
        "api_key_header": os.getenv(
            "LLM_GATEWAY_API_KEY_HEADER",
            "Authorization",
        ),
    }
    resolved: dict[str, set[str]] = {}
    preflight_path = run_dir(repo, config) / "preflight.json"
    if preflight_path.exists():
        try:
            preflight = json.loads(preflight_path.read_text())
            for check in preflight.get("checks", []):
                model = check.get("model")
                deployment = check.get("resolved_model")
                if model and deployment and check.get("pass"):
                    resolved.setdefault(str(model), set()).add(str(deployment))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": config.experiment_name,
        "prompt_revision": PROMPT_REVISION,
        "config": config_payload,
        "config_sha256": _canonical_hash(config_payload),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "cases_sha256": _canonical_hash(case_payload),
        "hmda_source_sha256": config.hmda_source_sha256,
        "source_sha256": _source_hash(repo),
        "dependency_lock": _dependency_lock(repo),
        "gateway_protocol": gateway_protocol,
        "credential_model_patterns": credential_patterns,
        "resolved_deployments": {
            model: sorted(deployments) for model, deployments in sorted(resolved.items())
        },
        "git_commit": git_commit,
        "git_dirty_at_start": git_dirty,
        "generator_model": config.generator_model,
        "supervisor_model": config.supervisor_model,
        "seed": config.seed,
    }


def ensure_manifest(
    repo: Path,
    config: ExperimentConfig,
    cases: list[Case],
) -> dict[str, Any]:
    path = manifest_path(repo, config)
    current = _immutable_payload(repo, config, cases)
    if path.exists():
        existing = json.loads(path.read_text())
        mismatches = [key for key, value in current.items() if existing.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "run manifest mismatch; refusing to combine outputs across "
                f"changed code, config, data, or cases: {', '.join(mismatches)}"
            )
        return existing
    generation_path = path.parent / "generations.jsonl"
    if generation_path.exists() and generation_path.stat().st_size:
        raise RuntimeError(
            "generations exist without a manifest; move the old run directory "
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


def record_phase(
    repo: Path,
    config: ExperimentConfig,
    *,
    phase: str,
    counters: dict[str, Any],
) -> None:
    path = manifest_path(repo, config)
    payload = json.loads(path.read_text())
    payload.setdefault("phases", {})[phase] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "counters": counters,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
