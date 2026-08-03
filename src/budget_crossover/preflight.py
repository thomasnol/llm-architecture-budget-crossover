from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import EXACT_MODEL, ExperimentConfig
from .gateway import GatewayClient, GatewayCompletionClient, GatewayPromptTokenizer
from .systems import CompletionClient

PREFLIGHT_SYSTEM = "Validate the exact completion and tokenizer contract."
PREFLIGHT_USER = 'Return only {"status":"ok"}.'


def _run_dir(repo: Path, config: ExperimentConfig) -> Path:
    root = config.run_root if config.run_root.is_absolute() else repo / config.run_root
    return root / config.experiment_name


def _strict_status_json(text: str) -> bool:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return type(payload) is dict and payload == {"status": "ok"}


async def run_preflight(
    *,
    repo: Path,
    config: ExperimentConfig,
    client: CompletionClient | None = None,
) -> dict[str, Any]:
    """Verify exact model resolution, JSON, usage, and tokenizer agreement."""
    owned_gateway: GatewayClient | None = None
    active_client: CompletionClient
    if client is None:
        owned_gateway = GatewayClient(timeout_seconds=config.request_timeout_seconds)
        tokenizer = GatewayPromptTokenizer(
            owned_gateway,
            tokenizer_id=config.tokenizer_id,
            tokenizer_sha256=config.tokenizer_sha256,
        )
        active_client = GatewayCompletionClient(
            gateway=owned_gateway,
            tokenizer=tokenizer,
            model=config.model,
            tokenizer_id=config.tokenizer_id,
            tokenizer_sha256=config.tokenizer_sha256,
        )
    else:
        active_client = client

    tokenizer_id = str(getattr(active_client, "tokenizer_id", ""))
    tokenizer_sha256 = str(getattr(active_client, "tokenizer_sha256", ""))
    counted_tokens: int | None = None
    response: Any = None
    error: str | None = None
    try:
        counted_tokens = await active_client.count_prompt_tokens(
            model=config.model,
            system=PREFLIGHT_SYSTEM,
            user=PREFLIGHT_USER,
        )
        response = await active_client.complete(
            model=config.model,
            system=PREFLIGHT_SYSTEM,
            user=PREFLIGHT_USER,
            max_tokens=32,
            stage="preflight",
        )
    except Exception as caught:  # noqa: BLE001 - diagnostic boundary reports failure
        error = f"{type(caught).__name__}: {caught}"
    finally:
        if owned_gateway is not None:
            await owned_gateway.close()

    usage = getattr(response, "usage", None)
    prompt_usage = getattr(usage, "prompt_tokens", None)
    completion_usage = getattr(usage, "completion_tokens", None)
    total_usage = getattr(usage, "total_tokens", None)
    authoritative_usage = (
        type(prompt_usage) is int
        and type(completion_usage) is int
        and type(total_usage) is int
        and total_usage == prompt_usage + completion_usage
    )
    exact_model_resolution = (
        config.model == EXACT_MODEL
        and config.deployment == EXACT_MODEL
        and getattr(response, "model", None) == EXACT_MODEL
    )
    strict_json_valid = response is not None and _strict_status_json(response.text)
    tokenizer_identity_valid = (
        tokenizer_id == config.tokenizer_id
        and tokenizer_sha256 == config.tokenizer_sha256
    )
    tokenizer_consistent = (
        type(counted_tokens) is int
        and counted_tokens >= 0
        and authoritative_usage
        and counted_tokens == prompt_usage
    )
    passed = all(
        (
            error is None,
            exact_model_resolution,
            strict_json_valid,
            authoritative_usage,
            tokenizer_identity_valid,
            tokenizer_consistent,
        )
    )
    non_empirical = config.execution_mode == "offline_fixture"
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": config.experiment_name,
        "requested_model": config.model,
        "resolved_model": getattr(response, "model", None),
        "exact_model_resolution": exact_model_resolution,
        "strict_json_valid": strict_json_valid,
        "authoritative_usage": authoritative_usage,
        "tokenizer_consistent": tokenizer_consistent,
        "tokenizer_identity_valid": tokenizer_identity_valid,
        "tokenizer_id": tokenizer_id,
        "tokenizer_sha256": tokenizer_sha256,
        "counted_prompt_tokens": counted_tokens,
        "usage": usage.model_dump(mode="json") if usage is not None else None,
        "request_id": getattr(response, "request_id", None),
        "pass": passed,
        "non_empirical_fixture": non_empirical,
        "eligible_for_empirical_run": passed and not non_empirical,
        "error": error,
    }
    output = _run_dir(repo, config) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
