from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .gateway import GatewayClient, GatewayRequestError
from .manifest import run_dir


def _usage_complete(response: Any) -> bool:
    usage = response.usage
    return (
        usage.prompt_tokens is not None
        and usage.completion_tokens is not None
        and usage.total_tokens is not None
        and usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens
    )


async def run_preflight(
    *,
    repo: Path,
    config: ExperimentConfig,
    client: GatewayClient | None = None,
) -> dict[str, Any]:
    """Exercise every configured experimental model on each eligible slot."""
    owned_client = client is None
    active_client = client or GatewayClient(timeout_seconds=config.request_timeout_seconds)
    models = sorted(
        {config.generator_model}
        if config.study_kind == "architecture"
        else {config.generator_model, config.supervisor_model}
    )
    checks: list[dict[str, Any]] = []
    try:
        for model in models:
            eligible = [slot for slot in active_client.slots if slot.supports(model)]
            if not eligible:
                checks.append(
                    {
                        "model": model,
                        "credential_slot": None,
                        "pass": False,
                        "usage_complete": False,
                        "error": "no eligible credential slot",
                    }
                )
                continue
            for slot in eligible:
                try:
                    response = await active_client.complete(
                        model=model,
                        system="You are validating an LLM gateway contract.",
                        user='Return only {"status":"ok"}.',
                        max_tokens=32,
                        temperature=0.0,
                        stage="preflight",
                        credential_slot=slot.index,
                    )
                    complete_usage = _usage_complete(response)
                    checks.append(
                        {
                            "model": model,
                            "resolved_model": response.model,
                            "credential_slot": slot.index,
                            "request_id": response.request_id,
                            "usage_complete": complete_usage,
                            "usage": response.usage.model_dump(),
                            "pass": complete_usage,
                        }
                    )
                except GatewayRequestError as error:
                    checks.append(
                        {
                            "model": model,
                            "credential_slot": slot.index,
                            "status_code": error.status_code,
                            "request_id": error.request_id,
                            "stage": error.stage,
                            "retryable": error.retryable,
                            "usage_complete": False,
                            "error": error.detail,
                            "pass": False,
                        }
                    )
                except Exception as error:  # noqa: BLE001 - diagnostic boundary
                    checks.append(
                        {
                            "model": model,
                            "credential_slot": slot.index,
                            "usage_complete": False,
                            "error": f"{type(error).__name__}: {error}",
                            "pass": False,
                        }
                    )
    finally:
        if owned_client:
            await active_client.close()

    passed = sum(bool(check["pass"]) for check in checks)
    report = {
        "experiment": config.experiment_name,
        "created_at": datetime.now(UTC).isoformat(),
        "checks_expected": len(checks),
        "checks_passed": passed,
        "pass": bool(checks) and passed == len(checks),
        "checks": checks,
    }
    output = run_dir(repo, config) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
