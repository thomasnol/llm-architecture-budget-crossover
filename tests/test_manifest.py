import json
from pathlib import Path

import pytest

from budget_crossover.config import ExperimentConfig
from budget_crossover.manifest import ensure_manifest, run_dir
from budget_crossover.records import Case


def _case() -> Case:
    return Case(
        case_id="case",
        pair_id="pair",
        counterfactual_variant="observed",
        source_row_id="source",
        state="DC",
        historical_action="originated",
        policy_decision="approve",
        policy_reason_codes=["meets_policy"],
        documents={
            "application": "application",
            "collateral": "collateral",
            "credit": "credit",
            "quality_control": "quality",
            "compliance_monitoring": "monitoring",
        },
        protected_attributes={"race": "White"},
        changed_protected_attribute="race",
        complexity="routine",
    )


def test_manifest_freezes_gateway_protocol_and_deployments_without_secrets(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.internal/v1")
    monkeypatch.setenv("LLM_GATEWAY_TOKEN_URL", "https://gateway.internal/token")
    monkeypatch.setenv("LLM_GATEWAY_CHAT_PATH", "/chat/completions")
    monkeypatch.setenv("LLM_GATEWAY_MAX_TOKENS_FIELD", "max_tokens")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "client-one")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "top-secret")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "gpt-5.4-mini,claude-sonnet-4-6")
    config = ExperimentConfig(
        experiment_name="manifest-test",
        hmda_source_sha256="0" * 64,
        require_preflight=False,
    )
    preflight = run_dir(tmp_path, config) / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(
        json.dumps(
            {
                "pass": True,
                "checks": [
                    {
                        "model": "gpt-5.4-mini",
                        "resolved_model": "gpt-5.4-mini-2026-03-17-eastus-dz",
                        "credential_slot": 1,
                        "pass": True,
                    }
                ],
            }
        )
    )

    manifest = ensure_manifest(tmp_path, config, [_case()])
    serialized = json.dumps(manifest)

    assert manifest["gateway_protocol"]["base_url"] == "https://gateway.internal/v1"
    assert manifest["gateway_protocol"]["max_tokens_field"] == "max_tokens"
    assert manifest["credential_model_patterns"] == {"1": ["gpt-5.4-mini", "claude-sonnet-4-6"]}
    assert manifest["resolved_deployments"] == {
        "gpt-5.4-mini": ["gpt-5.4-mini-2026-03-17-eastus-dz"]
    }
    assert "top-secret" not in serialized
    assert "client-one" not in serialized

    monkeypatch.setenv("LLM_GATEWAY_MAX_TOKENS_FIELD", "max_completion_tokens")
    with pytest.raises(RuntimeError, match="gateway_protocol"):
        ensure_manifest(tmp_path, config, [_case()])


def test_manifest_freezes_dependency_lock(monkeypatch, tmp_path: Path):
    lock = tmp_path / "uv.lock"
    lock.write_text("revision-one")
    config = ExperimentConfig(
        experiment_name="lock-test",
        hmda_source_sha256="0" * 64,
        require_preflight=False,
    )

    manifest = ensure_manifest(tmp_path, config, [_case()])

    assert manifest["dependency_lock"]["path"] == "uv.lock"
    assert len(manifest["dependency_lock"]["sha256"]) == 64

    lock.write_text("revision-two")
    with pytest.raises(RuntimeError, match="dependency_lock"):
        ensure_manifest(tmp_path, config, [_case()])
