import hashlib
import json
from pathlib import Path

import pytest

from budget_crossover.config import ExperimentConfig, SourceSnapshotConfig
from budget_crossover.manifest import (
    ensure_run_manifest,
    manifest_path,
    update_run_state,
    verify_run_manifest,
)
from budget_crossover.models import EvidenceItem, PublicCase


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case() -> PublicCase:
    return PublicCase(
        case_id="case-1",
        dataset="finqa",
        document_id="document-1",
        question="What is the value?",
        evidence=(
            EvidenceItem(
                evidence_id="evidence-1",
                document_id="document-1",
                kind="text",
                text="The value is 10.",
                ordinal=0,
            ),
        ),
        stratum="headroom",
    )


def _inputs(tmp_path: Path) -> tuple[ExperimentConfig, dict[str, Path], Path, Path]:
    (tmp_path / "uv.lock").write_text("frozen-dependencies")
    finqa = tmp_path / "finqa.json"
    tatqa = tmp_path / "tatqa.json"
    finance = tmp_path / "finance.json"
    public = tmp_path / "public.jsonl"
    for path, content in (
        (finqa, "finqa-source"),
        (tatqa, "tatqa-source"),
        (finance, "finance-source"),
        (public, '{"case_id":"case-1"}\n'),
    ):
        path.write_text(content)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "pass": True,
                "eligible_for_empirical_run": True,
                "resolved_model": "gpt-5.4-mini",
                "tokenizer_sha256": "c" * 64,
            }
        )
    )
    pilot_gate = tmp_path / "pilot_gate.json"
    pilot_gate.write_text(json.dumps({"passed": True, "override_allowed": False}))
    config = ExperimentConfig(
        experiment_name="manifest-test",
        finqa_snapshot=SourceSnapshotConfig(path=finqa, sha256=_sha(finqa)),
        tatqa_snapshot=SourceSnapshotConfig(path=tatqa, sha256=_sha(tatqa)),
        finance_complex_snapshot=SourceSnapshotConfig(path=finance, sha256=_sha(finance)),
        tokenizer_id="tokenizer-v1",
        tokenizer_sha256="c" * 64,
        development_cases=2,
        operational_pilot_cases=2,
        main_cases=2,
        easy_reserve_cases=2,
    )
    return config, {"public_main": public}, preflight, pilot_gate


def test_manifest_binds_complete_identity_without_credentials(monkeypatch, tmp_path: Path):
    config, artifacts, preflight, pilot_gate = _inputs(tmp_path)
    monkeypatch.setattr(
        "budget_crossover.manifest._git_identity", lambda _repo: ("1" * 40, True)
    )
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID_1", "client-secret-name")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET_1", "top-secret")
    monkeypatch.setenv("LLM_GATEWAY_MODELS_1", "gpt-5.4-mini")

    manifest = ensure_run_manifest(
        repo=tmp_path,
        config=config,
        cases=(_case(),),
        artifact_paths=artifacts,
        preflight_path=preflight,
        pilot_gate_path=pilot_gate,
    )

    serialized = json.dumps(manifest)
    identity = manifest["identity"]
    assert identity["case_inventory"] == [
        {
            "case_id": "case-1",
            "dataset": "finqa",
            "document_id": "document-1",
            "stratum": "headroom",
        }
    ]
    assert len(identity["expected_cell_keys"]) == 9
    assert identity["model"] == "gpt-5.4-mini"
    assert identity["deployment"] == "gpt-5.4-mini"
    assert identity["tokenizer"] == {"id": "tokenizer-v1", "sha256": "c" * 64}
    assert identity["git"] == {"commit": "1" * 40, "clean": True}
    assert identity["credential_model_patterns"] == {"1": ["gpt-5.4-mini"]}
    assert set(identity["version_hashes"]) == {"prompts", "systems", "checker", "retriever"}
    assert identity["preflight_sha256"] == _sha(preflight)
    assert identity["pilot_gate_sha256"] == _sha(pilot_gate)
    assert "top-secret" not in serialized
    assert "client-secret-name" not in serialized


def test_manifest_verification_refuses_any_upstream_hash_change(monkeypatch, tmp_path: Path):
    config, artifacts, preflight, pilot_gate = _inputs(tmp_path)
    monkeypatch.setattr(
        "budget_crossover.manifest._git_identity", lambda _repo: ("1" * 40, True)
    )
    ensure_run_manifest(
        repo=tmp_path,
        config=config,
        cases=(_case(),),
        artifact_paths=artifacts,
        preflight_path=preflight,
        pilot_gate_path=pilot_gate,
    )

    artifacts["public_main"].write_text("changed")

    with pytest.raises(RuntimeError, match="artifact_hashes"):
        verify_run_manifest(
            repo=tmp_path,
            config=config,
            cases=(_case(),),
            artifact_paths=artifacts,
            preflight_path=preflight,
            pilot_gate_path=pilot_gate,
        )


def test_mutable_progress_updates_only_run_state(monkeypatch, tmp_path: Path):
    config, artifacts, preflight, pilot_gate = _inputs(tmp_path)
    monkeypatch.setattr(
        "budget_crossover.manifest._git_identity", lambda _repo: ("1" * 40, True)
    )
    ensure_run_manifest(
        repo=tmp_path,
        config=config,
        cases=(_case(),),
        artifact_paths=artifacts,
        preflight_path=preflight,
        pilot_gate_path=pilot_gate,
    )
    frozen = manifest_path(tmp_path, config).read_bytes()

    state = update_run_state(
        repo=tmp_path,
        config=config,
        stage="run",
        counters={"completed": 4, "remaining": 5},
    )

    assert state["stages"]["run"]["counters"] == {"completed": 4, "remaining": 5}
    assert manifest_path(tmp_path, config).read_bytes() == frozen


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("preflight", {"pass": False, "eligible_for_empirical_run": False}, "preflight"),
        ("pilot_gate", {"passed": False, "override_allowed": False}, "pilot gate"),
    ],
)
def test_empirical_manifest_cannot_freeze_failed_prerequisites(
    monkeypatch,
    tmp_path: Path,
    filename: str,
    payload: dict,
    message: str,
):
    config, artifacts, preflight, pilot_gate = _inputs(tmp_path)
    monkeypatch.setattr(
        "budget_crossover.manifest._git_identity", lambda _repo: ("1" * 40, True)
    )
    target = preflight if filename == "preflight" else pilot_gate
    target.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=message):
        ensure_run_manifest(
            repo=tmp_path,
            config=config,
            cases=(_case(),),
            artifact_paths=artifacts,
            preflight_path=preflight,
            pilot_gate_path=pilot_gate,
        )
