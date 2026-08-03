import json
from pathlib import Path

import pytest

from budget_crossover.workflow import (
    analyze_stage,
    empirical_claim_status,
    run_offline_fixture,
    validate_stage,
)


def test_offline_fixture_runs_every_stage_but_can_never_emit_empirical_claims(tmp_path: Path):
    result = run_offline_fixture(tmp_path)

    assert result["stages"] == [
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
    ]
    run_dir = Path(result["run_dir"])
    status = empirical_claim_status(run_dir)
    assert status["allowed"] is False
    assert "non_empirical_fixture" in status["reasons"]
    assert "scripted_results" in status["reasons"]
    assert json.loads((run_dir / "validation.json").read_text())["pass"] is True
    assert json.loads((run_dir / "analysis" / "analysis.json").read_text())[
        "results_are_empirical"
    ] is False
    generated = (tmp_path / "paper" / "generated" / "results_section.tex").read_text()
    assert "no empirical conclusion is available" in generated.lower()


def test_offline_fixture_resumes_without_changing_manifest_bound_inputs(tmp_path: Path):
    first = run_offline_fixture(tmp_path)
    run_dir = Path(first["run_dir"])
    manifest_before = (run_dir / "run_manifest.json").read_bytes()
    preflight_before = (run_dir / "preflight.json").read_bytes()
    prepared_hashes_before = (
        tmp_path / first["config"].prepared_data_dir / "workflow_hashes.json"
    ).read_bytes()

    second = run_offline_fixture(tmp_path)

    assert second["stages"] == first["stages"]
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before
    assert (run_dir / "preflight.json").read_bytes() == preflight_before
    assert (
        tmp_path / first["config"].prepared_data_dir / "workflow_hashes.json"
    ).read_bytes() == prepared_hashes_before


def test_downstream_analysis_verifies_transitive_stage_outputs(tmp_path: Path):
    result = run_offline_fixture(tmp_path)
    run_dir = Path(result["run_dir"])
    results_path = run_dir / "results.jsonl"
    results_path.write_text(results_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="run output hash mismatch: results"):
        analyze_stage(repo=tmp_path, config=result["config"])


def test_incomplete_grid_and_manifest_hash_changes_fail_closed(tmp_path: Path):
    result = run_offline_fixture(tmp_path)
    config = result["config"]
    run_dir = Path(result["run_dir"])
    results_path = run_dir / "results.jsonl"
    lines = results_path.read_text().splitlines()
    results_path.write_text("\n".join(lines[:-1]) + "\n")

    validation = validate_stage(repo=tmp_path, config=config)

    assert validation["pass"] is False
    assert validation["complete_grid"] is False
    assert empirical_claim_status(run_dir)["allowed"] is False

    public_main = tmp_path / config.prepared_data_dir / "public" / "main.jsonl"
    public_main.write_text(public_main.read_text() + "{}\n")
    with pytest.raises(RuntimeError, match="artifact_hashes"):
        validate_stage(repo=tmp_path, config=config)


def test_failed_gate_protocol_violation_and_hash_mismatch_cannot_produce_claims(
    tmp_path: Path,
):
    result = run_offline_fixture(tmp_path)
    run_dir = Path(result["run_dir"])

    gate_path = run_dir / "pilot_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["passed"] = False
    gate_path.write_text(json.dumps(gate))
    status = empirical_claim_status(run_dir)
    assert status["allowed"] is False
    assert "failed_pilot_gate" in status["reasons"]
    assert "pilot_gate_hash_mismatch" in status["reasons"]

    validation_path = run_dir / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation["protocol_violation_count"] = 1
    validation_path.write_text(json.dumps(validation))
    status = empirical_claim_status(run_dir)
    assert "protocol_violations" in status["reasons"]
