import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

BUILD_PATH = Path(__file__).resolve().parents[1] / "paper" / "build_paper.py"
SPEC = spec_from_file_location("paper_build", BUILD_PATH)
assert SPEC is not None and SPEC.loader is not None
build_paper = module_from_spec(SPEC)
SPEC.loader.exec_module(build_paper)


def _write_artifacts(run_dir: Path, *, duplicates: int = 0) -> None:
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.json").write_text(
        json.dumps(
            {
                "pass": True,
                "requirements": {
                    "require_generations": True,
                    "require_pilot_gate": True,
                },
            }
        )
    )
    (run_dir / "analysis" / "analysis.json").write_text(
        json.dumps(
            {
                "results_are_empirical": True,
                "diagnostic": False,
                "incomplete": False,
                "expected_generations": 100,
                "observed_generations": 100,
                "unique_generation_cells": 100,
                "missing_generation_cells": 0,
                "extra_generation_cells": 0,
                "duplicate_generation_cells": duplicates,
                "generation_completion_rate": 1.0,
            }
        )
    )


def test_paper_results_gate_requires_complete_unique_validated_grid(
    monkeypatch,
    tmp_path: Path,
):
    run_dir = tmp_path / "main"
    monkeypatch.setattr(build_paper, "RUN_DIR", run_dir)

    valid, _ = build_paper._validated_results()
    assert not valid

    _write_artifacts(run_dir)
    valid, _ = build_paper._validated_results()
    assert valid

    _write_artifacts(run_dir, duplicates=1)
    valid, reason = build_paper._validated_results()
    assert not valid
    assert "gates do not pass" in reason
