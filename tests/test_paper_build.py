from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from budget_crossover.config import ExperimentConfig

BUILD_PATH = Path(__file__).resolve().parents[1] / "paper" / "build_paper.py"
SPEC = spec_from_file_location("paper_build", BUILD_PATH)
assert SPEC is not None and SPEC.loader is not None
build_paper = module_from_spec(SPEC)
SPEC.loader.exec_module(build_paper)


def test_standalone_protocol_build_is_available_before_any_run(tmp_path: Path):
    config = ExperimentConfig(
        experiment_name="not-run",
        execution_mode="offline_fixture",
        tokenizer_sha256="a" * 64,
        development_cases=2,
        operational_pilot_cases=2,
        main_cases=2,
        easy_reserve_cases=2,
    )

    status = build_paper.build(repo=tmp_path, config=config)

    assert status["validated_empirical_results"] is False
    section = (tmp_path / "paper" / "generated" / "results_section.tex").read_text()
    assert "No empirical conclusion is available" in section
    assert "neither supported nor cleanly falsified" in section
