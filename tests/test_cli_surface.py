from typer.testing import CliRunner

from budget_crossover.cli import app


def test_help_exposes_only_canonical_experiment_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in [
        "prepare",
        "preflight",
        "calibrate",
        "pilot",
        "run",
        "status",
        "analyze",
        "validate",
    ]:
        assert command in result.stdout

    for legacy in ["-v", "judge"]:
        assert legacy not in result.stdout
