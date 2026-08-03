from typer.testing import CliRunner

from budget_crossover.cli import app


def test_help_exposes_exact_linear_canonical_workflow():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    commands = (
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
    )
    for command in commands:
        assert command in result.stdout
    for removed in ("calibrate", "status", "judge", "gateway-check"):
        invocation = CliRunner().invoke(app, [removed])
        assert invocation.exit_code != 0
        assert "No such command" in invocation.stderr
