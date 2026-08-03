from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import load_experiment_config
from .workflow import (
    analyze_stage,
    build_paper_stage,
    develop_stage,
    diagnose_finance_complex_stage,
    gate_stage,
    pilot_stage,
    preflight_stage,
    prepare_stage,
    run_stage,
    validate_stage,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "main.yaml"
ConfigOption = Annotated[Path, typer.Option(exists=True, readable=True)]


def _execute(
    operation: Callable[..., dict[str, Any]],
    config_path: Path,
    *,
    require_pass: bool = False,
) -> None:
    config = load_experiment_config(config_path)
    try:
        result = operation(repo=REPO, config=config)
    except (RuntimeError, ValueError, OSError) as error:
        typer.echo(json.dumps({"pass": False, "error": str(error)}, indent=2))
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))
    verdict = result.get("pass", result.get("passed", True))
    if require_pass and verdict is not True:
        raise typer.Exit(code=1)


@app.command()
def prepare(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Prepare pinned FinQA and TAT-QA artifacts without network access."""
    _execute(prepare_stage, config)


@app.command(name="diagnose-finance-complex")
def diagnose_finance_complex(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Run the exploratory FinanceComplexQA boundary diagnostics."""
    _execute(diagnose_finance_complex_stage, config)


@app.command()
def preflight(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Require exact gpt-5.4-mini, strict JSON, usage, and tokenizer agreement."""
    _execute(preflight_stage, config, require_pass=True)


@app.command()
def develop(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Freeze outcome-blind mandatory-action calibration before pilot."""
    _execute(develop_stage, config)


@app.command()
def pilot(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Execute the immutable operational pilot grid."""
    _execute(pilot_stage, config, require_pass=True)


@app.command()
def gate(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Evaluate the non-overridable operational pilot gate."""
    _execute(gate_stage, config, require_pass=True)


@app.command(name="run")
def run_main(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Execute the main grid only after exact upstream hash verification."""
    _execute(run_stage, config, require_pass=True)


@app.command()
def validate(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Validate manifest identity, grid completeness, usage, and protocol."""
    _execute(validate_stage, config, require_pass=True)


@app.command()
def analyze(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Generate confirmatory and exploratory tables from a validated grid."""
    _execute(analyze_stage, config)


@app.command(name="build-paper")
def build_paper(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Generate paper prose through the shared empirical-claim gate."""
    _execute(build_paper_stage, config)
