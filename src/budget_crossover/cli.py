from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from .analysis import analyze_run
from .calibration import calibrate_run
from .config import ExperimentConfig, load_experiment_config
from .dataset import build_case_set, case_set_profile
from .io import write_jsonl
from .preflight import run_preflight
from .records import Case
from .runner import execute_generation
from .status import summarize_run
from .validation import assert_pilot_gate, validate_cases, validate_run

app = typer.Typer(no_args_is_help=True, add_completion=False)
REPO = Path(__file__).resolve().parents[2]


def _cases_for_config(
    config_path: Path,
) -> tuple[ExperimentConfig, list[Case]]:
    config = load_experiment_config(config_path)
    cases = build_case_set(REPO, config)
    selected_path = REPO / "data" / "processed" / f"cases_{config.experiment_name}.jsonl"
    write_jsonl(selected_path, cases)
    return config, cases


@app.command()
def prepare(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
) -> None:
    """Build and validate frozen HMDA policy-sandbox cases."""
    loaded, cases = _cases_for_config(config)
    validation = validate_cases(repo=REPO, config=loaded, cases=cases)
    result = {
        "experiment": loaded.experiment_name,
        **case_set_profile(cases),
        "source_sha256": loaded.hmda_source_sha256,
        "historical_action_used_as_gold": False,
        "post_decision_fields_supplied_to_models": False,
        "validation_pass": validation["pass"],
        "validation_issues": validation["issues"],
    }
    typer.echo(json.dumps(result, indent=2))
    if not validation["pass"]:
        raise typer.Exit(code=1)


def _execute(config_path: Path) -> None:
    loaded, cases = _cases_for_config(config_path)
    result = asyncio.run(execute_generation(repo=REPO, config=loaded, cases=cases))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def pilot(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "pilot.yaml"
    ),
) -> None:
    """Run the resumable architecture-by-budget pilot."""
    _execute(config)


@app.command(name="run")
def run_main(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
    force: Annotated[
        bool,
        typer.Option(help="Override the pilot gate with an audited reason."),
    ] = False,
    force_reason: Annotated[
        str | None,
        typer.Option(help="Required audit note when --force is used."),
    ] = None,
) -> None:
    """Run the gated main study."""
    loaded = load_experiment_config(config)
    if force:
        if not force_reason or not force_reason.strip():
            raise typer.BadParameter("--force requires --force-reason")
        override = (
            REPO / "experiments" / "runs" / loaded.experiment_name / "pilot_gate_override.json"
        )
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text(
            json.dumps(
                {
                    "experiment": loaded.experiment_name,
                    "reason": force_reason.strip(),
                    "recorded_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        assert_pilot_gate(REPO, loaded)
    _execute(config)


@app.command()
def analyze(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
    diagnostic: Annotated[
        bool,
        typer.Option(help="Allow incomplete input and watermark all outputs as diagnostic."),
    ] = False,
) -> None:
    """Generate validated statistical tables and figures."""
    loaded, cases = _cases_for_config(config)
    typer.echo(
        json.dumps(
            analyze_run(
                repo=REPO,
                config=loaded,
                cases=cases,
                diagnostic=diagnostic,
            ),
            indent=2,
        )
    )


@app.command()
def validate(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
    require_generations: Annotated[
        bool,
        typer.Option("--require-generations/--no-require-generations"),
    ] = True,
    require_pilot_gate: Annotated[
        bool,
        typer.Option("--require-pilot-gate/--no-require-pilot-gate"),
    ] = True,
) -> None:
    """Validate source, cases, grid completeness, and token accounting."""
    loaded, cases = _cases_for_config(config)
    report = validate_run(
        repo=REPO,
        config=loaded,
        cases=cases,
        require_generations=require_generations,
        require_pilot_gate=require_pilot_gate,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["pass"]:
        raise typer.Exit(code=1)


@app.command()
def preflight(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "pilot.yaml"
    ),
) -> None:
    """Verify real completions for every configured model and credential."""
    loaded = load_experiment_config(config)
    report = asyncio.run(run_preflight(repo=REPO, config=loaded))
    typer.echo(json.dumps(report, indent=2))
    if not report["pass"]:
        raise typer.Exit(code=1)


@app.command()
def calibrate(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "calibration.yaml"
    ),
) -> None:
    """Recommend four feasible budgets from calibration trajectories."""
    loaded = load_experiment_config(config)
    typer.echo(json.dumps(calibrate_run(repo=REPO, config=loaded), indent=2))


@app.command()
def status(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "pilot.yaml"
    ),
) -> None:
    """Summarize coverage and operational failures."""
    loaded, cases = _cases_for_config(config)
    expected = len(cases) * len(loaded.systems) * len(loaded.token_budgets) * loaded.repetitions
    typer.echo(
        json.dumps(
            summarize_run(repo=REPO, config=loaded, expected_cells=expected),
            indent=2,
        )
    )
