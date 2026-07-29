from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from .analysis import analyze_run
from .config import load_config
from .dataset import build_cases, sample_cases
from .gateway import GatewayClient
from .io import read_jsonl, write_jsonl
from .judging import execute_judging_run
from .models import Case
from .runner import execute_generation_run

app = typer.Typer(no_args_is_help=True, add_completion=False)
REPO = Path(__file__).resolve().parents[2]


def _cases_for_config(config_path: Path) -> tuple:
    config = load_config(config_path)
    all_cases_path = REPO / "data" / "processed" / "cases.jsonl"
    if all_cases_path.exists():
        all_cases = read_jsonl(all_cases_path, Case)
    else:
        all_cases = build_cases(
            REPO / "data" / "raw" / "train.parquet",
            max_context_chars=config.max_context_chars,
        )
        write_jsonl(all_cases_path, all_cases)
    selected = sample_cases(
        all_cases,
        sample_size=config.sample_size,
        seed=config.seed,
        task_quotas=config.task_quotas,
    )
    sample_path = REPO / "data" / "processed" / f"sample_{config.experiment_name}.jsonl"
    write_jsonl(sample_path, selected)
    return config, selected


@app.command()
def prepare(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
) -> None:
    """Build leakage-controlled cases and the deterministic study sample."""
    loaded, selected = _cases_for_config(config)
    typer.echo(
        json.dumps(
            {
                "experiment": loaded.experiment_name,
                "sample_size": len(selected),
                "tasks": {
                    task: sum(case.task == task for case in selected)
                    for task in sorted({case.task for case in selected})
                },
                "max_evidence_chars": max(case.evidence_chars for case in selected),
            },
            indent=2,
        )
    )


def _run(config_path: Path) -> None:
    loaded, cases = _cases_for_config(config_path)
    result = asyncio.run(execute_generation_run(repo=REPO, config=loaded, cases=cases))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def pilot(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "pilot.yaml"
    ),
) -> None:
    """Run the resumable pilot generation sweep."""
    _run(config)


@app.command(name="run")
def main_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
) -> None:
    """Run the resumable main generation sweep."""
    _run(config)


@app.command()
def judge(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
) -> None:
    """Run two blinded judges and adjudicate their disagreements."""
    loaded, cases = _cases_for_config(config)
    result = asyncio.run(execute_judging_run(repo=REPO, config=loaded, cases=cases))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def analyze(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "main.yaml"
    ),
) -> None:
    """Compute exact scores, paired uncertainty, crossover estimates, and figures."""
    loaded, cases = _cases_for_config(config)
    result = analyze_run(repo=REPO, config=loaded, cases=cases)
    typer.echo(json.dumps(result, indent=2))


@app.command("gateway-check")
def gateway_check() -> None:
    """Verify credentials and list gateway model deployment identifiers."""

    async def check() -> dict:
        client = GatewayClient()
        try:
            return await client.list_models()
        finally:
            await client.close()

    typer.echo(json.dumps(asyncio.run(check()), indent=2))
