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
from .v2_analysis import analyze_v2
from .v2_config import V2Config, load_v2_config
from .v2_dataset import (
    build_insurance_v2_cases,
    fetch_mmlu_pro,
    sample_mmlu_pro,
    stratified_sample_cases,
)
from .v2_judging import execute_v2_judging
from .v2_models import V2Case
from .v2_runner import execute_v2_generation

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


def _v2_cases_for_config(config_path: Path) -> tuple[V2Config, list[V2Case]]:
    config = load_v2_config(config_path)
    insurance = build_insurance_v2_cases(
        REPO / "data" / "raw" / "train.parquet",
        evidence_condition=config.evidence_condition,
        fixed_source_model=config.fixed_source_model,
        max_context_chars=config.max_context_chars,
    )
    insurance = stratified_sample_cases(
        insurance,
        sample_size=config.insurance_sample_size,
        seed=config.seed,
    )
    cases = list(insurance)
    if config.include_mmlu:
        raw_mmlu = fetch_mmlu_pro(REPO / "data" / "raw" / "mmlu_pro_test.parquet")
        cases.extend(
            sample_mmlu_pro(
                raw_mmlu,
                sample_size=config.mmlu_sample_size,
                seed=config.seed + 17,
            )
        )
    selected_path = (
        REPO
        / "data"
        / "processed"
        / f"cases_{config.experiment_name}.jsonl"
    )
    write_jsonl(selected_path, cases)
    return config, sorted(cases, key=lambda case: case.case_id)


@app.command("prepare-v2")
def prepare_v2(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "v2_main.yaml"
    ),
) -> None:
    """Prepare unbiased structured-label cases for the Version 2 study."""
    loaded, cases = _v2_cases_for_config(config)
    typer.echo(
        json.dumps(
            {
                "experiment": loaded.experiment_name,
                "cases": len(cases),
                "datasets": {
                    dataset: sum(case.dataset == dataset for case in cases)
                    for dataset in sorted({case.dataset for case in cases})
                },
                "tasks": {
                    task: sum(case.task == task for case in cases)
                    for task in sorted({case.task for case in cases})
                },
                "schema_parse_failures": 0,
                "evidence_selection_uses_historical_correctness": False,
            },
            indent=2,
        )
    )


def _run_v2(config_path: Path) -> None:
    loaded, cases = _v2_cases_for_config(config_path)
    result = asyncio.run(
        execute_v2_generation(repo=REPO, config=loaded, cases=cases)
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("pilot-v2")
def pilot_v2(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "v2_pilot.yaml"
    ),
) -> None:
    """Run the resumable Version 2 manipulation-check pilot."""
    _run_v2(config)


@app.command("run-v2")
def run_v2(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "v2_main.yaml"
    ),
) -> None:
    """Run the gated Version 2 main sweep."""
    _run_v2(config)


@app.command("judge-v2")
def judge_v2(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "v2_main.yaml"
    ),
) -> None:
    """Run secondary cross-family judges; structured exact scoring stays primary."""
    loaded, cases = _v2_cases_for_config(config)
    result = asyncio.run(
        execute_v2_judging(repo=REPO, config=loaded, cases=cases)
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("analyze-v2")
def analyze_v2_command(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = (
        REPO / "configs" / "v2_main.yaml"
    ),
) -> None:
    """Analyze accuracy, mechanisms, power, judge coverage, and Pareto efficiency."""
    loaded, cases = _v2_cases_for_config(config)
    result = analyze_v2(repo=REPO, config=loaded, cases=cases)
    typer.echo(json.dumps(result, indent=2))
