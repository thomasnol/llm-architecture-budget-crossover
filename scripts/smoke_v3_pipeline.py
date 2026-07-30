from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from budget_crossover.io import write_jsonl
from budget_crossover.models import GatewayResponse, Usage
from budget_crossover.v3_analysis import analyze_v3
from budget_crossover.v3_config import V3Config, load_v3_config
from budget_crossover.v3_dataset import build_v3_case_set
from budget_crossover.v3_manifest import ensure_v3_manifest, v3_run_dir
from budget_crossover.v3_models import V3Case, V3Generation
from budget_crossover.v3_runner import v3_generation_path
from budget_crossover.v3_systems import run_v3_system
from budget_crossover.v3_validation import validate_v3_run

REPO = Path(__file__).resolve().parents[1]


class ScriptedGateway:
    """Deterministic, case-aware gateway used only for offline pipeline QA."""

    def __init__(self, case: V3Case) -> None:
        self.case = case
        self.calls: list[dict[str, Any]] = []

    def _decision(self) -> str:
        return json.dumps(
            {
                "decision": self.case.policy_decision,
                "reason_codes": self.case.policy_reason_codes,
                "confidence": 0.96,
                "rationale": "The supplied sandbox thresholds determine this result.",
            },
            separators=(",", ":"),
        )

    async def complete(self, **kwargs: Any) -> GatewayResponse:
        self.calls.append(kwargs)
        user = str(kwargs["user"])
        if '"request":["tool_name"' in user:
            text = json.dumps(
                {
                    "request": ["application", "collateral", "credit"],
                    "rationale": "The policy depends on income, DTI, LTV, and term.",
                },
                separators=(",", ":"),
            )
        elif "SPECIALIST REPORTS" in user:
            text = self._decision()
        elif '"accept":true_or_false' in user:
            text = (
                '{"accept":true,"policy_errors":[],"prohibited_field_use":false,'
                '"required_correction":"none"}'
            )
        elif '"prohibited_for_decision"' in user:
            text = (
                '{"prohibited_for_decision":["race","sex","ethnicity","age_band"],'
                '"data_quality_flags":[],"instruction":"ignore monitoring fields"}'
            )
        elif '"recommended_decision"' in user:
            text = json.dumps(
                {
                    "recommended_decision": self.case.policy_decision,
                    "reason_codes": self.case.policy_reason_codes,
                    "material_facts": ["sandbox policy applied"],
                },
                separators=(",", ":"),
            )
        else:
            text = self._decision()
        prompt_tokens = math.ceil(
            (len(str(kwargs["system"])) + len(user)) / 4.0
        ) + 8
        completion_tokens = math.ceil(len(text) / 4.0)
        await asyncio.sleep(0)
        return GatewayResponse(
            text=text,
            model=str(kwargs["model"]),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_seconds=0.002,
            credential_slot=(
                1 if str(kwargs["model"]).startswith("claude") else 2
            ),
            request_id=f"offline-{len(self.calls)}",
            raw_finish_reason="stop",
        )


async def _generate(config: V3Config, cases: list[V3Case]) -> list[V3Generation]:
    rows: list[V3Generation] = []
    for case in cases:
        for system in config.systems:
            for budget in config.token_budgets:
                started = time.monotonic()
                result = await run_v3_system(
                    ScriptedGateway(case),
                    case=case,
                    system=system,
                    token_budget=budget,
                    config=config,
                    run_id=f"{config.experiment_name}-{case.case_id}-{system}-b{budget}",
                )
                result.wall_time_seconds = time.monotonic() - started
                rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise the complete v3 orchestration and analysis pipeline offline."
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Refuse to replace an existing ignored smoke-run directory.",
    )
    args = parser.parse_args()

    main_config = load_v3_config(REPO / "configs" / "v3_main.yaml")
    config = main_config.model_copy(
        update={
            "experiment_name": "v3_offline_smoke",
            "execution_mode": "offline_smoke",
            "base_application_count": 4,
            "exclude_pilot_applications": False,
            "bootstrap_replicates": 250,
            "runtime_hours": 1.0,
        }
    )
    run_dir = v3_run_dir(REPO, config)
    if run_dir.exists():
        if args.keep_existing:
            raise SystemExit(f"smoke run already exists: {run_dir}")
        shutil.rmtree(run_dir)

    cases = build_v3_case_set(REPO, config)
    ensure_v3_manifest(REPO, config, cases)
    generations = asyncio.run(_generate(config, cases))
    write_jsonl(v3_generation_path(REPO, config), generations)
    report = analyze_v3(repo=REPO, config=config, cases=cases)
    validation = validate_v3_run(
        repo=REPO,
        config=config,
        cases=cases,
        require_generations=True,
        require_pilot_gate=False,
    )
    report["offline_scripted_gateway"] = True
    report["scored_model_evidence"] = False
    report["offline_validation_pass"] = validation["pass"]
    (run_dir / "analysis" / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    if not validation["pass"]:
        raise SystemExit(
            "offline smoke validation failed: "
            + "; ".join(validation["issues"])
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
