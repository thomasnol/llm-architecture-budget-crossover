from __future__ import annotations

import json
from pathlib import Path

from budget_crossover.config import ExperimentConfig, load_experiment_config
from budget_crossover.workflow import generate_paper_results

REPO = Path(__file__).resolve().parents[1]


def build(*, repo: Path, config: ExperimentConfig) -> dict[str, object]:
    return generate_paper_results(repo=repo, config=config)


def main() -> None:
    result = build(
        repo=REPO,
        config=load_experiment_config(REPO / "configs" / "main.yaml"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
