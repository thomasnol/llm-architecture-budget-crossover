from __future__ import annotations

import json
from pathlib import Path

from budget_crossover.workflow import empirical_claim_status, run_offline_fixture

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    result = run_offline_fixture(REPO)
    status = empirical_claim_status(Path(result["run_dir"]))
    print(
        json.dumps(
            {
                "status": "NON_EMPIRICAL_OFFLINE_FIXTURE",
                "stages": result["stages"],
                "validated_empirical_results": status["allowed"],
                "claim_block_reasons": status["reasons"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if status["allowed"]:
        raise RuntimeError("offline scripted fixture must never enable empirical claims")


if __name__ == "__main__":
    main()
