from pathlib import Path

from budget_crossover.calibration import recommend_budgets


def test_recommended_budgets_use_realized_quantiles_and_feasibility_floor(
    tmp_path: Path,
):
    trajectories = [
        {"system": "monolith", "total_tokens": 3000},
        {"system": "retrieval", "total_tokens": 4000},
        {"system": "committee", "total_tokens": 5000},
        {"system": "guardrail", "total_tokens": 6000},
        {"system": "adaptive", "total_tokens": 7000},
    ]

    report = recommend_budgets(
        trajectories=trajectories,
        architecture_minima={
            "monolith": 1200,
            "retrieval": 2400,
            "committee": 3500,
            "guardrail": 3200,
            "adaptive": 1800,
        },
        round_to=256,
    )

    assert report["feasibility_floor"] == 3500
    assert report["recommended_budgets"] == [4096, 5120, 6144, 6912]
    assert len(set(report["recommended_budgets"])) == 4
    assert report["recommended_budgets"] == sorted(report["recommended_budgets"])
