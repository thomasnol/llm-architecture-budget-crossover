from budget_crossover.config import ExperimentConfig
from budget_crossover.records import Case, Generation
from budget_crossover.validation import generation_grid_status


def _case() -> Case:
    return Case(
        case_id="case",
        pair_id="pair",
        counterfactual_variant="observed",
        source_row_id="source",
        state="DC",
        historical_action="originated",
        policy_decision="approve",
        policy_reason_codes=["meets_policy"],
        documents={
            "application": "application",
            "collateral": "collateral",
            "credit": "credit",
            "quality_control": "quality",
            "compliance_monitoring": "monitoring",
        },
        protected_attributes={"race": "White"},
        changed_protected_attribute="race",
        complexity="routine",
    )


def test_grid_validation_includes_repetitions_in_cell_identity():
    config = ExperimentConfig(
        experiment_name="grid",
        hmda_source_sha256="0" * 64,
        token_budgets=[4096],
        systems=["monolith", "adaptive"],
        repetitions=2,
        bootstrap_replicates=100,
    )
    generations = [
        Generation(
            run_id=f"{system}-{repetition}",
            case_id="case",
            pair_id="pair",
            counterfactual_variant="observed",
            system=system,
            token_budget=4096,
            repetition=repetition,
            model=config.generator_model,
        )
        for system in config.systems
        for repetition in range(config.repetitions)
    ]

    complete = generation_grid_status([_case()], generations, config)
    missing = generation_grid_status([_case()], generations[:-1], config)

    assert complete == {
        "expected": 4,
        "observed": 4,
        "missing": 0,
        "extra": 0,
        "duplicates": 0,
        "complete": True,
    }
    assert missing["missing"] == 1
    assert missing["complete"] is False
