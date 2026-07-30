from pathlib import Path

from budget_crossover.config import ExperimentConfig
from budget_crossover.io import write_jsonl
from budget_crossover.records import FailureAttempt
from budget_crossover.runner import error_path
from budget_crossover.status import summarize_run


def test_status_groups_operational_failures_by_attributable_dimensions(
    tmp_path: Path,
):
    config = ExperimentConfig(
        experiment_name="status-test",
        hmda_source_sha256="0" * 64,
        token_budgets=[2048],
        systems=["monolith", "adaptive"],
        bootstrap_replicates=100,
        require_preflight=False,
    )
    attempts = [
        FailureAttempt(
            run_id=f"run-{index}",
            case_id=f"case-{index}",
            pair_id=f"pair-{index}",
            counterfactual_variant="observed",
            system="adaptive",
            token_budget=2048,
            attempted_model="claude-sonnet-4-6",
            stage="compliance_audit",
            credential_slot=1,
            status_code=400,
            request_id=f"request-{index}",
            retryable=False,
            error_type="GatewayRequestError",
            detail="unsupported parameter",
            signature="400:claude-sonnet-4-6:compliance_audit:1:unsupported",
            attempt_number=1,
        )
        for index in range(2)
    ]
    write_jsonl(error_path(tmp_path, config), attempts)

    report = summarize_run(repo=tmp_path, config=config, expected_cells=4)

    assert report["scored_cells"] == 0
    assert report["remaining_cells"] == 4
    assert report["error_attempts"] == 2
    assert report["unique_failed_cells"] == 2
    assert report["failures_by_model"] == {"claude-sonnet-4-6": 2}
    assert report["failures_by_stage"] == {"compliance_audit": 2}
    assert report["failures_by_http_status"] == {"400": 2}
    assert report["failures_by_signature"] == {
        "400:claude-sonnet-4-6:compliance_audit:1:unsupported": 2
    }
