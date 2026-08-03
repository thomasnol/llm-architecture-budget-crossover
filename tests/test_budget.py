import pytest

from budget_crossover.budget import (
    BUDGET_TIERS,
    BudgetExceeded,
    BudgetLedger,
    BudgetTier,
    ProtocolViolation,
)
from budget_crossover.models import Usage


def test_initial_budget_tiers_have_the_frozen_prespecified_limits():
    assert BUDGET_TIERS == {
        "low": BudgetTier(
            name="low",
            token_limit=4096,
            retrieval_limit=2,
            planned_query_limit=1,
            candidate_limit=1,
            repair_limit=0,
        ),
        "middle": BudgetTier(
            name="middle",
            token_limit=12288,
            retrieval_limit=6,
            planned_query_limit=2,
            candidate_limit=2,
            repair_limit=1,
        ),
        "high": BudgetTier(
            name="high",
            token_limit=32768,
            retrieval_limit=12,
            planned_query_limit=4,
            candidate_limit=4,
            repair_limit=1,
        ),
    }


def test_ledger_reserves_exact_prompt_and_output_then_releases_unused_tokens_on_commit():
    ledger = BudgetLedger(BUDGET_TIERS["low"])

    reservation = ledger.authorize(stage="candidate", prompt_tokens=1000, max_output_tokens=256)

    assert reservation.reserved_tokens == 1256
    assert ledger.reserved_tokens == 1256
    assert ledger.available_tokens == 2840

    event = ledger.commit(
        reservation,
        Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
    )

    assert event.usage.authoritative_total == 1100
    assert event.protocol_violation is False
    assert ledger.spent_tokens == 1100
    assert ledger.reserved_tokens == 0
    assert ledger.available_tokens == 2996
    assert ledger.events == (event,)


def test_ledger_refuses_unaffordable_calls_including_active_reservations_and_exhaustion():
    ledger = BudgetLedger(BUDGET_TIERS["low"])
    reservation = ledger.authorize(stage="candidate", prompt_tokens=4000, max_output_tokens=96)

    with pytest.raises(BudgetExceeded):
        ledger.authorize(stage="repair", prompt_tokens=0, max_output_tokens=1)

    ledger.commit(
        reservation,
        Usage(prompt_tokens=4000, completion_tokens=96, total_tokens=4096),
    )
    assert ledger.available_tokens == 0
    with pytest.raises(BudgetExceeded):
        ledger.authorize(stage="repair", prompt_tokens=0, max_output_tokens=1)


def test_ledger_rejects_missing_authoritative_usage_without_releasing_reservation():
    ledger = BudgetLedger(BUDGET_TIERS["low"])
    reservation = ledger.authorize(stage="candidate", prompt_tokens=10, max_output_tokens=10)

    with pytest.raises(ValueError, match="authoritative"):
        ledger.commit(reservation, Usage())

    assert ledger.reserved_tokens == 20
    assert ledger.spent_tokens == 0


@pytest.mark.parametrize(
    ("usage", "expected_code"),
    [
        (Usage(prompt_tokens=101, completion_tokens=10, total_tokens=111), "prompt_token_mismatch"),
        (Usage(prompt_tokens=100, completion_tokens=11, total_tokens=111), "output_reservation_overrun"),
    ],
)
def test_ledger_rejects_over_reserved_usage_and_records_hard_cap_protocol_violations(
    usage,
    expected_code,
):
    tier = BudgetTier(
        name="test",
        token_limit=110,
        retrieval_limit=1,
        planned_query_limit=1,
        candidate_limit=1,
        repair_limit=0,
    )
    ledger = BudgetLedger(tier)
    reservation = ledger.authorize(stage="candidate", prompt_tokens=100, max_output_tokens=10)

    with pytest.raises(ProtocolViolation) as raised:
        ledger.commit(reservation, usage)

    assert expected_code in raised.value.codes
    assert "hard_cap_overrun" in raised.value.codes
    assert ledger.protocol_violations == raised.value.codes
    assert ledger.spent_tokens == 111
    assert ledger.reserved_tokens == 0
    assert ledger.available_tokens == -1
    assert ledger.events[-1].protocol_violation is True
