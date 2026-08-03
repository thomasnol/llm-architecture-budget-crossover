from decimal import Decimal

import pytest

from budget_crossover.analysis import (
    MaskedPilotDiscordance,
    MatchedBlock,
    PairedCaseOutcome,
    SystemCaseMetric,
    analyze_run,
    cluster_bootstrap_crossover,
    confirm_crossover,
    exact_mcnemar_power,
    exact_mcnemar_test,
    holm_adjust,
    minimum_paired_sample_size,
    pareto_dominance_probabilities,
    score_itt_results,
    size_internal_pilot,
)
from budget_crossover.models import (
    AnswerSpec,
    Candidate,
    CellResult,
    HiddenLabel,
    MechanismTrace,
    PublicCase,
)


def test_exact_one_sided_mcnemar_uses_the_conditional_binomial_tail():
    low = exact_mcnemar_test(improved=0, regressed=6, alternative="less")
    high = exact_mcnemar_test(improved=6, regressed=0, alternative="greater")
    tie = exact_mcnemar_test(improved=0, regressed=0, alternative="greater")

    assert low.p_value == pytest.approx(0.015625)
    assert low.reject is True
    assert high.p_value == pytest.approx(0.015625)
    assert high.reject is True
    assert tie.p_value == 1.0
    assert tie.reject is False
    assert high.convention == "conditional_binomial_discordant_pairs"


def test_exact_mcnemar_power_integrates_over_random_discordant_counts():
    # With q=.05 and a five-point alternative, every discordant pair favors the
    # alternative. At N=5 rejection occurs only when all five pairs discord,
    # hence power = .05**5 = 0.0000003125.
    power = exact_mcnemar_power(
        independent_pairs=5,
        discordance_rate=0.05,
        alternative_difference=0.05,
    )

    assert power == pytest.approx(0.0000003125, rel=1e-12)


def test_exact_sample_size_lookup_returns_the_first_n_reaching_ninety_percent():
    sizing = minimum_paired_sample_size(discordance_rate=0.05)

    assert sizing.required_n == 158
    assert sizing.achieved_power == pytest.approx(0.9004240718552209)
    assert sizing.previous_power == pytest.approx(0.8974508517633653)
    assert sizing.target_power == 0.90


@pytest.mark.parametrize(
    ("discordant", "required_n", "hard_n", "easy_n", "stop"),
    [
        (25, 886, 900, 100, False),
        (26, 921, 921, 79, False),
        (30, 1060, None, None, True),
    ],
)
def test_blinded_internal_pilot_allocates_or_stops_without_unblinding(
    discordant: int,
    required_n: int,
    hard_n: int | None,
    easy_n: int | None,
    stop: bool,
):
    masked = MaskedPilotDiscordance(
        independent_documents=100,
        low_discordant=discordant,
        high_discordant=discordant,
        repetitions=5,
    )

    result = size_internal_pilot(masked)

    assert result.required_hard_n == required_n
    assert result.allocated_hard_n == hard_n
    assert result.allocated_easy_n == easy_n
    assert result.underpowered_stop is stop
    assert result.architecture_identity_available is False
    assert result.discordance_direction_available is False
    assert result.unblinded is False
    assert result.independent_n_used == 100
    assert result.repetitions_increase_n is False


def _public(case_id: str, document_id: str) -> PublicCase:
    return PublicCase(
        case_id=case_id,
        dataset="finqa",
        document_id=document_id,
        question="What is the value?",
        evidence=(),
        stratum="headroom",
    )


def _label(case_id: str) -> HiddenLabel:
    return HiddenLabel(
        case_id=case_id,
        answer=AnswerSpec(value=Decimal(10), unit=None, entity=None, period=None),
        gold_derivation="10",
        gold_support_ids=(),
        source_lineage=("snapshot", "doc", case_id),
    )


def _cell(
    case_id: str,
    system: str,
    repetition: int,
    *,
    value: str | None,
) -> CellResult:
    return CellResult(
        case_id=case_id,
        system=system,
        tier="low",
        repetition=repetition,
        status=("ok" if value is not None else "architecture_failure"),
        candidate=(
            Candidate(
                value=value,
                unit=None,
                entity=None,
                period=None,
                expression=None,
                citations=(),
            )
            if value is not None
            else None
        ),
        trace=MechanismTrace(
            exit_reason=("completed" if value is not None else "architecture_error")
        ),
    )


def test_itt_exact_scoring_excludes_whole_unresolved_external_blocks_and_not_repetitions():
    cases = (
        _public("kept", "doc-kept"),
        _public("external-a", "doc-a"),
        _public("external-b", "doc-b"),
    )
    labels = tuple(_label(case.case_id) for case in cases)
    results = tuple(
        _cell(case.case_id, system, repetition, value=("10" if system == "verified_search" else None))
        for case in cases
        for repetition in range(2)
        for system in ("monolith", "verified_search")
    )
    blocks = (
        MatchedBlock(
            block_id="external-block",
            case_ids=("external-a", "external-b"),
            external=True,
            resolved=False,
        ),
    )

    scored = score_itt_results(cases, labels, results, matched_blocks=blocks)

    assert scored.excluded_case_ids == ("external-a", "external-b")
    assert scored.excluded_matched_blocks == ("external-block",)
    assert scored.independent_documents == 1
    assert scored.scored_cells == 4
    assert scored.primary_cells == 2
    assert scored.repetitions_increase_n is False
    kept = {(row.system, row.repetition): row.correct for row in scored.outcomes}
    assert kept == {
        ("monolith", 0): False,
        ("monolith", 1): False,
        ("verified_search", 0): True,
        ("verified_search", 1): True,
    }


def _endpoint_fixture(*, low_pattern: str) -> tuple[PairedCaseOutcome, ...]:
    rows = []
    for index in range(6):
        low = (
            (True, False)
            if low_pattern == "verified_worse"
            else (True, True)
        )
        rows.extend(
            [
                PairedCaseOutcome(
                    document_id=f"doc-{index}",
                    tier="low",
                    monolith_correct=low[0],
                    verified_search_correct=low[1],
                ),
                PairedCaseOutcome(
                    document_id=f"doc-{index}",
                    tier="high",
                    monolith_correct=False,
                    verified_search_correct=True,
                ),
            ]
        )
    return tuple(rows)


def test_primary_confirmation_requires_both_directional_exact_endpoint_tests():
    confirmed = confirm_crossover(
        _endpoint_fixture(low_pattern="verified_worse"),
        bootstrap_replicates=200,
        seed=17,
    )
    threshold = confirm_crossover(
        _endpoint_fixture(low_pattern="equal"),
        bootstrap_replicates=200,
        seed=17,
    )

    assert confirmed.label == "strict_crossover_confirmed"
    assert confirmed.confirmed is True
    assert confirmed.endpoint_reversal is True
    assert confirmed.low.difference == -1.0
    assert confirmed.low.exact.p_value == pytest.approx(0.015625)
    assert confirmed.low.one_sided_bound == -1.0
    assert confirmed.high.difference == 1.0
    assert confirmed.high.exact.p_value == pytest.approx(0.015625)
    assert confirmed.high.two_sided_interval == (1.0, 1.0)
    assert confirmed.low.sesoi_interpretation == "five_point_margin_supported"
    assert confirmed.sesoi_is_design_alternative_not_automatic_margin is True

    assert threshold.label == "threshold_benefit_only"
    assert threshold.confirmed is False
    assert threshold.endpoint_reversal is False
    assert threshold.low.difference == 0.0
    assert threshold.low.exact.reject is False
    assert threshold.high.exact.reject is True


def test_five_point_design_alternative_is_not_automatically_a_proven_margin():
    outcomes = tuple(
        PairedCaseOutcome(
            document_id=f"doc-{index}",
            tier=tier,
            monolith_correct=(tier == "low" or index >= 2),
            verified_search_correct=(tier == "high"),
        )
        for index in range(20)
        for tier in ("low", "high")
    )

    result = confirm_crossover(outcomes, bootstrap_replicates=1000, seed=23)

    assert result.high.difference == pytest.approx(0.10)
    assert result.high.one_sided_bound < 0.05
    assert (
        result.high.sesoi_interpretation
        == "point_meets_five_point_sesoi_margin_not_proven"
    )
    assert result.sesoi_is_design_alternative_not_automatic_margin is True


def test_cluster_bootstrap_retains_non_crossing_mass_and_equality_never_crosses():
    mixed = tuple(
        PairedCaseOutcome(
            document_id=document_id,
            tier=tier,
            monolith_correct=monolith,
            verified_search_correct=verified,
        )
        for document_id, pattern in {
            "crossing": {
                "low": (True, False),
                "middle": (True, False),
                "high": (False, True),
            },
            "tied": {
                "low": (True, True),
                "middle": (True, True),
                "high": (True, True),
            },
        }.items()
        for tier, (monolith, verified) in pattern.items()
    )

    bootstrap = cluster_bootstrap_crossover(
        mixed,
        tier_values={"low": 4096, "middle": 12288, "high": 32768},
        bootstrap_replicates=100,
        seed=7,
    )
    equality = cluster_bootstrap_crossover(
        tuple(row for row in mixed if row.document_id == "tied"),
        tier_values={"low": 4096, "middle": 12288, "high": 32768},
        bootstrap_replicates=20,
        seed=7,
    )

    assert bootstrap.endpoint_reversal is True
    assert bootstrap.transition_estimated is True
    assert bootstrap.transition_value == 22528.0
    assert bootstrap.crossing_replicates + bootstrap.non_crossing_replicates == 100
    assert bootstrap.non_crossing_replicates > 0
    assert bootstrap.crossing_support < 1.0
    assert bootstrap.conditional_crossing_interval == (22528.0, 22528.0)
    assert bootstrap.confidence_set.includes_no_crossing is True

    assert equality.endpoint_reversal is False
    assert equality.transition_estimated is False
    assert equality.transition_value is None
    assert equality.crossing_support == 0.0
    assert equality.non_crossing_replicates == 20
    assert equality.conditional_crossing_interval is None
    assert equality.confidence_set.includes_no_crossing is True


def test_exploratory_family_uses_holm_adjustment():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})

    assert adjusted.method == "holm"
    assert adjusted.adjusted_p_values == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )


def test_pareto_dominance_claims_are_separate_for_each_realized_resource():
    metrics = tuple(
        SystemCaseMetric(
            document_id=f"doc-{document}",
            system=system,
            tier="high",
            correct=(system == "verified_search"),
            realized_tokens=(80 if system == "verified_search" else 100),
            realized_cost=(2.0 if system == "verified_search" else 1.0),
            realized_latency=(0.8 if system == "verified_search" else 1.0),
        )
        for document in range(2)
        for system in ("monolith", "verified_search")
    )

    analysis = pareto_dominance_probabilities(metrics, bootstrap_replicates=50, seed=11)
    verified = {
        row.resource: row.dominance_probability
        for row in analysis.comparisons
        if row.candidate == "verified_search" and row.comparator == "monolith"
    }

    assert analysis.claims_are_resource_specific is True
    assert verified == {"tokens": 1.0, "cost": 0.0, "latency": 1.0}


def test_legacy_analysis_bridge_fails_closed_with_a_migration_error():
    with pytest.raises(RuntimeError, match="legacy analyze workflow was removed"):
        analyze_run()
