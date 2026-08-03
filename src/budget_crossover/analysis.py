from __future__ import annotations

"""Exact paired inference and joint case-cluster resampling."""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binom

from .models import CellResult, FrozenModel, HiddenLabel, PublicCase, SystemName, TierName
from .scoring import score_candidate

Alternative = Literal["less", "greater"]


class McNemarResult(FrozenModel):
    improved: int
    regressed: int
    discordant: int
    alternative: Alternative
    alpha: float
    p_value: float
    reject: bool
    convention: str = "conditional_binomial_discordant_pairs"


class SampleSizeResult(FrozenModel):
    discordance_rate: float
    alternative_difference: float
    alpha: float
    target_power: float
    required_n: int
    achieved_power: float
    previous_power: float


class MaskedPilotDiscordance(FrozenModel):
    """Direction-free low/high discordance exposed by the internal pilot."""

    independent_documents: int = Field(gt=0)
    low_discordant: int = Field(ge=0)
    high_discordant: int = Field(ge=0)
    repetitions: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_masked_counts(self) -> MaskedPilotDiscordance:
        if max(self.low_discordant, self.high_discordant) > self.independent_documents:
            raise ValueError("discordant counts cannot exceed independent documents")
        if min(self.low_discordant, self.high_discordant) * 20 < self.independent_documents:
            raise ValueError("five-point design requires discordance rates of at least .05")
        return self


class PilotSizingResult(FrozenModel):
    blinded: bool = True
    architecture_identity_available: bool = False
    discordance_direction_available: bool = False
    unblinded: bool = False
    independent_unit: str = "case_or_document"
    independent_n_used: int
    repetitions_observed: int
    repetitions_increase_n: bool = False
    low_discordance_rate: float
    high_discordance_rate: float
    low_component: SampleSizeResult
    high_component: SampleSizeResult
    required_hard_n: int
    allocated_hard_n: int | None
    allocated_easy_n: int | None
    allocated_total_n: int | None
    underpowered_stop: bool
    stop_reason: str | None


class MatchedBlock(FrozenModel):
    block_id: str
    case_ids: tuple[str, ...]
    external: bool
    resolved: bool

    @model_validator(mode="after")
    def validate_cases(self) -> MatchedBlock:
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("matched-block case IDs must be nonempty and unique")
        return self


class ITTOutcome(FrozenModel):
    case_id: str
    document_id: str
    system: SystemName
    tier: TierName
    repetition: int
    primary: bool
    correct: bool
    status: str


class ITTScoringResult(FrozenModel):
    independent_unit: str = "document_or_case"
    independent_documents: int
    scored_cells: int
    primary_cells: int
    repetitions_increase_n: bool = False
    excluded_case_ids: tuple[str, ...]
    excluded_matched_blocks: tuple[str, ...]
    outcomes: tuple[ITTOutcome, ...]


class PairedCaseOutcome(FrozenModel):
    document_id: str
    tier: TierName
    monolith_correct: bool
    verified_search_correct: bool


class EndpointEffect(FrozenModel):
    tier: Literal["low", "high"]
    independent_documents: int
    difference: float
    improved: int
    regressed: int
    exact: McNemarResult
    one_sided_bound: float
    one_sided_bound_type: Literal["upper", "lower"]
    two_sided_interval: tuple[float, float]
    sesoi: float
    sesoi_interpretation: str


class CrossoverConfirmation(FrozenModel):
    alpha: float
    sesoi: float
    low: EndpointEffect
    high: EndpointEffect
    endpoint_reversal: bool
    confirmed: bool
    label: str
    sesoi_is_design_alternative_not_automatic_margin: bool = True


class CrossoverConfidenceSet(FrozenModel):
    conditional_numeric_interval: tuple[float, float] | None
    includes_no_crossing: bool


class CrossoverBootstrapResult(FrozenModel):
    bootstrap_unit: str = "document_or_case"
    bootstrap_replicates: int
    observed_differences: dict[str, float]
    endpoint_reversal: bool
    transition_estimated: bool
    transition_value: float | None
    crossing_replicates: int
    non_crossing_replicates: int
    crossing_support: float
    conditional_crossing_interval: tuple[float, float] | None
    confidence_set: CrossoverConfidenceSet


class HolmAdjustment(FrozenModel):
    method: str = "holm"
    adjusted_p_values: dict[str, float]


class SystemCaseMetric(FrozenModel):
    document_id: str
    system: SystemName
    tier: TierName
    correct: bool
    realized_tokens: float = Field(ge=0)
    realized_cost: float | None = Field(default=None, ge=0)
    realized_latency: float | None = Field(default=None, ge=0)


class ParetoComparison(FrozenModel):
    tier: TierName
    candidate: SystemName
    comparator: SystemName
    resource: Literal["tokens", "cost", "latency"]
    dominance_probability: float | None
    bootstrap_replicates: int


class ParetoAnalysis(FrozenModel):
    bootstrap_unit: str = "document_or_case"
    claims_are_resource_specific: bool = True
    comparisons: tuple[ParetoComparison, ...]


def exact_mcnemar_test(
    *,
    improved: int,
    regressed: int,
    alternative: Alternative,
    alpha: float = 0.05,
) -> McNemarResult:
    """Conditional exact McNemar test on the paired discordant counts.

    ``improved`` counts comparison-correct/baseline-wrong pairs. The ``less``
    alternative therefore uses the lower tail of that count; ``greater`` uses
    the upper tail. Concordant pairs condition out of the exact test.
    """
    if type(improved) is not int or type(regressed) is not int:
        raise TypeError("discordant counts must be integers")
    if improved < 0 or regressed < 0:
        raise ValueError("discordant counts must be nonnegative")
    if alternative not in {"less", "greater"}:
        raise ValueError("alternative must be 'less' or 'greater'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    elif alternative == "greater":
        p_value = float(binom.sf(improved - 1, discordant, 0.5))
    else:
        p_value = float(binom.cdf(improved, discordant, 0.5))
    return McNemarResult(
        improved=improved,
        regressed=regressed,
        discordant=discordant,
        alternative=alternative,
        alpha=alpha,
        p_value=p_value,
        reject=p_value <= alpha,
    )


def exact_mcnemar_power(
    *,
    independent_pairs: int,
    discordance_rate: float,
    alternative_difference: float = 0.05,
    alpha: float = 0.05,
) -> float:
    """Power of either directional component under a symmetric alternative.

    The total discordant count is Binomial(N, q). Conditional on that count,
    the favorable discordant count is Binomial(D, (q + delta) / (2q)).
    """
    if type(independent_pairs) is not int or independent_pairs < 1:
        raise ValueError("independent_pairs must be a positive integer")
    if not 0 < alternative_difference <= discordance_rate <= 1:
        raise ValueError("discordance_rate must be in [alternative_difference, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    discordant = np.arange(independent_pairs + 1)
    favorable_probability = (
        discordance_rate + alternative_difference
    ) / (2 * discordance_rate)
    critical = np.asarray(binom.isf(alpha, discordant, 0.5), dtype=int) + 1
    discordant_mass = binom.pmf(discordant, independent_pairs, discordance_rate)
    conditional_rejection = binom.sf(
        critical - 1,
        discordant,
        favorable_probability,
    )
    return float(np.dot(discordant_mass, conditional_rejection))


@lru_cache(maxsize=256)
def minimum_paired_sample_size(
    *,
    discordance_rate: float,
    alternative_difference: float = 0.05,
    alpha: float = 0.05,
    target_power: float = 0.90,
    max_n: int = 10_000,
) -> SampleSizeResult:
    """Find the first independent-pair count with target component power."""
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between zero and one")
    if type(max_n) is not int or max_n < 1:
        raise ValueError("max_n must be a positive integer")
    previous_power = 0.0
    for independent_pairs in range(1, max_n + 1):
        power = exact_mcnemar_power(
            independent_pairs=independent_pairs,
            discordance_rate=discordance_rate,
            alternative_difference=alternative_difference,
            alpha=alpha,
        )
        if power >= target_power:
            return SampleSizeResult(
                discordance_rate=discordance_rate,
                alternative_difference=alternative_difference,
                alpha=alpha,
                target_power=target_power,
                required_n=independent_pairs,
                achieved_power=power,
                previous_power=previous_power,
            )
        previous_power = power
    raise RuntimeError(f"target power was not reached by max_n={max_n}")


def size_internal_pilot(masked: MaskedPilotDiscordance) -> PilotSizingResult:
    """Size the main study without exposing architecture identity or direction."""
    low_rate = masked.low_discordant / masked.independent_documents
    high_rate = masked.high_discordant / masked.independent_documents
    low_component = minimum_paired_sample_size(discordance_rate=low_rate)
    high_component = minimum_paired_sample_size(discordance_rate=high_rate)
    required_hard_n = max(low_component.required_n, high_component.required_n)
    if required_hard_n <= 900:
        hard_n: int | None = 900
        easy_n: int | None = 100
    elif required_hard_n <= 1000:
        hard_n = required_hard_n
        easy_n = 1000 - required_hard_n
    else:
        hard_n = None
        easy_n = None
    underpowered_stop = required_hard_n > 1000
    return PilotSizingResult(
        independent_n_used=masked.independent_documents,
        repetitions_observed=masked.repetitions,
        low_discordance_rate=low_rate,
        high_discordance_rate=high_rate,
        low_component=low_component,
        high_component=high_component,
        required_hard_n=required_hard_n,
        allocated_hard_n=hard_n,
        allocated_easy_n=easy_n,
        allocated_total_n=(
            hard_n + easy_n
            if hard_n is not None and easy_n is not None
            else None
        ),
        underpowered_stop=underpowered_stop,
        stop_reason=("required_hard_n_exceeds_1000" if underpowered_stop else None),
    )


def score_itt_results(
    cases: Sequence[PublicCase],
    labels: Sequence[HiddenLabel],
    results: Sequence[CellResult],
    *,
    matched_blocks: Sequence[MatchedBlock] = (),
) -> ITTScoringResult:
    """Score every retained cell exactly; architecture failures are incorrect."""
    case_by_id = {case.case_id: case for case in cases}
    label_by_id = {label.case_id: label for label in labels}
    if len(case_by_id) != len(cases):
        raise ValueError("public case IDs must be unique")
    if len(label_by_id) != len(labels) or set(label_by_id) != set(case_by_id):
        raise ValueError("hidden labels must join one-to-one with public cases")
    block_case_ids = [case_id for block in matched_blocks for case_id in block.case_ids]
    if len(block_case_ids) != len(set(block_case_ids)):
        raise ValueError("a case cannot belong to more than one matched block")
    if not set(block_case_ids) <= set(case_by_id):
        raise ValueError("matched blocks contain unknown case IDs")
    excluded_blocks = tuple(
        sorted(block.block_id for block in matched_blocks if block.external and not block.resolved)
    )
    excluded_cases = tuple(
        sorted(
            case_id
            for block in matched_blocks
            if block.external and not block.resolved
            for case_id in block.case_ids
        )
    )
    excluded_set = set(excluded_cases)
    observed_keys: set[tuple[str, str, str, int]] = set()
    outcomes: list[ITTOutcome] = []
    for result in results:
        if result.case_id not in case_by_id:
            raise ValueError(f"result contains unknown case ID: {result.case_id}")
        key = (result.case_id, result.system, result.tier, result.repetition)
        if key in observed_keys:
            raise ValueError("result cells must be unique before ITT scoring")
        observed_keys.add(key)
        if result.case_id in excluded_set:
            continue
        correct = (
            result.status == "ok"
            and result.candidate is not None
            and score_candidate(result.candidate, label_by_id[result.case_id].answer).correct
        )
        outcomes.append(
            ITTOutcome(
                case_id=result.case_id,
                document_id=case_by_id[result.case_id].document_id,
                system=result.system,
                tier=result.tier,
                repetition=result.repetition,
                primary=result.repetition == 0,
                correct=correct,
                status=result.status,
            )
        )
    included_documents = {
        case.document_id for case in cases if case.case_id not in excluded_set
    }
    return ITTScoringResult(
        independent_documents=len(included_documents),
        scored_cells=len(outcomes),
        primary_cells=sum(outcome.primary for outcome in outcomes),
        excluded_case_ids=excluded_cases,
        excluded_matched_blocks=excluded_blocks,
        outcomes=tuple(outcomes),
    )


def _sesoi_interpretation(
    *,
    difference: float,
    one_sided_bound: float,
    direction: Alternative,
    direction_rejected: bool,
    sesoi: float,
) -> str:
    margin_supported = (
        one_sided_bound >= sesoi
        if direction == "greater"
        else one_sided_bound <= -sesoi
    )
    point_meets = difference >= sesoi if direction == "greater" else difference <= -sesoi
    if margin_supported:
        return "five_point_margin_supported"
    if point_meets:
        return "point_meets_five_point_sesoi_margin_not_proven"
    if direction_rejected:
        return "direction_supported_below_five_point_sesoi"
    return "direction_not_established"


def confirm_crossover(
    outcomes: Sequence[PairedCaseOutcome],
    *,
    alpha: float = 0.05,
    sesoi: float = 0.05,
    bootstrap_replicates: int = 5000,
    seed: int = 20260803,
) -> CrossoverConfirmation:
    """Apply the prespecified intersection-union endpoint confirmation."""
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not 0 < sesoi < 1:
        raise ValueError("sesoi must be between zero and one")
    keyed: dict[tuple[str, str], PairedCaseOutcome] = {}
    for outcome in outcomes:
        if outcome.tier not in {"low", "high"}:
            continue
        key = (outcome.document_id, outcome.tier)
        if key in keyed:
            raise ValueError("primary paired outcomes must be unique by document and tier")
        keyed[key] = outcome
    low_ids = {document_id for document_id, tier in keyed if tier == "low"}
    high_ids = {document_id for document_id, tier in keyed if tier == "high"}
    if not low_ids or low_ids != high_ids:
        raise ValueError("every independent document requires both low and high outcomes")
    document_ids = sorted(low_ids)
    difference_matrix = np.asarray(
        [
            [
                int(keyed[(document_id, tier)].verified_search_correct)
                - int(keyed[(document_id, tier)].monolith_correct)
                for tier in ("low", "high")
            ]
            for document_id in document_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0,
        len(document_ids),
        size=(bootstrap_replicates, len(document_ids)),
    )
    bootstrap_means = difference_matrix[indexes].mean(axis=1)

    endpoint_effects: dict[str, EndpointEffect] = {}
    for column, (tier, alternative, bound_quantile, bound_type) in enumerate(
        (
            ("low", "less", 0.95, "upper"),
            ("high", "greater", 0.05, "lower"),
        )
    ):
        values = difference_matrix[:, column]
        improved = int(np.count_nonzero(values == 1))
        regressed = int(np.count_nonzero(values == -1))
        exact = exact_mcnemar_test(
            improved=improved,
            regressed=regressed,
            alternative=alternative,
            alpha=alpha,
        )
        two_sided_interval = tuple(
            float(value)
            for value in np.quantile(bootstrap_means[:, column], [0.025, 0.975])
        )
        one_sided_bound = float(
            np.quantile(bootstrap_means[:, column], bound_quantile)
        )
        difference = float(values.mean())
        endpoint_effects[tier] = EndpointEffect(
            tier=tier,
            independent_documents=len(document_ids),
            difference=difference,
            improved=improved,
            regressed=regressed,
            exact=exact,
            one_sided_bound=one_sided_bound,
            one_sided_bound_type=bound_type,
            two_sided_interval=two_sided_interval,
            sesoi=sesoi,
            sesoi_interpretation=_sesoi_interpretation(
                difference=difference,
                one_sided_bound=one_sided_bound,
                direction=alternative,
                direction_rejected=exact.reject,
                sesoi=sesoi,
            ),
        )
    low = endpoint_effects["low"]
    high = endpoint_effects["high"]
    endpoint_reversal = low.difference < 0 < high.difference
    confirmed = endpoint_reversal and low.exact.reject and high.exact.reject
    if confirmed:
        label = "strict_crossover_confirmed"
    elif high.difference > 0 and high.exact.reject:
        label = "threshold_benefit_only"
    elif low.difference < 0 and low.exact.reject:
        label = "low_budget_harm_only"
    elif endpoint_reversal:
        label = "endpoint_reversal_not_confirmed"
    else:
        label = "no_directional_pattern"
    return CrossoverConfirmation(
        alpha=alpha,
        sesoi=sesoi,
        low=low,
        high=high,
        endpoint_reversal=endpoint_reversal,
        confirmed=confirmed,
        label=label,
    )


def _strict_transition(
    tier_values: Sequence[float], differences: Sequence[float]
) -> float | None:
    """Interpolate only an adjacent strict negative-to-positive transition."""
    if not differences[0] < 0 < differences[-1]:
        return None
    for left_index in range(len(differences) - 1):
        left_difference = differences[left_index]
        right_difference = differences[left_index + 1]
        if left_difference < 0 < right_difference:
            fraction = -left_difference / (right_difference - left_difference)
            return float(
                tier_values[left_index]
                + fraction * (tier_values[left_index + 1] - tier_values[left_index])
            )
    return None


def cluster_bootstrap_crossover(
    outcomes: Sequence[PairedCaseOutcome],
    *,
    tier_values: Mapping[str, float],
    bootstrap_replicates: int = 5000,
    seed: int = 20260803,
) -> CrossoverBootstrapResult:
    """Resample documents once per replicate and carry every tier together."""
    tier_order = ("low", "middle", "high")
    if set(tier_values) != set(tier_order):
        raise ValueError("tier_values must define low, middle, and high")
    numeric_tiers = tuple(float(tier_values[tier]) for tier in tier_order)
    if list(numeric_tiers) != sorted(set(numeric_tiers)):
        raise ValueError("tier values must be unique and strictly increasing")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    keyed: dict[tuple[str, str], PairedCaseOutcome] = {}
    for outcome in outcomes:
        key = (outcome.document_id, outcome.tier)
        if key in keyed:
            raise ValueError("paired outcomes must be unique by document and tier")
        keyed[key] = outcome
    document_ids = sorted({outcome.document_id for outcome in outcomes})
    if not document_ids or any(
        (document_id, tier) not in keyed
        for document_id in document_ids
        for tier in tier_order
    ):
        raise ValueError("every document requires all three tiers")
    matrix = np.asarray(
        [
            [
                int(keyed[(document_id, tier)].verified_search_correct)
                - int(keyed[(document_id, tier)].monolith_correct)
                for tier in tier_order
            ]
            for document_id in document_ids
        ],
        dtype=float,
    )
    observed = matrix.mean(axis=0)
    observed_transition = _strict_transition(numeric_tiers, observed)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0,
        len(document_ids),
        size=(bootstrap_replicates, len(document_ids)),
    )
    replicate_means = matrix[indexes].mean(axis=1)
    transitions = [
        _strict_transition(numeric_tiers, replicate.tolist())
        for replicate in replicate_means
    ]
    crossing_values = [value for value in transitions if value is not None]
    crossing_count = len(crossing_values)
    non_crossing_count = bootstrap_replicates - crossing_count
    conditional_interval = (
        tuple(
            float(value)
            for value in np.quantile(crossing_values, [0.025, 0.975])
        )
        if crossing_values
        else None
    )
    return CrossoverBootstrapResult(
        bootstrap_replicates=bootstrap_replicates,
        observed_differences={
            tier: float(observed[index]) for index, tier in enumerate(tier_order)
        },
        endpoint_reversal=bool(observed[0] < 0 < observed[-1]),
        transition_estimated=observed_transition is not None,
        transition_value=observed_transition,
        crossing_replicates=crossing_count,
        non_crossing_replicates=non_crossing_count,
        crossing_support=crossing_count / bootstrap_replicates,
        conditional_crossing_interval=conditional_interval,
        confidence_set=CrossoverConfidenceSet(
            conditional_numeric_interval=conditional_interval,
            includes_no_crossing=non_crossing_count > 0,
        ),
    )


def holm_adjust(p_values: Mapping[str, float]) -> HolmAdjustment:
    """Control an exploratory family with the step-down Holm procedure."""
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be between zero and one")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[name] = running
    return HolmAdjustment(adjusted_p_values=adjusted)


def pareto_dominance_probabilities(
    metrics: Sequence[SystemCaseMetric],
    *,
    bootstrap_replicates: int = 5000,
    seed: int = 20260803,
) -> ParetoAnalysis:
    """Estimate pairwise accuracy-resource dominance by document bootstrap."""
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    keyed: dict[tuple[str, str, str], SystemCaseMetric] = {}
    for metric in metrics:
        key = (metric.document_id, metric.system, metric.tier)
        if key in keyed:
            raise ValueError("metrics must be unique by document, system, and tier")
        keyed[key] = metric
    document_ids = sorted({metric.document_id for metric in metrics})
    combinations = sorted({(metric.tier, metric.system) for metric in metrics})
    if not document_ids or any(
        (document_id, system, tier) not in keyed
        for document_id in document_ids
        for tier, system in combinations
    ):
        raise ValueError("each document must carry every system and tier together")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0,
        len(document_ids),
        size=(bootstrap_replicates, len(document_ids)),
    )
    accuracy: dict[tuple[str, str], np.ndarray] = {}
    resources: dict[tuple[str, str, str], np.ndarray | None] = {}
    field_by_resource = {
        "tokens": "realized_tokens",
        "cost": "realized_cost",
        "latency": "realized_latency",
    }
    for tier, system in combinations:
        rows = [keyed[(document_id, system, tier)] for document_id in document_ids]
        correct = np.asarray([int(row.correct) for row in rows], dtype=float)
        accuracy[(tier, system)] = correct[indexes].mean(axis=1)
        for resource, field in field_by_resource.items():
            values = [getattr(row, field) for row in rows]
            resources[(tier, system, resource)] = (
                None
                if any(value is None for value in values)
                else np.asarray(values, dtype=float)[indexes].mean(axis=1)
            )
    comparisons: list[ParetoComparison] = []
    tiers = sorted({tier for tier, _system in combinations})
    for tier in tiers:
        systems = sorted(
            system
            for candidate_tier, system in combinations
            if candidate_tier == tier
        )
        for candidate in systems:
            for comparator in systems:
                if candidate == comparator:
                    continue
                candidate_accuracy = accuracy[(tier, candidate)]
                comparator_accuracy = accuracy[(tier, comparator)]
                for resource in ("tokens", "cost", "latency"):
                    candidate_resource = resources[(tier, candidate, resource)]
                    comparator_resource = resources[(tier, comparator, resource)]
                    if candidate_resource is None or comparator_resource is None:
                        probability = None
                    else:
                        dominates = (
                            (candidate_accuracy >= comparator_accuracy)
                            & (candidate_resource <= comparator_resource)
                            & (
                                (candidate_accuracy > comparator_accuracy)
                                | (candidate_resource < comparator_resource)
                            )
                        )
                        probability = float(dominates.mean())
                    comparisons.append(
                        ParetoComparison(
                            tier=tier,
                            candidate=candidate,
                            comparator=comparator,
                            resource=resource,
                            dominance_probability=probability,
                            bootstrap_replicates=bootstrap_replicates,
                        )
                    )
    return ParetoAnalysis(comparisons=tuple(comparisons))


# Import-only bridge for the Task 5 CLI rebuild. It fails closed so the
# superseded analysis workflow cannot emit empirical artifacts.
def analyze_run(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError(
        "the legacy analyze workflow was removed; migrate to the canonical Task 5 workflow"
    )
