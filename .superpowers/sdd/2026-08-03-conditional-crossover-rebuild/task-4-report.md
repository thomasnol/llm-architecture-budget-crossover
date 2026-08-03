# Task 4 Report: Calibration, Validation, Power, and Crossover Inference

## Scope

Rebuilt the canonical, unversioned modules:

- `src/budget_crossover/calibration.py`
- `src/budget_crossover/validation.py`
- `src/budget_crossover/analysis.py`

The implementation consumes the Task 1–3 immutable `PublicCase`, `HiddenLabel`,
`CellResult`, `SystemName`, and `TierName` contracts. Legacy HMDA analysis,
calibration, and pilot-gate behavior was removed. Narrow import-only bridges remain
for the Task 5 CLI transition, and every bridge raises a migration error before
performing work or exposing outcomes.

## Implemented contracts

### Outcome-free calibration

- `DevelopmentFitObservation` contains only `case_id` and mandatory prompt/action
  token demand; frozen/forbidden extra fields make correctness, accuracy, and
  system-difference data structurally unavailable.
- `select_calibration_ceiling` advances only through
  `8192, 16384, 24576, 32768, 49152, 65536` and only while the observed failure
  rate is strictly greater than 1%.
- Failure above the final ceiling is explicit and cannot be frozen as feasible.
- `freeze_calibration` requires low, middle, and high selections together, refuses
  an already-started pilot, refuses infeasible tiers, and creates the artifact
  exactly once.

### Non-overridable operational gates

`evaluate_operational_gates` emits an immutable artifact containing the complete
input snapshot and 18 named components:

1. complete grid;
2. unique paired cells;
3. authoritative usage;
4. zero label leakage;
5. zero budget overruns;
6. schema validity >=99%;
7. unresolved external matched blocks <=1%;
8. exact mechanism counts;
9. complete low-tier feasibility;
10. low-to-middle verified-search median token growth >=20%;
11. middle-to-high verified-search median token growth >=20%;
12. easy monolith accuracy >=90%;
13. hard monolith accuracy >=30%;
14. hard monolith accuracy <=85%;
15. checker specificity >=95%;
16. checker sensitivity >=60%;
17. correct-to-wrong repair <=5%;
18. correction of checker-detected wrong first drafts >=20%.

The artifact has no override argument and always emits `override_allowed=false`.
Zero checker or repair opportunities fail closed because those mechanisms were not
exercised. External matched blocks are the sole zero-denominator exception: 0/0
passes as no external blocks requiring resolution, and the exception is named in
the component artifact.

### Blinded pilot sizing and exact power

- One-sided exact McNemar tests use the conditional
  `Binomial(discordant, 0.5)` tail.
- Component power integrates over `D ~ Binomial(N, q)` and then the favorable
  discordant count conditional on `D`, using
  `p=(q+0.05)/(2q)`, `q>=0.05`, exact alpha .05 critical regions, and 90% target
  power.
- The literal boundary fixture at `q=.05` gives minimum `N=158`, with power
  `0.8974508517633653` at 157 and `0.9004240718552209` at 158.
- `MaskedPilotDiscordance` accepts only unsigned low/high discordant counts,
  independent documents, and the observed repetition count. Counts cannot be
  multiplied by repetitions.
- Required `N<=900` allocates 900 hard + 100 easy; `901..1000` allocates required
  hard and fills the remaining slots with easy controls; `N>1000` returns an
  underpowered stop with no allocation and without unblinding.

### ITT scoring and primary confirmation

- `score_itt_results` exact-scores canonical candidates against hidden labels;
  missing candidates and architecture failures are incorrect under ITT.
- Unresolved external matched blocks are excluded atomically, never case by case.
- The independent count is unique documents/cases. Repetition zero is marked as
  primary and repeated generations never increase `N`.
- `confirm_crossover` reports ordinary paired differences, exact one-sided
  McNemar results, one-sided bootstrap bounds, two-sided bootstrap intervals, and
  SESOI interpretation tiers.
- Confirmation is the strict intersection-union rule: low verified-search minus
  monolith is negative and passes the one-sided `less` test, while high is positive
  and passes `greater`, each at alpha .05.
- High-only superiority is labeled `threshold_benefit_only`.
- A five-point point estimate is explicitly labeled as margin-not-proven unless
  the relevant one-sided bound supports the five-point margin.

### Cluster bootstrap, multiplicity, and Pareto analysis

- All tiers are carried through the same deterministic document/case resample.
- Equality is never a transition. A transition estimate requires observed endpoint
  reversal and an adjacent strict negative-to-positive segment.
- Every bootstrap replicate is retained. The result reports crossing and
  non-crossing counts, crossing support, a crossing-conditional numeric interval,
  and a confidence set that includes no crossing whenever retained mass is
  non-crossing.
- Exploratory p-value families use the step-down Holm procedure.
- Pareto dominance probabilities resample the complete system-by-tier grid jointly
  and report separate pairwise claims for accuracy versus realized tokens, cost,
  and latency. Missing resource measurements yield `None`, not a substituted claim.

## TDD evidence

All production behavior followed an observed RED before GREEN. Representative
cycles:

| Behavior | Observed RED | GREEN evidence |
|---|---|---|
| Calibration API/progression | missing import, then `[8192]` instead of the required frozen progression | calibration focused suite passed |
| Calibration blindness | forbidden outcome fields did not raise | strict immutable extra-forbid input passed |
| Calibration freeze | missing API, then partial tiers did not raise | write-once three-tier freeze passed |
| Gate artifact | missing canonical API | complete 18-component artifact passed |
| Gate thresholds | all 18 disallowed-boundary cases incorrectly passed | 18/18 boundary cases passed |
| Inclusive boundaries | hard accuracy 85% rejected under temporary strict comparison | inclusive bound passed |
| Zero checker/repair denominators | Pydantic rejected input before emitting a gate artifact | all four mechanism components emit fail-closed results |
| Exact McNemar | missing API | literal `1/64` one-sided tails passed |
| Exact power | `NotImplementedError` | literal `.05**5` fixture passed |
| Sample lookup | missing API | literal N=158 boundary passed |
| Pilot allocation/stop | missing masked model | 886/921/1060 lookup branches passed |
| Matched-block ITT | missing models/API | atomic exclusion and repetition-invariant N passed |
| Strict/threshold labels | missing confirmation API | strict crossover and threshold-only fixtures passed |
| SESOI interpretation | point estimate was mislabeled `direction_not_established` | margin-not-proven tier passed |
| Bootstrap | missing API | retained non-crossing mass and equality/no-crossing passed |
| Multiplicity | missing API | literal Holm values `.03/.06/.06` passed |
| Pareto analysis | missing API | resource-specific 1/0/1 dominance fixture passed |
| Compatibility bridges | CLI collection import failure | every bridge imports and raises a migration error |

## Final verification

Fresh final verification commands and outputs:

- `.venv/bin/pytest -q`: `186 passed in 3.53s`;
- `.venv/bin/ruff check .`: `All checks passed!`;
- `git diff --check`: exit 0 with no output.

## Known handoff concerns

- Task 5 must replace the fail-closed CLI bridge callers with adapters that build
  the canonical typed inputs and persist/hash the resulting artifacts.
- Pilot discordance below five points is rejected as incompatible with the
  prespecified five-point design alternative (`q>=.05`) rather than silently
  clamped or unblinded.
- Crossing intervals are intentionally conditional and must always be presented
  beside support and the no-crossing member of the confidence set.
