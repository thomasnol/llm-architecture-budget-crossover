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

## Fix round 1: canonical-invariant hardening

The first review round identified four ways a caller or incomplete run could
change the confirmatory claim after the protocol was frozen. All four are now
closed:

- `confirm_crossover` no longer accepts caller-controlled alpha or SESOI values.
  The confirmatory contract and its endpoint artifacts use literal `.05` values,
  while interval quantiles are derived from the frozen alpha.
- Calibration selections now carry the exact development observations used to
  derive them. Their examined ceiling prefix, terminal selected ceiling, recorded
  rates, and feasibility verdict are recomputed during validation. Freeze rebuilds
  each selection from its serialized data before accepting it.
- Operational-gate input maps, component values, and thresholds are recursively
  immutable. Artifact summaries and all 18 components are recomputed from a
  revalidated typed input snapshot. `model_copy`, direct construction, nested
  forged components, and `model_construct` cannot substitute contradictory cached
  pass summaries.
- The complete-grid gate compares exact expected and observed `CellKey` identities
  and reports deterministic missing/unexpected keys. Duplicate identities remain a
  separate failure. ITT scoring accepts a frozen expected-primary grid and
  materializes absent retained repetition-zero cells as incorrect architecture
  outcomes; unresolved external blocks remain excluded atomically.

### Fix-round RED/GREEN ledger

| Review regression | Observed RED | GREEN evidence |
|---|---|---|
| Frozen confirmatory alpha | caller override at `.10` did not raise | override is rejected and the `.0625` endpoint fixture remains unconfirmed at `.05` |
| Frozen five-point SESOI | caller override at `.20` did not raise | override is rejected and artifacts carry literal `.05` |
| Derived calibration selection | forged copy/direct constructor did not raise | progression, selected ceiling, rates, and feasibility are rederived from observations |
| Calibration freeze revalidation | a `model_construct` selection with a cached `True` verdict did not raise | freeze reconstructs and rejects the forged selection |
| Gate summary integrity | contradictory constructor/copy payloads did not raise | summary fields are derived from freshly evaluated canonical components |
| Gate deep immutability | nested input/component mapping mutation did not raise | all nested maps reject mutation with `TypeError` |
| Gate construct bypass | direct `model_construct` did not raise | artifact `model_construct` routes through full validation |
| Nested forged gate component | an internally consistent cached failure unrelated to the input snapshot validated | all components are recomputed from revalidated typed inputs and the forgery is rejected |
| Exact grid identity | equal cell counts hid one missing and one unexpected cell | `complete_grid` reports both exact keys and fails while uniqueness passes |
| Missing primary ITT cell | `expected_primary_keys` was not accepted and the missing architecture vanished from the denominator | the absent retained primary cell is emitted as incorrect with `missing_architecture_cell` status |

### Fix-round verification

Fresh verification of the fix-round tree:

- `.venv/bin/pytest tests/test_analysis.py tests/test_calibration.py tests/test_validation.py -q`:
  `54 passed in 2.94s`;
- `.venv/bin/pytest -q`: `197 passed in 3.64s`;
- `.venv/bin/ruff check .`: `All checks passed!`;
- `git diff --check`: exit 0 with no output.

## Fix round 2: finite gate telemetry

Operational gate telemetry now fails closed at its typed boundary when any float
is NaN or positive/negative infinity. The same recursive finite-number check
guards all nested `GateComponent.value` and `GateComponent.threshold` data, and
`GateComponent.model_construct` routes through validation rather than exposing a
bypass. This prevents non-finite component payloads and avoids the misleading
artifact-summary error previously caused by `NaN != NaN` during canonical
component recomputation.

Finite telemetry that merely misses a gate remains an ordinary operational result:
the verified-search 19% low-to-middle growth fixture emits a failed artifact with
the named growth component rather than raising a protocol validation error.

### Fix-round 2 RED/GREEN ledger

| Regression | Observed RED | GREEN evidence |
|---|---|---|
| Non-finite verified-search medians | NaN and positive infinity were accepted; negative infinity produced only the positivity error | NaN, `+inf`, and `-inf` all raise the explicit finite-telemetry validation error at `OperationalGateInputs` construction |
| Non-finite component values | nested NaN/`+inf`/`-inf` values validated | normal construction and `model_construct` reject all three values |
| Non-finite component thresholds | nested NaN/`+inf`/`-inf` thresholds validated | normal construction and `model_construct` reject all three thresholds |
| Ordinary finite gate failure | existing behavior characterization stayed green | 19% growth emits the expected failed component artifact with finite value `0.19` |

### Fix-round 2 verification

Fresh verification of the fix-round 2 tree:

- `.venv/bin/pytest tests/test_analysis.py tests/test_calibration.py tests/test_validation.py -q`:
  `64 passed in 2.96s`;
- `.venv/bin/pytest -q`: `207 passed in 3.61s`;
- `.venv/bin/ruff check .`: `All checks passed!`;
- `git diff --check`: exit 0 with no output.
