# Task 2 report: systems and execution engine

## Status

Implemented the canonical Task 2 systems and execution engine on
`codex/conditional-crossover-rebuild`, based on Task 1 commit
`e25b6b78726959537f74635ebe066fecca270309`.

## Changes

- Replaced the legacy HMDA policy orchestration in `src/budget_crossover/systems.py` with:
  - a small asynchronous `CompletionClient` protocol that obtains exact prompt-token counts
    before ledger authorization and returns authoritative raw usage;
  - one-call `monolith` retrieval/answer behavior;
  - mandatory-planning, sequential candidate checking, first-pass acceptance, bounded repair,
    recheck, and no-draft exhaustion behavior for `verified_search`;
  - full affordable candidate generation and stable normalized plurality selection for the
    exploratory `unverified_search` ablation;
  - a shared configured model, core instructions, evidence serialization, candidate schema,
    and exact 128/256 output caps;
  - complete typed mechanism traces with SHA-256 query hashes, retrieval identities, candidate
    opportunities, check results, repair/acceptance state, enriched call events, realized usage,
    and frozen exit reasons.
- Rebuilt `src/budget_crossover/runner.py` around:
  - immutable `(case_id, system, tier, repetition)` keys;
  - deterministic case/repetition/tier/system interleaving;
  - bounded asynchronous workers;
  - append-only terminal result and infrastructure-attempt logs;
  - resume skipping and duplicate-result-log rejection;
  - retryable versus permanent infrastructure classification;
  - three-equivalent-permanent-error circuit breaking, restored from the attempt log on resume.
- Tightened `CellResult` system/tier/status and `MechanismTrace.exit_reason` to frozen typed
  vocabularies in `src/budget_crossover/models.py`.
- Replaced legacy orchestration/runner tests with focused canonical Task 2 tests in
  `tests/test_systems.py` and `tests/test_runner.py`.
- Kept only narrow import compatibility for untouched later-task modules:
  `PROMPT_REVISION`, legacy path helpers, and `execute_generation = execute_cells`. No old
  architecture branch or config-driven execution behavior was restored.

## Files changed

- `src/budget_crossover/models.py`
- `src/budget_crossover/systems.py`
- `src/budget_crossover/runner.py`
- `tests/test_systems.py`
- `tests/test_runner.py`
- `.superpowers/sdd/2026-08-03-conditional-crossover-rebuild/task-2-report.md`

## TDD evidence

Each production behavior was introduced from a focused failure. The meaningful RED commands and
observed failures were:

1. `.venv/bin/pytest -q tests/test_systems.py::test_monolith_retrieves_tier_limit_and_makes_one_capped_answer_call`
   - RED: collection failed because canonical `CORE_INSTRUCTIONS` did not exist in the legacy
     systems module.
   - GREEN: `1 passed in 0.11s`.
2. `.venv/bin/pytest -q tests/test_systems.py::test_verified_search_always_plans_at_low_budget_and_accepts_first_pass`
   - RED: `ValueError: unsupported system: verified_search`.
   - GREEN: `1 passed in 0.11s`.
3. `.venv/bin/pytest -q tests/test_systems.py::test_verified_search_checks_sequentially_and_accepts_a_later_candidate`
   - RED: only `planner, candidate_0` occurred; `candidate_1` was missing.
   - GREEN: `1 passed in 0.10s`.
4. `.venv/bin/pytest -q tests/test_systems.py::test_verified_search_repairs_once_from_checker_findings_and_rechecks`
   - RED: repair call was missing after two checker failures.
   - GREEN: `1 passed in 0.10s`.
5. `.venv/bin/pytest -q tests/test_systems.py::test_verified_search_budget_exhaustion_never_returns_a_rejected_draft`
   - RED: `BudgetExceeded` escaped while authorizing `candidate_1`.
   - GREEN: `1 passed in 0.11s`.
6. `.venv/bin/pytest -q tests/test_systems.py::test_unverified_search_generates_full_opportunity_and_uses_stable_plurality`
   - RED: `ValueError: unsupported system: unverified_search`.
   - GREEN: `1 passed in 0.10s`.
7. `.venv/bin/pytest -q tests/test_systems.py::test_mechanism_trace_rejects_exit_reasons_outside_the_frozen_vocabulary`
   - RED: invalid exit reason did not raise validation error.
   - GREEN: `1 passed in 0.10s`.
8. `.venv/bin/pytest -q tests/test_runner.py::test_cell_grid_is_immutable_and_deterministically_interleaved_within_cases`
   - RED: canonical `CellKey`/grid API was absent; legacy runner collection also depended on the
     removed legacy prompt constant.
   - GREEN: `1 passed in 0.10s`.
9. `.venv/bin/pytest -q tests/test_runner.py::test_execution_is_bounded_and_appends_each_terminal_result`
   - RED: canonical `execute_cells` did not exist.
   - GREEN: `1 passed in 0.14s`.
10. `.venv/bin/pytest -q tests/test_runner.py::test_resume_skips_terminal_keys_without_duplicate_rows`
    - RED: the completed monolith key was relaunched and duplicated.
    - GREEN: `1 passed in 0.12s`.
11. `.venv/bin/pytest -q tests/test_runner.py::test_infrastructure_errors_are_unscored_attempts_but_architecture_failures_are_terminal`
    - RED: typed `InfrastructureAttempt` did not exist. The first implementation rerun exposed
      and then removed a stale `del attempts_path` error.
    - GREEN: `1 passed in 0.11s`.
12. `.venv/bin/pytest -q tests/test_runner.py::test_three_equivalent_permanent_errors_open_and_resume_the_circuit`
    - RED: all six cells launched instead of stopping after three permanent errors.
    - GREEN: `1 passed in 0.13s`.
13. `.venv/bin/pytest -q tests/test_systems.py::test_unaffordable_initial_calls_are_terminal_architecture_failures`
    - RED: both monolith and verified planner authorization raised `BudgetExceeded` instead of
      returning terminal architecture failures.
    - GREEN: `2 passed in 0.11s`.
14. `.venv/bin/pytest -q tests/test_systems.py::test_unaffordable_repair_is_terminal_and_never_returns_a_rejected_draft`
    - RED: repair authorization raised `BudgetExceeded`.
    - GREEN: `1 passed in 0.11s`.
15. `.venv/bin/pytest -q tests/test_systems.py::test_malformed_candidate_is_counted_and_scores_as_invalid_output`
    - RED: trace reported candidate count `0` and `checker_exhausted` after one malformed
      candidate call.
    - GREEN (with repair/plurality regressions): `3 passed in 0.10s`.
16. `.venv/bin/pytest -q tests/test_systems.py::test_cell_results_reject_values_outside_the_canonical_execution_vocabulary`
    - RED: invalid system, tier, and status values all validated successfully.
    - GREEN: `3 passed in 0.10s`.
17. `.venv/bin/pytest -q tests/test_systems.py::test_invalid_plan_does_not_claim_that_retrieval_occurred`
    - RED: invalid-plan trace falsely reported all evidence IDs as pre-truncation retrieval IDs.
    - GREEN: `1 passed in 0.10s`.

Focused consolidation after the edge fixes:

```text
$ .venv/bin/pytest -q tests/test_systems.py tests/test_runner.py
.....................                                                    [100%]
21 passed in 0.25s
```

## Full verification

Initial full-suite verification correctly exposed import-only compatibility gaps after replacing
the legacy modules:

- first run: eight collection errors from the removed legacy `PROMPT_REVISION` and path helpers;
- second run: one collection error from the removed `execute_generation` import.

Minimal import bridges were added without restoring old behavior. A subsequent full run passed
`106 passed in 9.15s`. Ruff then identified two `B009` constant-`getattr` style errors in the
compatibility path helpers; both were corrected.

Fresh verification after all self-review fixes:

```text
$ .venv/bin/pytest -q
........................................................................ [ 63%]
.........................................                                [100%]
113 passed in 9.15s

$ .venv/bin/ruff check .
All checks passed!

$ git diff --check
# no output; exit 0
```

## Self-review

- Confirmed no hidden-label type or scorer is imported by systems or runner.
- Confirmed every model request uses the configured model and shared core instructions.
- Confirmed monolith has one model call, no checker invocation, and no repair branch.
- Confirmed verified planning is unconditional, candidates are sequential, every parsed candidate
  is checked, first pass stops generation, repair is at most once, and neither exhaustion nor
  checker failure returns a rejected candidate.
- Confirmed unverified search never receives checker results and plurality uses normalized
  value/unit/entity/period with stable first-candidate tie breaking.
- Confirmed authoritative prompt plus completion usage is committed into `BudgetLedger`; missing,
  mismatched, or over-reserved usage remains a protocol/infrastructure exception rather than a
  scored result.
- Confirmed trace candidate count means actual candidate calls, including malformed output, while
  accepted indices refer to original candidate opportunity order.
- Confirmed invalid planning reports no retrieval activity.
- Confirmed architecture failures are persisted as terminal `CellResult` rows, while transport and
  gateway errors are append-only `InfrastructureAttempt` rows and remain resumable.
- Confirmed existing duplicate terminal keys abort resume rather than silently corrupting the grid.
- Confirmed circuit state is reconstructed from permanent error signatures in the attempt log.

## Concerns / handoff notes

- `CompletionClient` intentionally requires an exact asynchronous prompt-token count before each
  reservation. The existing `GatewayClient` transport remains tested and unchanged but does not
  yet implement this protocol directly; the later canonical CLI/config integration must supply a
  gateway tokenizer/counting adapter rather than falling back to character estimates.
- The narrow `execute_generation = execute_cells` bridge preserves imports only. Legacy callers
  passing HMDA `ExperimentConfig` arguments are intentionally not supported; later tasks should
  migrate CLI/config/manifest wiring to canonical cell/tier arguments.
- With concurrency greater than one, calls already in flight when the third permanent signature is
  observed cannot be unlaunched. The engine stops dequeuing new cells immediately; the exact-three
  test uses one worker to establish the circuit threshold deterministically.
