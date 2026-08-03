# Task 3 Report: canonical dataset preparation and FinanceComplexQA diagnostics

## Status

Implemented and verified on `codex/conditional-crossover-rebuild` from base
`70511d16093f950f14ca7c12e27b06e5262ba02c`.

## Changes

- Replaced the HMDA implementation in `src/budget_crossover/dataset.py` with deterministic,
  offline adapters for pinned FinQA and TAT-QA JSON snapshots.
- Added a Decimal-only safe derivation evaluator and FinQA linear-program compiler. Calls, names,
  unsafe AST nodes, unsupported operations, division by zero, forward/negative references, and
  malformed syntax fail closed.
- Added FinQA support validation against explicit `gold_inds` and conservative TAT-QA operand
  localization. Unannotated operands that occur in multiple evidence items reject as
  `ambiguous_evidence` instead of selecting a location heuristically.
- Added exact answer/scale validation, categorical rejection lineage, content/document
  deduplication, deterministic seeded one-question-per-document selection, and dataset-namespaced
  document IDs.
- Classified `headroom` only when the derivation has at least two operations and a required support
  item ranks 3-12 under the frozen retriever. Classified `easy_control` only when the derivation has
  one operation and all support ranks are at most two.
- Added exact balanced, document-disjoint split preparation with defaults of 100 development,
  60 operational pilot, up to 1,000 hard main, and 100 easy reserve cases. Configured quotas are
  never relaxed; eligibility shortfalls write `preparation_discrepancy.json` and abort.
- Added separate public and hidden JSONL outputs, a categorical rejection ledger, lineage/profile
  report, source hashes, and artifact hashes. Rejection sidecars do not echo operands, programs,
  gold values, or support annotations.
- Added `src/budget_crossover/diagnostics.py` for the pinned FinanceComplexQA
  Pro/English/Numerical-Comparison diagnostic subset. It excludes overall/evaluation,
  alternate-language, and alternate-scene rows; deduplicates canonical question/reference-document
  identity; preserves reference-document lineage; defaults to exactly 113 cases; and writes a
  discrepancy report before aborting on count mismatch.
- Added scorer gold round-trip plus two adversarial value perturbations per case, reference-lineage
  and recursive public-key leakage audit, public-only oracle-evidence export, and reference/planned/
  production retrieval ladders with pre/post-truncation reference-document recall.
- Added a machine-readable boundary report with explicit scorer, lineage/leakage,
  model-with-oracle-evidence, retrieval, and orchestration attribution. FinanceComplexQA is always
  marked `exploratory_only` and never confirmation-pool eligible. The system-run gate requires
  exactly 100% gold scorer correctness, 100% reference linkage, zero leakage, and at least 95%
  production reference-document recall.
- Kept only fail-closed import bridges for the untouched Task 4-5 analysis/CLI/validation modules;
  calling the old HMDA preparation or validation entry points raises immediately.

Task 2 execution files were not modified.

## TDD evidence

### RED/GREEN 1: safe derivation execution

RED:

```text
$ .venv/bin/pytest -q tests/test_dataset.py
ImportError: cannot import name 'DerivationError' from 'budget_crossover.dataset'
1 error in 0.38s
```

GREEN:

```text
$ .venv/bin/pytest -q tests/test_dataset.py
.....                                                                    [100%]
5 passed in 0.38s
```

### RED/GREEN 2: FinQA/TAT-QA adapters, rejection reasons, strata, and one-doc sampling

RED:

```text
$ .venv/bin/pytest -q tests/test_dataset.py -k 'adapters or selects'
ImportError: cannot import name 'DatasetSnapshot' from 'budget_crossover.dataset'
1 error in 0.42s
```

The first implementation run exposed that the negative synthetic documents were intentionally
content-identical and therefore correctly hit `duplicate_document` before question-specific
reasons. After making only the non-duplicate fixture contexts distinct, GREEN was:

```text
$ .venv/bin/pytest -q tests/test_dataset.py
........                                                                 [100%]
8 passed in 0.13s
```

### RED/GREEN 3: exact splits, artifacts, hashes, and abort boundary

RED:

```text
$ .venv/bin/pytest -q tests/test_dataset.py -k 'preparation or checksum'
ImportError: cannot import name 'PreparationAbort' from 'budget_crossover.dataset'
1 error in 0.14s
```

After implementing exact balanced slices and deterministic artifacts:

```text
$ .venv/bin/pytest -q tests/test_dataset.py
...........                                                              [100%]
11 passed in 0.16s
```

### RED/GREEN 4: FinanceComplexQA adapter and diagnostic boundaries

RED:

```text
$ .venv/bin/pytest -q tests/test_diagnostics.py
ModuleNotFoundError: No module named 'budget_crossover.diagnostics'
1 error in 0.07s
```

GREEN:

```text
$ .venv/bin/pytest -q tests/test_diagnostics.py
.........                                                                [100%]
9 passed in 0.13s
```

### RED/GREEN 5: adversarial and safety hardening

Focused REDs proved that Python negative indexing permitted `#-1`, the scorer emitted only one
perturbation per case, and arbitrary `gold_*` keys were not detected:

```text
$ .venv/bin/pytest -q tests/test_dataset.py::test_derivations_reject_unsafe_or_unsupported_programs \
  tests/test_dataset.py::test_tatqa_accepts_count_questions_with_executable_evidence_backed_derivations \
  tests/test_diagnostics.py::test_scorer_lineage_leakage_and_oracle_evidence_boundaries
2 failed, 5 passed in 0.16s
```

GREEN after rejecting negative references, emitting two independently wrong value perturbations,
and recognizing the broader hidden-key family:

```text
.......                                                                  [100%]
7 passed in 0.13s
......................                                                   [100%]
22 passed in 0.14s
```

### RED/GREEN 6: cross-dataset identity, ambiguous evidence, and rejection leakage

Three independent mutation tests were observed RED before their minimal fixes:

```text
test_document_identity_is_namespaced_across_primary_datasets
E assert 1 == 2

test_tatqa_rejects_ambiguous_operand_locations_instead_of_guessing
E assert (AdaptedCase(...),) == ()

test_adapters_record_specific_rejections_instead_of_relaxing_eligibility
E assert all(rejection.detail is None ...)
```

Focused GREEN evidence:

```text
$ .venv/bin/pytest -q <identity-and-split-slice>
2 passed in 0.11s
$ .venv/bin/pytest -q <ambiguity-and-regression-slice>
3 passed in 0.12s
$ .venv/bin/pytest -q <rejection-ledger-slice>
2 passed in 0.12s
```

### Full-suite import seam

Removing HMDA preparation initially exposed three collection errors because the untouched Task 4-5
modules still imported the old names. Import-only bridges were added; legacy preparation/validation
calls raise rather than restoring behavior. The next full suite passed 136 tests. Subsequent
hardening added four more tests, included in final verification below.

## Final verification

Fresh final commands after all code and test changes:

```text
$ .venv/bin/pytest -q tests/test_dataset.py tests/test_diagnostics.py
........................                                                 [100%]
24 passed in 0.14s

$ .venv/bin/pytest -q
........................................................................ [ 51%]
....................................................................     [100%]
140 passed in 4.19s

$ .venv/bin/ruff check .
All checks passed!

$ git diff --check
# no output; exit 0
```

## Assumptions and concerns

- No pinned FinQA, TAT-QA, or FinanceComplexQA source snapshot is present in this checkout, so
  adapters are validated offline against synthetic fixtures shaped like canonical FinQA
  (`pre_text`/`table`/`post_text`/`qa`) and TAT-QA (`table`/`paragraphs`/`questions`) records. The
  FinanceComplexQA fixture uses explicit `subset`, `language`, `category`, `scene`, `scope`, `split`,
  `documents`, and `reference_document_ids` fields. A real pinned FinanceComplexQA snapshot should
  receive a schema-characterization run before acquisition hashes are frozen.
- Development, pilot, and main quotas are interpreted as balanced FinQA/TAT-QA `headroom` cases;
  easy reserve is balanced `easy_control`. All configured totals must be even.
- Program/answer agreement is deliberately conservative and exact in displayed units. Unexpected
  source formatting, ambiguous operand locations, or rounded derivations reject and surface as a
  quota/count discrepancy rather than being normalized heuristically or silently admitted.

## Fix round 1: critical and important independent-review findings

Base commit: `eee6bccfe5c5208ce291ac1c53caf268dbaeb126`.

### Changes

- Implemented explicit TAT-QA count semantics. A count question now requires a nonempty `facts`
  list whose normalized exact text uniquely identifies each counted evidence item. Missing and
  ambiguous facts reject; the annotated answer must be an integer equal to the number of unique
  cited items. The canonical derivation is strict `count("evidence-id", ...)`, and the documented
  aggregation-operation count is `max(1, n - 1)`, so two facts can be `easy_control` while three or
  more can satisfy `headroom` only through the same frozen-retrieval rank rule.
- Added narrow checker and candidate-prompt support for strict count expressions. The checker
  parses only a `count` call containing unique string evidence IDs, requires the expression IDs to
  match citations exactly, and compares the resulting count with the strict numeric candidate.
- Restricted FinanceComplex operand localization to declared reference-document evidence. The
  lineage audit independently re-executes the hidden derivation and verifies its operands inside
  references; oracle-evidence export refuses support IDs sourced from distractors.
- Made retrieval ladders explicitly low/middle/high. Reference and planned queries require exact
  case coverage, production requires exactly all three known tiers and exact case coverage within
  every tier, tier limits must be exact and positive, and the report preserves pre/post recall for
  every ladder/tier. The system-run gate now reads only high-tier production post-truncation recall.
- Before a quota-shortfall abort, primary preparation now emits the categorical rejection ledger,
  an aborted diagnostic lineage/profile, the discrepancy report, and source/artifact hashes.
  Sidecars remain free of answers, programs, operands, and support annotations.
- FinQA recognized `gold_inds` keys now require normalized exact agreement with annotated text.
  Unknown-key exact-text fallback rejects zero matches and ambiguous duplicate matches.
- Scorer-oracle diagnostics now add wrong scale for every case and wrong unit, entity, and period
  whenever specified, in addition to two numeric perturbations. Per-field totals/rejections are
  emitted and the boundary passes only when every adversary is rejected.
- Fixed same-basename source-hash collisions with a stable content-qualified source identifier,
  reused consistently by source hashes, rejection rows, and hidden lineage.

### RED/GREEN evidence

#### Explicit count semantics and checker

RED:

```text
$ .venv/bin/pytest -q \
  tests/test_dataset.py::test_tatqa_accepts_count_questions_with_executable_evidence_backed_derivations \
  tests/test_dataset.py::test_tatqa_count_rejects_missing_or_ambiguous_counted_evidence \
  tests/test_checking.py::test_checker_validates_explicit_counted_evidence_without_numeric_operands
FFF                                                                      [100%]
3 failed in 0.16s
```

The candidate prompt then received its own RED because it still described arithmetic only:

```text
$ .venv/bin/pytest -q \
  tests/test_systems.py::test_planner_system_instructions_do_not_conflict_with_its_plan_schema
F                                                                        [100%]
1 failed in 0.15s
```

GREEN:

```text
$ .venv/bin/pytest -q <count-dataset-checker-prompt-slice>
....                                                                     [100%]
4 passed in 0.11s
```

The separate three-fact fixture also passed and proves the `n - 1` headroom rule.

#### FinQA support annotation validation

RED:

```text
$ .venv/bin/pytest -q \
  tests/test_dataset.py::test_finqa_validates_support_keys_and_rejects_ambiguous_text_fallback
F                                                                        [100%]
E assert (AdaptedCase(...), AdaptedCase(...)) == ()
1 failed in 0.15s
```

GREEN with existing adapter regressions:

```text
...                                                                      [100%]
3 passed in 0.11s
```

#### FinanceComplex reference-only support

RED:

```text
$ .venv/bin/pytest -q \
  tests/test_diagnostics.py::test_financecomplex_rejects_operands_supported_only_by_distractor_documents \
  tests/test_diagnostics.py::test_lineage_audit_and_oracle_export_reject_support_outside_references
FF                                                                       [100%]
2 failed in 0.15s
```

After restricting preparation/export, a stronger audit mutation (declared reference exists but
lacks required operands while a distractor contains them) was observed RED:

```text
E assert 1.0 == 0.0
1 failed, 1 passed in 0.17s
```

GREEN with scorer/lineage regressions:

```text
...                                                                      [100%]
3 passed in 0.11s
..                                                                       [100%]
2 passed in 0.12s
```

#### Field-wise scorer adversaries

RED:

```text
$ .venv/bin/pytest -q \
  tests/test_diagnostics.py::test_scorer_lineage_leakage_and_oracle_evidence_boundaries \
  tests/test_diagnostics.py::test_scorer_oracle_perturbs_every_specified_answer_field
FF                                                                       [100%]
2 failed in 0.15s
```

GREEN:

```text
..                                                                       [100%]
2 passed in 0.11s
```

#### Tiered retrieval and exact coverage

RED:

```text
$ .venv/bin/pytest -q <tiered-ladder-and-boundary-slice>
FFF....F                                                                 [100%]
4 failed, 4 passed in 0.18s
```

Failures were the missing `tier_limits` API, the old un-tiered production structure, and the gate
reading the old un-tiered recall field. GREEN:

```text
........                                                                 [100%]
8 passed in 0.12s
```

This includes missing reference coverage, extra planned coverage, low/middle-only production,
unknown production tier, missing high-tier case, 0.94 high-tier failure, and 0.95 high-tier pass.

#### Quota-abort diagnostic artifacts

RED:

```text
$ .venv/bin/pytest -q \
  tests/test_dataset.py::test_preparation_aborts_with_machine_readable_shortfalls
F                                                                        [100%]
FileNotFoundError: .../rejections.jsonl
1 failed in 0.17s
```

GREEN with the successful preparation regression:

```text
..                                                                       [100%]
2 passed in 0.11s
```

#### Content-qualified source identifiers

RED:

```text
$ .venv/bin/pytest -q \
  tests/test_dataset.py::test_source_identifiers_distinguish_same_basenames_and_join_lineage
F                                                                        [100%]
E assert 1 == 2
1 failed in 0.15s
```

GREEN with checksum repinning regression:

```text
..                                                                       [100%]
2 passed in 0.12s
```

### Fix-round final verification

```text
$ .venv/bin/pytest -q tests/test_dataset.py tests/test_diagnostics.py \
  tests/test_checking.py tests/test_systems.py
......................................................................   [100%]
70 passed in 0.22s

$ .venv/bin/pytest -q
........................................................................ [ 48%]
........................................................................ [ 96%]
.....                                                                    [100%]
149 passed in 4.22s

$ .venv/bin/ruff check .
All checks passed!

$ git diff --check
# no output; exit 0
```

### Fix-round concern

No real pinned TAT-QA snapshot is present locally. Explicit count support therefore uses the
actual-like empty arithmetic derivation plus an exact `facts` evidence list in synthetic fixtures.
If the acquired pinned snapshot uses a different count-support field, schema characterization must
map that field to the same exact-evidence contract; questions without explicit counted evidence
will reject rather than be inferred heuristically.

## Fix round 2: retrieval tier provenance

### Finding

Tier labels previously lived only in the diagnostic result mapping. The same `RetrievalResult`
could therefore be placed under both `low` and `high`, and the high-tier gate would consume the
relabeled recall without proving which tier, retrieval limit, query list, or public input generated
it.

### RED evidence

The relabeled-low-as-high regression failed before boundary validation:

```text
$ .venv/bin/pytest -q \
  tests/test_diagnostics.py::test_retrieval_ladders_reject_relabelled_tiers_and_requested_k_mismatch
F                                                                        [100%]
E Failed: DID NOT RAISE <class 'ValueError'>
1 failed in 0.15s
```

The stale-query variant then failed until the boundary recomputed the expected query hash:

```text
F                                                                        [100%]
E Failed: DID NOT RAISE <class 'ValueError'>
1 failed in 0.15s
```

The high gate also accepted a recall-only mapping with no validated provenance:

```text
$ .venv/bin/pytest -q \
  tests/test_diagnostics.py::test_boundary_run_gate_rejects_high_recall_without_validated_high_provenance
F                                                                        [100%]
E assert True is False
1 failed in 0.16s
```

### Implementation and GREEN evidence

`RetrievalResult` now stores frozen, validated `tier_id`, `requested_k`, `query_hash`, and
`input_hash` provenance generated by `retrieve()`. It also validates the returned ID sequence and
its relationship to `requested_k`. Empty system retrievals carry the same tier/input provenance.

`retrieval_ladder_boundary` now requires the expected per-tier production query lists and rejects
any result whose tier, configured limit, recomputed semantic query hash, or recomputed complete
public-case hash does not match. Only after all results pass does it emit
`provenance_validated: true`. This covers low-as-high relabeling, wrong `requested_k`, stale query,
and stale public input. The exploratory high gate requires that validated high-tier metadata before
using high recall.

```text
$ .venv/bin/pytest -q tests/test_diagnostics.py
...............                                                          [100%]
15 passed in 0.17s

$ .venv/bin/pytest -q tests/test_retrieval.py tests/test_dataset.py \
  tests/test_diagnostics.py tests/test_systems.py tests/test_checking.py
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 0.24s
```

### Fix-round 2 final verification

```text
$ .venv/bin/pytest -q
........................................................................ [ 47%]
........................................................................ [ 94%]
........                                                                 [100%]
152 passed in 4.49s

$ .venv/bin/ruff check .
All checks passed!

$ git diff --check
# no output; exit 0
```

### Fix-round 2 concern

Production retrieval diagnostics must now supply the exact per-tier query list alongside each
result so provenance can be recomputed. This is an intentional fail-closed API change; stale or
missing query capture is rejected instead of allowing an unverified high-tier gate.
