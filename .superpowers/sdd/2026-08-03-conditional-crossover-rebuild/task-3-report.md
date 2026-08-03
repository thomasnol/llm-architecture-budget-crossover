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
