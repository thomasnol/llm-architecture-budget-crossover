# Conditional Crossover Rebuild Implementation Plan

## Global constraints

- Work only on branch `codex/conditional-crossover-rebuild`.
- Treat the checked-out implementation and historical runs as diagnostic history, not empirical evidence for the new study.
- Rebuild the canonical `budget_crossover` package; do not add version-prefixed parallel modules.
- Use test-driven development for every production behavior: add a focused failing test, record the expected failure, implement the minimum behavior, and record the passing run.
- Primary confirmatory systems are exactly `monolith` and `verified_search`; `unverified_search` is exploratory only.
- Primary datasets are FinQA and TAT-QA with one case per source document/context, hidden labels, executable gold derivations, and outcome-independent strata.
- FinanceComplexQA remains exploratory until its scorer, lineage, leakage, and retrieval gates pass.
- Raw prompt plus completion tokens from authoritative gateway usage are the hard budget. Tool calls, CPU time, wall time, and dollar cost are separate telemetry.
- Equality is never a crossover. Confirmation requires a negative low-budget paired difference and a positive high-budget paired difference, both passing their prespecified directional tests.
- Invalid output, refusal, architecture-caused tool failure, budget exhaustion, or abstention score incorrect under intention-to-treat. Infrastructure failures remain unscored and trigger matched-block handling.
- No empirical claim may be produced without a complete, hash-matching, validated run and passing gate artifact.

## Task 1: Canonical contracts, scoring, retrieval, checking, and budget ledger

Replace the legacy record/schema split with a single typed contract in the canonical package.

Implement immutable, validated models for:

- `PublicCase`: case ID, dataset, document ID, question, public evidence corpus, stratum, and descriptive metadata only.
- `HiddenLabel`: case ID, typed `AnswerSpec`, gold derivation, gold support IDs, and source lineage.
- `AnswerSpec`: decimal value, unit, scale, entity, period, absolute tolerance, and relative tolerance.
- `EvidenceItem`: stable ID, document ID, kind, text, table/header metadata, and ordinal.
- `Candidate`: strict value/unit/scale/entity/period/expression/citations schema.
- `CheckResult`, `Usage`, `Reservation`, `CallEvent`, `MechanismTrace`, `CellResult`, and immutable `RunManifest` primitives.

Implement deterministic numeric scoring that never extracts a first or last number from prose. It must parse only the strict candidate schema, normalize `ones`, `thousand`, `million`, `billion`, percent, currencies, negative parentheses, and decimal forms, enforce unit/entity/period compatibility when labels specify them, and use `Decimal` with the label tolerances. Add a gold-oracle serializer used only after the hidden-label join.

Implement a deterministic retriever with stable tie ordering, query-union ranking, table-aware evidence items, per-item truncation that preserves row/header/unit/period metadata, and explicit pre/post truncation IDs. Retrieval must not inspect labels.

Implement a label-blind checker that validates citation existence, operand provenance, safe arithmetic, candidate/expression consistency, unit/scale/entity/period consistency, division-by-zero safety, and rejects unsafe AST nodes or code execution. It may return structured findings but never correctness or a gold value.

Implement `BudgetTier` and `BudgetLedger`. The ledger authorizes a call from exact prompt-token count plus maximum output reservation, refuses unaffordable calls, releases unused reservation on commit, sums authoritative prompt and completion usage, rejects missing/negative/over-reserved usage, and flags any hard-cap overrun as a protocol violation. Provide initial tiers:

- low: 4,096 tokens, retrieval 2, one planned query, one candidate, zero repairs;
- middle: 12,288 tokens, retrieval 6, two planned queries, two candidates, one repair;
- high: 32,768 tokens, retrieval 12, four planned queries, four candidates, one repair.

Test scoring adversarially across trailing years, multiple numbers, signs, percentages, currency/scale, tolerances, entity and period mismatches. Test retrieval determinism and metadata preservation. Test checker valid/invalid arithmetic, fabricated citations, unsupported operands, unsafe syntax, scale mismatch, and division by zero. Test exact ledger reservation/commit/exhaustion/overrun behavior.

## Task 2: Monolith, verified search, exploratory ablation, and execution engine

Build the three systems against the Task 1 interfaces and a small asynchronous `CompletionClient` protocol. All systems use the same configured model, core instructions, evidence format, and candidate schema.

`monolith`:

- creates one direct question query;
- retrieves exactly the tier evidence limit;
- makes exactly one answer call capped at 256 output tokens;
- never calls the checker or revises.

`verified_search`:

- always makes a planner call capped at 128 output tokens;
- parses structured decomposition steps and up to the tier query count;
- retrieves deterministically using those queries;
- generates candidates sequentially, each capped at 256 tokens;
- checks every candidate and stops at the first pass;
- at middle/high, if all candidates fail and the tier permits it, performs at most one 256-token repair using only checker findings, then rechecks it;
- never routes on self-reported confidence and never returns an unapproved draft after exhaustion.

`unverified_search` is exploratory. It uses the same planner, retrieval, and tier candidate opportunity, generates the full candidate count when affordable, and chooses a normalized plurality answer with a stable candidate-order tie break. It never receives checker feedback.

Every system must emit a complete typed mechanism trace: planned/actual queries, query hashes, retrieval IDs before/after truncation, candidate cap/count, checks/results, repair, accepted index, answer change, per-call usage, realized total, and one of the frozen exit reasons.

Rebuild the resumable execution engine around immutable cell keys `(case_id, system, tier, repetition)`, deterministic interleaving within cases, bounded concurrency, append-only attempt logs, duplicate prevention, and three-permanent-error circuit breaking. Infrastructure errors are attempts, not scored results. Architecture failures become terminal scored statuses. Preserve only gateway transport/retry/concurrency behavior whose tests still establish its contract.

Test one-call monolith behavior, mandatory verified planning at low budget, early acceptance, later candidate acceptance, repair/recheck, budget exhaustion with no draft fallback, full unverified candidate generation/plurality, identical initial model/evidence contract, trace completeness, deterministic grid order, resume without duplication, and infrastructure-versus-architecture failure semantics.

## Task 3: FinQA/TAT-QA preparation and FinanceComplexQA diagnostic boundaries

Replace the HMDA/domain registry preparation path with dataset adapters that consume pinned local source snapshots. Network download is a separate explicit acquisition step; preparation itself must be deterministic and offline once snapshots exist.

FinQA adapter:

- accept numeric questions whose supported gold program safely executes to the annotated answer;
- locate every operand in supporting table/text evidence;
- reject unsupported operations, ambiguous scales, missing evidence, duplicate documents, or mismatched programs.

TAT-QA adapter:

- accept arithmetic/count questions with safely executable derivations;
- locate every operand in the associated table/text context;
- apply the same rejection and lineage rules.

Use at most one selected question per source document/context. Build `headroom` from at least two derivation operations plus at least one required evidence item ranked 3-12 by the frozen baseline retriever. Build `easy_control` from one operation with all required evidence in the first two results. Splits are deterministic and document-disjoint: 100 development, 60 operational pilot, up to 1,000 balanced hard main cases, and 100 balanced easy reserves. Emit public cases, hidden labels, rejection ledger, lineage/profile report, and source/artifact hashes. Abort rather than silently relaxing quotas or eligibility.

Implement a FinanceComplexQA diagnostic adapter for the pinned Pro/English/Numerical-Comparison subset, deduplicated by canonical question/document identity and expected to contain 113 cases. Abort with a discrepancy report when the pinned snapshot count differs. Exclude overall/evaluation/alternate-language/alternate-scene duplicates. Preserve reference-document lineage.

Implement the four diagnostic boundaries:

1. scorer oracle with gold round-trip and adversarial perturbations;
2. evidence-lineage and leakage audit;
3. oracle-evidence case export/run input;
4. reference, planned-query, and production retrieval ladders with pre/post truncation recall.

The exploratory system-run gate requires 100% gold scorer correctness, 100% reference-document linkage, zero leakage, and at least 95% high-tier reference-document recall. Emit a machine-readable boundary report that attributes failure to scorer, lineage/leakage, model-with-oracle-evidence, retrieval, or orchestration; never pool this domain into confirmation.

Test safe derivation execution, rejection reasons, deterministic one-document sampling, split disjointness, exact quotas on synthetic fixtures, zero public-label leakage, source/hash changes, headroom/easy classification, FinanceComplex deduplication/count discrepancy, and each diagnostic gate.

## Task 4: Calibration, validation, power, crossover inference, and Pareto analysis

Implement outcome-free development calibration. A tier ceiling may only advance through `8192, 16384, 24576, 32768, 49152, 65536` when more than 1% of development cases cannot fit mandatory prompts/actions. Accuracy, correctness, and between-system differences must be unavailable to calibration. Freeze selected ceilings before pilot.

Implement operational gates for complete grids, unique paired cells, authoritative usage, zero label leakage and budget overruns, at least 99% schema validity, at most 1% unresolved external matched blocks, exact mechanism counts, low-tier feasibility, at least 20% median verified-search token growth between adjacent tiers, easy monolith accuracy at least 90%, hard monolith accuracy between 30% and 85%, checker specificity at least 95%, checker sensitivity at least 60%, correct-to-wrong repair at most 5%, and correction of at least 20% of checker-detected wrong first drafts. Emit all component values and a non-overridable pass/fail artifact.

Implement blinded internal-pilot sizing using only low/high paired discordance with architecture identity and direction masked. Compute exact one-sided McNemar power for a five-point alternative and 90% component power. Allocate 900 hard plus 100 easy when required N <= 900; allocate required hard and reduce easy to keep total 1,000 for 901-1,000; return an underpowered stop without unblinding when N > 1,000. Repeated generations never increase N.

Implement intention-to-treat exact scoring and matched-block exclusion. Primary confirmation is an intersection-union test: low `verified_search - monolith < 0` and high `> 0`, each one-sided exact McNemar alpha .05. Report ordinary paired differences, one-sided bounds, two-sided intervals, five-point SESOI interpretation tiers, and a threshold-benefit label when only high superiority passes.

Implement case/document-cluster bootstrap carrying all systems and tiers together. Equality is never a crossing; retain non-crossing replicates; report crossing support, conditional crossing interval, and a confidence set including no crossing where applicable. Estimate a transition only after endpoint reversal. Exploratory families use Holm or simultaneous cluster bands. Report Pareto dominance probabilities separately for accuracy versus realized tokens/cost/latency.

Test calibration blindness and ceiling progression, every operational gate boundary, exact McNemar fixtures, sample-size lookup/underpowered stop, tie/no-crossing behavior, retained non-crossing bootstrap mass, matched-block ITT behavior, threshold versus crossover labels, and Pareto dominance.

## Task 5: Immutable workflow, CLI, preregistration, historical audit, and gated paper

Build the linear CLI commands:

1. `prepare`
2. `diagnose-finance-complex`
3. `preflight`
4. `develop`
5. `pilot`
6. `gate`
7. `run`
8. `validate`
9. `analyze`
10. `build-paper`

The immutable run manifest freezes resolved configuration, source and artifact hashes, exact case/document IDs and strata, prompts, systems/checker/retriever versions, model/deployment/tokenizer, retry policy, secret-free credential patterns, dependency lock, clean git commit, expected grid, preflight, and pilot-gate hash. Mutable counters belong only in `run_state.json`. Every downstream command verifies upstream hashes and refuses mismatches. Preflight requires exact `gpt-5.4-mini` resolution, authoritative usage, valid strict JSON, and tokenizer consistency; it must stop rather than substitute a model.

Replace configuration with the three systems, three tiers, frozen failure semantics, default sample cap, and strict scoring tolerances. Remove stale HMDA/domain architecture CLI paths and legacy tests that assert superseded behavior, while retaining historical material in Git history.

Write a detailed historical audit explaining why prior results are not empirical evidence: inert nominal budgets, tie-as-crossover logic, conditional bootstrap intervals, first/last-number scoring, FinanceComplex flattening/truncation/eight-case granularity, correlated same-model agents, scripted smoke outputs, missing empirical main artifacts, and the historical always-escalate router. State that the hypothesis is currently neither supported nor cleanly falsified.

Rewrite the protocol/paper title to `When Extra Inference Structure Pays: A Preregistered Test of Token-Budget Crossovers in Financial Reasoning`. Cover the conditional hypothesis, datasets/lineage, hidden labels, systems/mechanisms, action-backed budgets, gates, power, intersection-union inference, FinanceComplex diagnostics, limitations, and exact result interpretations. Generate table interfaces for lineage/rejections, diagnostic boundaries, resource manipulation, mechanisms, paired effects, failures, domain estimates, and Pareto status. A protocol-only build must say no empirical conclusion is available; empirical prose must be generated only from a complete validated manifest whose gate hashes match.

Add an offline end-to-end fixture covering prepare through paper build with a deterministic completion client. The fixture must be clearly marked non-empirical and must prove that incomplete grids, failed gates, hash mismatches, scripted results, and protocol violations cannot produce empirical claims. Update the README with the new workflow and scientific status.

Run the full test suite, linter, type/static checks configured by the project, offline smoke workflow, and paper build. Commit the complete canonical rebuild only after all required checks are clean.
