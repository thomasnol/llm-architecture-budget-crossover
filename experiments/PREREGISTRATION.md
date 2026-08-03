# Preregistration: token-budget crossovers in financial reasoning

Frozen protocol date: 2026-08-03. No canonical empirical main run was available
at freeze. Repository tests and deterministic offline execution are not model
evaluations.

Paper title: **When Extra Inference Structure Pays: A Preregistered Test of
Token-Budget Crossovers in Financial Reasoning**.

## 1. Confirmatory question and hypothesis

The primary comparison holds the underlying model fixed and asks whether the
value of verified inference structure changes with a hard case-level token
budget.

For each independent source document and tier, define the paired binary
difference as correctness of `verified_search` minus correctness of `monolith`.
The conditional hypothesis requires both:

1. low tier: `verified_search - monolith < 0`;
2. high tier: `verified_search - monolith > 0`.

Each component must pass its preregistered one-sided exact McNemar test at
alpha .05. This is an intersection-union test; there is no multiplicity discount
for requiring both components. Equality is not a crossover. High-tier
superiority without low-tier harm is `threshold_benefit_only`, not confirmation.

The five-point SESOI is the design alternative for power and interpretation. It
is not automatically a proven margin. The hypothesis may fail at either or both
endpoints. Non-significance is not equivalence.

## 2. Primary data and lineage

Primary data are pinned local snapshots of FinQA and TAT-QA. Acquisition is
separate from preparation; preparation makes no network request.

Eligibility requires:

- numeric arithmetic or count question supported by a safely executable gold
  program/derivation;
- agreement between executed derivation and annotated answer;
- every operand located in public table/text evidence;
- unambiguous unit and scale;
- unique source document/context;
- no unsupported operation or executable syntax.

At most one case is selected per source document/context. Public cases contain
only case ID, dataset, document ID, question, public evidence, stratum, and
descriptive metadata. Hidden labels contain a typed answer specification, gold
derivation, gold support IDs, and source lineage. Labels join only after
generation.

`headroom` requires at least two derivation operations and at least one required
item ranked 3-12 by the frozen baseline retriever. `easy_control` requires one
operation with all required evidence in the first two results.

Document-disjoint balanced splits are:

- development: 100;
- operational pilot: 60;
- hard main: up to 1,000;
- easy reserve: 100.

Preparation aborts on checksum mismatch, quota shortfall, eligibility shortfall,
document overlap, or public-label leakage. It never relaxes a quota silently.

## 3. Systems

All calls use exactly `gpt-5.4-mini`, the same core instructions, evidence
format, candidate schema, and deterministic retriever. Model or deployment
substitution is prohibited.

### Monolith

- query is the question;
- retrieves exactly the tier evidence limit;
- one answer call, maximum 256 output tokens;
- no checker and no revision.

### Verified search

- mandatory planner, maximum 128 output tokens;
- up to the tier planned-query count;
- deterministic union retrieval;
- sequential candidates, each maximum 256 output tokens;
- every candidate receives a label-blind provenance/arithmetic check;
- first passing candidate is accepted;
- if all candidates fail, middle/high may make at most one 256-token repair
  using checker findings only, then recheck;
- no confidence routing and no unapproved draft fallback.

### Unverified search (exploratory)

- same planner, retrieval, and candidate opportunity;
- generates every affordable candidate;
- never receives checker feedback;
- normalized plurality with stable candidate-order tie break.

Each result records planned/actual queries and hashes, retrieval IDs before and
after truncation, candidate opportunities, checks, repair, accepted index,
answer change, call usage, realized total, and frozen exit reason.

## 4. Scoring and failure semantics

Scoring parses only the strict candidate schema. It does not extract a first or
last number from prose. Decimal scoring normalizes signs, parentheses,
currencies, percent, ones, thousand, million, and billion; checks unit, entity,
and period when specified; and uses frozen absolute/relative tolerances no
looser than 0.000001.

Invalid output, refusal, architecture-caused tool failure, budget exhaustion,
and abstention score incorrect under intention-to-treat. Infrastructure failure
is not scored and triggers whole matched-block handling. No architecture failure
is silently dropped.

## 5. Resource intervention

Hard budget is the sum of authoritative prompt and completion usage over all
calls in the cell. Tool calls, CPU time, wall time, latency, and dollar cost are
separate telemetry.

Frozen initial tiers:

| Tier | Tokens | Retrieval | Queries | Candidates | Repairs |
|---|---:|---:|---:|---:|---:|
| low | 4,096 | 2 | 1 | 1 | 0 |
| middle | 12,288 | 6 | 2 | 2 | 1 |
| high | 32,768 | 12 | 4 | 4 | 1 |

Before a call, the ledger receives an exact tokenizer count for the full chat
prompt and reserves maximum output. It refuses unaffordable calls. Authoritative
usage releases unused reservation. Missing/negative usage, prompt mismatch,
output-reservation overrun, and hard-cap overrun are protocol failures.

Development can advance a tier ceiling only through 8,192, 16,384, 24,576,
32,768, 49,152, and 65,536 when more than 1% of development cases cannot fit
mandatory prompts/actions. Correctness and architecture differences are
unavailable during this selection. Ceilings freeze before pilot.

## 6. Exact gateway preflight

Preflight must establish all of:

- requested model is exactly `gpt-5.4-mini`;
- response-resolved model is exactly `gpt-5.4-mini`;
- completion is valid strict JSON with exactly `{"status":"ok"}`;
- prompt and completion usage are authoritative and total is their sum;
- exact tokenizer ID and SHA-256 match configuration;
- exact pre-call prompt count equals gateway prompt usage.

Missing local tokenizer support does not authorize an estimator. The configured
gateway tokenizer endpoint must supply the exact frozen tokenizer contract.
Failure stops execution.

## 7. Operational pilot gate

The pilot gate is non-overridable. It reports every component and passes only if
all are true:

- complete exact grid and unique paired cells;
- authoritative usage for every observed cell;
- zero label leakage and budget overruns;
- schema validity at least 99%;
- unresolved external matched blocks at most 1%;
- exact expected mechanism counts;
- all low-tier cases feasible;
- median verified-search token growth at least 20% from low to middle and middle
  to high;
- easy monolith accuracy at least 90%;
- hard monolith accuracy between 30% and 85%;
- checker specificity at least 95%;
- checker sensitivity at least 60%;
- correct-to-wrong repair at most 5%;
- at least 20% of checker-detected wrong first drafts corrected.

Any failed component blocks main execution. There is no `--force` path.

## 8. Blinded power and allocation

Internal-pilot sizing uses low/high paired discordance with architecture identity
and direction masked. Exact one-sided McNemar power targets 90% for a five-point
alternative. Repetitions do not increase independent N.

- required hard N <= 900: allocate 900 hard plus 100 easy;
- required hard N 901-1,000: allocate required hard and reduce easy so total is
  1,000;
- required hard N > 1,000: stop underpowered without unblinding.

## 9. Confirmatory inference

Intention-to-treat scoring includes all architecture outcomes; whole unresolved
external matched blocks are excluded. The independent unit is source document
or context.

For low and high endpoints report:

- paired accuracy difference;
- improved and regressed discordant cells;
- one-sided exact McNemar p-value;
- one-sided bound in the tested direction;
- two-sided document-cluster bootstrap interval;
- five-point SESOI interpretation.

Strict crossover confirmation requires observed endpoint reversal plus rejection
of both directional tests. A transition is estimated only after endpoint
reversal.

The case/document cluster bootstrap carries every system and tier together.
Non-crossing replicates are retained. Report crossing support, non-crossing mass,
conditional crossing interval, and a confidence set including no crossing when
applicable. Exploratory families use Holm or simultaneous cluster bands.
Pareto dominance probabilities are resource-specific for tokens, cost, and
latency.

## 10. FinanceComplexQA exploratory boundary

Use only the pinned Pro/English/Numerical-Comparison subset, deduplicated by
canonical question/document identity, with expected count 113. Exclude overall,
evaluation, alternate-language, and alternate-scene duplicates. Preserve
reference-document lineage.

Boundaries:

1. scorer gold round-trip and adversarial perturbation;
2. reference lineage and leakage audit;
3. oracle-evidence export/model input;
4. reference, planned-query, and production retrieval ladders with pre/post
   truncation recall.

Exploratory system execution requires 100% scorer correctness, 100% reference
linkage, zero leakage, and at least 95% high-tier reference-document recall.
Failure is attributed to scorer, lineage/leakage, model with oracle evidence,
retrieval, or orchestration. FinanceComplexQA is never pooled into confirmation.

## 11. Immutable artifact chain

The main manifest freezes:

- resolved configuration and exact source/artifact hashes;
- exact case IDs, document IDs, strata, and expected CellKey grid;
- prompts and system/checker/retriever/code version hashes;
- exact model, deployment, tokenizer ID/hash;
- retry and failure policy;
- secret-free credential patterns and gateway protocol;
- dependency lock and clean Git commit;
- preflight and pilot-gate hashes.

Mutable counters belong only in `run_state.json`. Every downstream stage verifies
upstream hashes before parsing. Empirical paper prose additionally requires a
complete unique validated grid, authoritative usage, zero protocol violations,
matching manifest/gate hashes, and non-scripted gateway provenance.

## 12. Reporting and current status

Frozen table interfaces cover lineage/rejections, diagnostic boundaries,
resource manipulation, mechanisms, paired effects, failures, domain estimates,
and Pareto status. All denominators and failed cells remain visible.

Protocol-only paper output must say no empirical conclusion is available.
Empirical prose can be generated only from a complete validated manifest whose
gate hashes match. At freeze, no such run exists. The hypothesis is currently
neither supported nor cleanly falsified.
