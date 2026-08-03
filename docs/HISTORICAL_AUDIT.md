# Historical audit: why prior repository results are not empirical evidence

## Scope and conclusion

This audit concerns implementations and artifacts that preceded the canonical
conditional-crossover rebuild. They remain available in Git history for
reproducibility and diagnosis, but they do not answer the current hypothesis.
The current hypothesis is neither supported nor cleanly falsified.

The present estimand asks whether `verified_search - monolith` is negative at a
low action-backed token budget and positive at a high action-backed token
budget, using document-independent FinQA/TAT-QA cases, strict candidate scoring,
authoritative usage, a non-overridable pilot gate, and a complete immutable
cell grid. Historical artifacts fail one or more of these defining conditions.

## 1. Nominal budgets were inert

Earlier workflows labeled cells with nominal budgets, but the labels did not
reliably change the actions available to an architecture. Prompt content,
retrieval depth, candidate opportunities, checking, and routing could remain
the same across budget labels. A completion ceiling is not a resource
intervention when the system neither reserves exact prompt tokens nor changes
its feasible action sequence. Those runs cannot estimate a crossover caused by
budget.

The rebuild uses an exact tokenizer before each call, reserves prompt plus
maximum output, refuses unaffordable calls, commits authoritative usage, and
freezes tier-specific retrieval, query, candidate, and repair opportunities.

## 2. Equality was treated as a crossover

Historical crossover logic could count a tie or a boundary equality as a sign
change. That convention manufactures crossings when the paired effect is zero.
The current definition is strict: the low endpoint must be negative and the
high endpoint positive. Equality is never a crossover.

## 3. Bootstrap intervals were conditional on crossing

Earlier reports summarized crossing locations only among bootstrap replicates
that crossed. Omitting non-crossing replicates hides uncertainty about whether a
crossing exists at all. A narrow conditional interval can coexist with very low
crossing support.

The rebuilt analysis retains every document-cluster replicate, reports crossing
support and non-crossing mass, labels the numeric interval as conditional, and
includes no crossing in the confidence set whenever any retained replicate does
not cross.

## 4. Scoring extracted a first or last number

Some historical scoring paths extracted the first or last numeric substring
from free prose. Years, row numbers, percentages, and explanatory calculations
could therefore be mistaken for the answer. Unit, scale, entity, period, sign,
and tolerance semantics were not reliably enforced.

Canonical scoring accepts only a strict candidate object and uses decimal
arithmetic, explicit normalization, and typed compatibility checks. Historical
scores produced by substring extraction are not comparable.

## 5. FinanceComplexQA lost lineage and granularity

Historical FinanceComplex handling flattened reference documents, truncated
content without preserving pre/post retrieval identity, and reduced evaluation
to an eight-case granularity. This obscured which source document supported an
operand, whether truncation removed required evidence, and whether duplicate or
alternate records entered the sample. Eight cases also cannot support the
current confirmatory inference.

FinanceComplexQA now remains exploratory behind a 113-case pinned-snapshot
count check, canonical deduplication, reference-document lineage, leakage audit,
gold scorer perturbations, oracle-evidence export, and retrieval ladders with
pre/post truncation recall.

## 6. Same-model agents had correlated errors

Historical multi-agent descriptions sometimes treated additional roles as if
they supplied independent votes. Agents using the same base model, prompt
family, and evidence have correlated errors. Majority agreement is not an
independent replication and cannot by itself validate an answer.

The rebuilt study treats same-model correlation as part of the architecture.
Verified search earns credit only through observable retrieval and a
label-blind arithmetic/provenance checker. Unverified plurality is exploratory.

## 7. Smoke outputs were scripted

The historical smoke pipeline returned scripted completions. Scripted outputs
are valuable for exercising serialization, resumption, and analysis code, but
they are not observations of model behavior. Any table or plot generated from
those outputs is software evidence only.

The current offline fixture is irreversibly marked non-empirical in preflight,
stage receipts, manifest, validation, analysis, and paper status. Even a
complete passing scripted grid cannot enable empirical claims.

## 8. Empirical main artifacts were missing

The repository did not contain a complete authoritative main grid with unique
expected cell keys, matching preflight and gate hashes, hidden-label join,
validated protocol usage, and analysis provenance. In particular, no historical
summary can substitute for missing case-level main artifacts. Absence of a
valid run is not a null result.

The paper gate now requires the complete manifest, non-overridable gate,
validation, and analysis artifacts to agree on their hashes and cell counts.

## 9. The historical router always escalated

An earlier nominally adaptive router effectively escalated every case. That
behavior removes selective routing as a treatment and turns the system into a
fixed multi-call architecture. Its cost and accuracy cannot identify the value
of conditional escalation, and it is not the current verified-search system.

The canonical systems have frozen action counts and exit reasons. Mechanism
tables expose whether planning, candidates, checks, and repair occurred exactly
as prescribed.

## Disposition

Historical source and outputs should remain in Git history, not be migrated into
the new run directory or cited as findings. They are useful for explaining why
the protocol changed and for regression tests of failure modes. They do not
support, weaken, or falsify the current conditional crossover hypothesis.
